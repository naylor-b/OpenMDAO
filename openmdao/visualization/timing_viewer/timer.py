"""
Classes and functions for timing method calls.
"""
import os
import sys
import sqlite3
from collections import defaultdict
from time import perf_counter
from contextlib import contextmanager
from functools import wraps, partial
import numpy as np

import openmdao.utils.hooks as hooks
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.mpi import MPI
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.solvers.solver import Solver
from openmdao.core.problem import _problem_names
from openmdao.visualization.tables.table_builder import generate_table

# can use this to globally turn timing on/off so we can time specific sections of code
_timing_active = False
_total_time = 0.
_timing_managers = {}


def _timing_iter(all_timing_managers):
    # iterates over all of the timing managers and yields all timing info
    for rank, (timing_managers, tot_time, nprobs) in enumerate(all_timing_managers):
        for probname, tmanager in timing_managers.items():
            for sysname, timers in tmanager._timers.items():
                level = len(sysname.split('.')) if sysname else 0
                for t, parallel, nprocs, classname in timers:
                    if t.info.ncalls > 0:
                        fname = t.name.rpartition('.')[2]
                        yield rank, probname, classname, sysname, level, parallel, nprocs, fname,\
                            t.info.ncalls, t.info.min, t.info.max, t.info.total, t.children


def _get_par_child_info(timing_iter, method_info):
    # puts timing info for direct children of parallel groups into a dict.
    parents = {}
    klass, method = method_info

    for rank, probname, classname, sysname, level, parallel, nprocs, func, ncalls, tmin, tmax,\
            ttot, children in timing_iter:
        if not parallel or method != func:
            continue

        avg = ttot / ncalls
        parent = sysname.rpartition('.')[0]
        key = (probname, parent)
        if key not in parents:
            parents[key] = {}

        if sysname not in parents[key]:
            parents[key][sysname] = []

        parents[key][sysname].append((rank, ncalls, avg, tmin, tmax, ttot))

    return parents


class _AttrMatcher(object):
    __slots__ = ['classes', 'matches', 'name']

    def __init__(self, classes, method_name):
        assert isinstance(classes, tuple), \
            f"_AttrMatcher 'classes' arg must be a tuple, not a {type(classes).__name__}."
        assert isinstance(method_name, str), \
            f"_AttrMatcher 'method_name' arg must be a str, not a {type(method_name).__name__}."
        self.classes = classes
        self.matches = {c.__name__ for c in classes}
        self.name = method_name

    def _match_obj(self, obj):
        return isinstance(obj, self.classes)

    def _match_cname(self, class_name):
        return class_name in self.matches

    def __str__(self):
        return ','.join([c.__name__ + '.' + self.name for c in self.classes])

    def __iter__(self):
        yield self.classes
        yield self.name


_timer_methods = {
    'default': [
        _AttrMatcher((Group,), '_solve_nonlinear'),
        _AttrMatcher((System,), '_solve_linear'),
        _AttrMatcher((System,), '_apply_nonlinear'),
        _AttrMatcher((System,), '_apply_linear'),
        _AttrMatcher((System,), '_linearize'),
        _AttrMatcher((ExplicitComponent,), 'compute'),
        _AttrMatcher((ExplicitComponent,), 'compute_partials'),
        _AttrMatcher((ExplicitComponent,), 'compute_jacvec_product'),
        _AttrMatcher((ImplicitComponent,), 'linearize'),
        _AttrMatcher((ImplicitComponent,), 'solve_nonlinear'),
        _AttrMatcher((ImplicitComponent,), 'solve_linear'),
        _AttrMatcher((ImplicitComponent,), 'apply_nonlinear'),
        _AttrMatcher((ImplicitComponent,), 'apply_linear'),
        _AttrMatcher((Solver,), 'solve'),
        _AttrMatcher((Solver,), '_solve'),
        _AttrMatcher((Solver,), '_iter_initialize'),
        _AttrMatcher((Solver,), '_run_apply'),
        _AttrMatcher((Solver,), '_single_iteration'),
    ]
}


class FuncTimerInfo(object):

    __slots__ = ['ncalls', 'min', 'max', 'total']

    def __init__(self):
        self.ncalls = 0
        self.min = 1e99
        self.max = 0
        self.total = 0

    def __iter__(self):
        yield self.ncalls
        yield self.min
        yield self.max
        yield self.total

    def called(self, dt):
        if dt < self.min:
            self.min = dt
        if dt > self.max:
            self.max = dt
        self.total += dt
        self.ncalls += 1


class FuncTimer(object):
    """
    Keep track of execution times for a function.

    Parameters
    ----------
    name : str
        Name of the instance whose methods will be wrapped.
    objname : str
        Name of the object instance if it has one, else class name.
    stack : list
        Call stack.

    Attributes
    ----------
    name : str
        Name of the instance whose methods will be timed.
    objname : str
        Name of the object instance if it has one, else class name.
    stack : list
        Call stack.
    info : FuncTimerInfo
        Keeps track of timing and number of calls.
    """

    def __init__(self, name, objname, stack):
        """
        Initialize data structures.
        """
        self.name = objname + '.' + name if objname else name
        self.stack = stack
        self.children = defaultdict(FuncTimerInfo)
        self.info = FuncTimerInfo()

    def pre(self):
        """
        Record the method start time.
        """
        global _timing_active
        if _timing_active:
            if self.stack and self.stack[-1][0] is self:
                self.stack[-1][1] += 1
            else:
                self.stack.append([self, 1])
            self.start = perf_counter()

    def post(self):
        """
        Update ncalls, tot, min, and max.
        """
        global _timing_active
        if _timing_active:
            dt = perf_counter() - self.start
            self.info.called(dt)
            s = None
            stack = self.stack
            while stack and s is not self:
                s, count = stack.pop()
            if count > 1:
                stack.push([self, count-1])
            if stack:
                stack[-1][0].children[self.name].called(dt)


def _timer_wrap(f, timer):
    """
    Wrap a method to keep track of its execution time.

    Parameters
    ----------
    f : method
        The method being wrapped.
    timer : Timer
        Object to keep track of timing data.

    Returns
    -------
    method
        The wrapper for the give method f.
    """
    def do_timing(*args, **kwargs):
        timer.pre()
        try:
            ret = f(*args, **kwargs)
        finally:
            timer.post()
        return ret

    dec = wraps(f)(do_timing)
    dec._orig_func_ = f
    return dec


class TimingManager(object):
    """
    Keeps track of FuncTimer objects.

    Parameters
    ----------
    options : argparse options or None
        Command line options.

    Attributes
    ----------
    _timers : dict
        Mapping of instance name to FuncTimer lists.
    _par_groups : set
        Set of pathnames of ParallelGroups.
    _par_only : bool
        If True, only instrument direct children of ParallelGroups.
    """

    def __init__(self, options=None):
        """
        Initialize data structures.
        """
        self._timers = {}
        self._par_groups = set()
        self._par_only = options is None or options.view.lower() == 'text'
        self._call_stack = []

    def add_timings(self, name_obj_proc_iter, methods):
        """
        Add FuncTimers for all instances in name_obj_iter and all matching methods.

        Parameters
        ----------
        name_obj_proc_iter : iterator
            Yields (name, object, nprocs) tuples.
        methods : list of str
            List of names of methods to wrap.
        """
        for name, obj, nprocs in name_obj_proc_iter:
            for _type, method_name in methods:
                if isinstance(obj, _type):
                    self.add_timing(name, obj, nprocs, method_name)

    def add_timing(self, name, obj, nprocs, method_name):
        """
        Add a FuncTimer for the given method of the given object.

        Parameters
        ----------
        name : str
            Instance name.
        obj : object
            The instance.
        nprocs : int
            Number of MPI procs given to the object.
        method_name : str
            The name of the method to wrap.
        """
        method = getattr(obj, method_name, None)
        if method is not None:
            try:
                obj_name = obj._get_inst_id()
                if obj_name is None:
                    obj_name = type(obj).__name__
            except AttributeError:
                obj_name = type(obj).__name__
            timer = FuncTimer(method_name, obj_name, self._call_stack)
            if isinstance(obj, System):
                if isinstance(obj, ParallelGroup):
                    self._par_groups.add(obj.pathname)
                parent = obj.pathname.rpartition('.')[0]
                is_par_child = parent in self._par_groups
            else:
                is_par_child = False
            if is_par_child or not self._par_only:
                if name not in self._timers:
                    self._timers[name] = []
                self._timers[name].append((timer, is_par_child, nprocs, type(obj).__name__))
                setattr(obj, method_name, _timer_wrap(method, timer))


@contextmanager
def timing_context(active=True):
    """
    Context manager to set whether timing is active or not.

    Note that this will only work if the --use_context arg is passed to the `openmdao timing`
    command line tool.  Otherwise it will be ignored and the entire python script will be
    timed.

    Parameters
    ----------
    active : bool
        Indicates if timing is active or inactive.

    Yields
    ------
    nothing
    """
    global _timing_active, _total_time

    active = bool(active)
    ignore = _timing_active and active
    if ignore:
        issue_warning("Timing is already active outside of this timing_context, so it will be "
                      "ignored.")

    start_time = perf_counter()

    save = _timing_active
    _timing_active = active
    try:
        yield
    finally:
        _timing_active = save
        if active and not ignore:
            _total_time += perf_counter() - start_time


def _obj_nprocs_iter(problem):
    yield (problem._name, problem, problem.comm.size)
    yield (problem._name + '.driver', problem.driver, problem.comm.size)
    for s in problem.model.system_iter(include_self=True, recurse=True):
        commsize = 1 if s.comm is None else s.comm.size
        yield (s.pathname, s, commsize)
        if s.linear_solver:
            yield (s.pathname + '.linear_solver', s.linear_solver, commsize)

        if s.nonlinear_solver:
            yield (s.pathname + '.nonlinear_solver', s.nonlinear_solver, commsize)


def _setup_sys_timers(options, problem, system, methods):
    # decorate all specified System methods
    global _timing_managers

    probname = system._problem_meta['name']
    # name_sys_procs = ((s.pathname, s, s.comm.size) for s in system.system_iter(include_self=True,
    #                                                                            recurse=True))
    if probname not in _timing_managers:
        _timing_managers[probname] = TimingManager(options)
    tmanager = _timing_managers[probname]
    tmanager.add_timings(_obj_nprocs_iter(problem), methods)


def _setup_timers(options, problem, system):
    global _global_timer_start

    # hook called after _setup_procs to decorate all specified System methods
    global _timing_managers

    timed_methods = _timer_methods['default']

    tmanager = _timing_managers.get(system._problem_meta['name'])
    if tmanager is not None and not tmanager._timers:
        _setup_sys_timers(options, problem, system, methods=timed_methods)


def _set_timer_setup_hook(options, problem):
    # This just sets a hook into the top level system of the model after we know it exists.
    # Note that this means that no timings can happen until AFTER _setup_procs is done.
    global _timing_managers

    probname = problem._name
    if probname not in _timing_managers:
        _timing_managers[probname] = TimingManager(options)
        hooks._register_hook('_setup_procs', 'System', inst_id='',
                             post=partial(_setup_timers, options, problem))
        hooks._setup_hooks(problem.model)


def _save_timing_data(options):
    # this is called by atexit after all timing data has been collected
    # Note that this will not be called if the program exits via sys.exit() with a nonzero
    # exit code.
    timing_file = options.outfile

    if timing_file is None:
        timing_file = 'timings.db'

    nprobs = len(_problem_names)

    timing_data = (_timing_managers, _total_time, nprobs)

    if MPI is not None:
        # need to consolidate the timing data from different procs
        all_managers = MPI.COMM_WORLD.gather(timing_data, root=0)
        if MPI.COMM_WORLD.rank != 0:
            return
    else:
        all_managers = [timing_data]

    try:
        os.remove(timing_file)
        issue_warning(f'The existing timing database, {timing_file},'
                      ' is being overwritten.', category=UserWarning)
    except OSError:
        pass

    tup2id = {}
    with sqlite3.connect(timing_file) as c:
        c.execute("CREATE TABLE func_index(id INTEGER PRIMARY KEY, rank INT, "
                  "prob_name TEXT, class_name TEXT, sys_name TEXT, method TEXT, "
                  "level INT, parallel INT, nprocs INT, ncalls INT, ftime REAL, "
                  "tmin REAL, tmax REAL)")

        c.execute("CREATE TABLE call_tree(id INTEGER PRIMARY KEY, parent_name TEXT, "
                  "parent_id INT, child_name TEXT, child_id INT, ncalls INT, ftime REAL, "
                  "tmin REAL, tmax REAL)")

        c.execute("CREATE TABLE global(id INTEGER PRIMARY KEY, total_time REAL, nprocs INT)")

        cur = c.cursor()
        childlist = []
        for fid, tup in enumerate(_timing_iter(all_managers)):
            rank, probname, classname, objname, level, parallel, nprocs, funcname, \
                ncalls, tmin, tmax, total, children = tup

            # uniquely identifies a function called from a specific parent
            idtup = (rank, probname, objname, funcname)
            childlist.append((children, idtup))
            assert(idtup not in tup2id)
            tup2id[idtup] = fid + 1  # add 1 to id because SQL ids start at 1

            cur.execute("INSERT INTO func_index(rank, prob_name, "
                        "class_name, sys_name, method, level, parallel, nprocs, ncalls,"
                        "ftime, tmin, tmax) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rank, probname, classname, objname, funcname, level, parallel, nprocs,
                         ncalls, total, tmin, tmax))

        for parent_id, (children, parent_tup) in enumerate(childlist):
            rank, probname, parent_name, funcname = parent_tup
            parentfunc = parent_name + '.' + funcname
            for chname, (ncalls, tmin, tmax, total) in children.items():
                childsys, _, childfunc = chname.rpartition('.')
                chid = tup2id[rank, probname, childsys, childfunc]
                cur.execute("INSERT INTO call_tree(parent_name, parent_id, child_name, child_id, "
                            "ncalls, ftime, tmin, tmax) VALUES(?,?,?,?,?,?,?,?)",
                            (parentfunc, parent_id + 1, chname, chid, ncalls, total, tmin, tmax))

        cur.execute("INSERT INTO global(total_time, nprocs) VALUES (?, ?)", (_total_time, nprocs))

    return timing_file


def wherestr(**kwargs):
    lst = []
    for name, val in kwargs.items():
        if val is None:
            continue
        if isinstance(val, str):
            lst.append(f"{name}='{val}'")
        else:
            lst.append(f"{name}={val}")
    if lst:
        return "WHERE " + " AND ".join(lst)

    return ''


def _global_info(db_fname):
    with sqlite3.connect(db_fname) as dbcon:
        cur = dbcon.cursor()
        try:
            for row in cur.execute(f"SELECT total_time, nprocs from global"):
                return row
        except sqlite3.OperationalError as err:
            print(err, file=sys.stderr)


def id2func_info(dbcon, func_id):
    cur = dbcon.cursor()
    try:
        for row in cur.execute(f"SELECT * from func_index WHERE id={func_id}"):
            return row
    except sqlite3.OperationalError as err:
        print(err, file=sys.stderr)


def func_info_iter(dbcon, sys_name, method=None, prob_name=None, rank=None):
    where = wherestr(sys_name=sys_name, method=method, prob_name=prob_name, rank=rank)
    cur = dbcon.cursor()
    try:
        for row in cur.execute(f"SELECT * from func_index {where}"):
            yield row
    except sqlite3.OperationalError as err:
        print(err, file=sys.stderr)
        yield None


def calls_iter(dbcon, func_id, child_id=None):
    where = wherestr(parent_id=func_id, child_id=child_id)
    cur = dbcon.cursor()
    for row in cur.execute(f"SELECT * from call_tree {where}"):
        yield row


def called_by_iter(dbcon, child_id):
    """
    Yield rows from db for all callers of the given function id.

    Parameters
    ----------
    dbcon : Database connection
        Connection to an open database.
    child_id : int
        Id into the top level function index table.

    Yields
    ------
    tuple
        Entries for a givel database row.
    """
    cur = dbcon.cursor()
    for row in cur.execute(f"SELECT * from call_tree WHERE child_id = {child_id}"):
        yield row


def _main_table_row_iter(db_fname):
    with sqlite3.connect(db_fname) as dbcon:
        cur = dbcon.cursor()
        for fid, rank, pname, cname, objname, fname, level, par, nprocs, calls, time, tmin, tmax \
                in cur.execute(f"SELECT * from func_index"):
            tavg = time / calls
            yield {
                'id': fid,
                'method': fname,
                'rank': rank,
                'nprocs': nprocs,
                'probname': pname,
                'class': cname,
                'sysname': objname,
                'level': level,
                'parallel': par,
                'ncalls': calls,
                'avg': tavg,
                'tmin': tmin,
                'tmax': tmax,
                'total': time,
            }


def db2table(rows, headers=None, outfile=None, format='simple_grid'):
    table = generate_table(sorted(rows, key=lambda x: x[3], reverse=True), tablefmt=format,
                           headers=headers)
    table.display(outfile=outfile)


def func_tree(dbcon, sys_name, method=None, prob_name=None, rank=None):
    for row in func_info_iter(dbcon, sys_name, method, prob_name=prob_name, rank=rank):
        fid, rank, pname, class_name, sname, fname, level, parallel, nprocs, ncalls, ftime, \
            tmin, tmax = row

        print(f"{sys_name}.{method} {ncalls}  {ftime}, {tmin}  {tmax}")

        stack = [calls_iter(dbcon, fid)]
        seen = set()
        while stack:
            indent = '  ' * len(stack)
            try:
                child_id, child_name, ncalls, ftime = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue

            print(f"{indent}{child_name}, {ncalls}, {ftime}")

            if child_id not in seen:
                stack.append(calls_iter(dbcon, child_id))


def obj_tree(dbcon):
    cur = dbcon.cursor()
    probdct = defaultdict(lambda: defaultdict(list))
    for row in cur.execute("SELECT rank, prob_name, sys_name, method, ncalls, ftime, tmin, "
                           "tmax from func_index ORDER BY ftime DESC"):
        rank, pname, sname, fname, ncalls, ftime, tmin, tmax = row

        if sname.endswith('.nonlinear_solver'):
            sname = sname.rpartition('.')[0]
            fname = 'nonlinear_solver.' + fname
        elif sname.endswith('.linear_solver'):
            sname = sname.rpartition('.')[0]
            fname = 'linear_solver.' + fname

        probdct[pname][sname].append((fname, ncalls, ftime, tmin, tmax))


    for probname, dct in probdct.items():
        sdict = {}
        for sname, rows in dct.items():
            sdict[sname] = np.sum(r[2] for r in rows)

        print("In problem", probname)
        for sname, ftime in sorted(sdict.items(), key=lambda x: x[1], reverse=True):
            rows = dct[sname]
            print(f"\nSystem: '{sname}', time: {ftime}")
            # print("type", type(rows), 'row0', type(rows[0]))
            generate_table(sorted(rows, key=lambda x: x[2], reverse=True), tablefmt='simple_grid',
                                  headers=['Function', 'Calls', 'Total Time', 'Min Time', 'Max Time']).display()
