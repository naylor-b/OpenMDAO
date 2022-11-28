"""
Classes and functions for timing method calls.
"""
import pickle
from collections import defaultdict
from time import perf_counter
from contextlib import contextmanager
from functools import wraps, partial

import openmdao.utils.hooks as hooks
from openmdao.utils.om_warnings import issue_warning
from openmdao.core.parallel_group import ParallelGroup

# can use this to globally turn timing on/off so we can time specific sections of code
_timing_active = False
_total_time = 0.
_timing_managers = {}


class _RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        # Only allow classes from this module.
        if module == 'openmdao.visualization.timing_viewer.timer' or name == 'defaultdict':
            return globals().get(name)
        # Forbid everything else.
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden in timing file.")


def _restricted_load(f):
    # Like pickle.load() but restricted to a specific set of classes.
    return _RestrictedUnpickler(f).load()


def _timing_iter(all_timing_managers):
    tot_nprocs = len(all_timing_managers)

    # iterates over all of the timing managers and yields all timing info
    for rank, (timing_managers, tot_time, nprobs) in enumerate(all_timing_managers):
        if tot_nprocs == 1:
            rank = None
        for probname, tmanager in timing_managers.items():
            if nprobs == 1:
                probname = None
            for sysname, timers in tmanager._timers.items():
                level = len(sysname.split('.')) if sysname else 0
                for t, parallel, nprocs, classname in timers:
                    if t.info.ncalls > 0:
                        yield rank, probname, classname, sysname, level, parallel, nprocs, t.name,\
                            t.info.ncalls, t.avg(), t.info.min, t.info.max, t.info.total, tot_time,\
                            t.children


def _timer_obj_iter(all_timing_managers):
    tot_nprocs = len(all_timing_managers)

    # iterates over all of the timing managers and yields all timing info
    for rank, (timing_managers, tot_time, nprobs) in enumerate(all_timing_managers):
        if tot_nprocs == 1:
            rank = None
        for probname, tmanager in timing_managers.items():
            if nprobs == 1:
                probname = None
            for sysname, timers in tmanager._timers.items():
                level = len(sysname.split('.')) if sysname else 0
                for t, parallel, nprocs, classname in timers:
                    if t.info.ncalls > 0:
                        yield rank, probname, classname, sysname, level, parallel, nprocs, t.name,\
                            t.info.ncalls, t.avg(), t.info.min, t.info.max, t.info.total, tot_time,\
                            t.children


def _timing_file_iter(timing_file):
    # iterates over the given timing file
    with open(timing_file, 'rb') as f:
        yield from _timing_iter(_restricted_load(f))


def _get_par_child_info(timing_iter, method_info):
    # puts timing info for direct children of parallel groups into a dict.
    parents = {}
    klass, method = method_info

    for rank, probname, classname, sysname, level, parallel, nprocs, func, ncalls, avg, tmin, tmax,\
            ttot, tot_time, children in timing_iter:
        if not parallel or method != func:
            continue

        parent = sysname.rpartition('.')[0]
        key = (probname, parent)
        if key not in parents:
            parents[key] = {}

        if sysname not in parents[key]:
            parents[key][sysname] = []

        parents[key][sysname].append((rank, ncalls, avg, tmin, tmax, ttot))

    return parents


class FuncTimerInfo(object):

    __slots__ = ['ncalls', 'min', 'max', 'total']

    def __init__(self):
        self.ncalls = 0
        self.min = 1e99
        self.max = 0
        self.total = 0

    def called(self, dt):
        if dt < self.min:
            self.min = dt
        if dt > self.max:
            self.max = dt
        self.total += dt
        self.ncalls += 1

    def avg(self):
        """
        Return the average elapsed time for a method call.

        Returns
        -------
        float
            The average elapsed time for a method call.
        """
        if self.ncalls > 0:
            return self.total / self.ncalls
        return 0.


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
        self.name = objname + '.' + name
        self.stack = stack
        self.children = defaultdict(FuncTimerInfo)
        self.info = FuncTimerInfo()

    def pre(self):
        """
        Record the method start time.
        """
        global _timing_active
        if _timing_active:
            self.stack.append(self)
            self.start = perf_counter()

    def post(self):
        """
        Update ncalls, tot, min, and max.
        """
        global _timing_active
        if _timing_active:
            dt = perf_counter() - self.start
            self.info.called(dt)
            self.stack.pop()
            if self.stack:
                self.stack[-1].children[self.name].called(dt)

    def avg(self):
        """
        Return the average elapsed time for a method call.

        Returns
        -------
        float
            The average elapsed time for a method call.
        """
        return self.info.avg()


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
        ret = f(*args, **kwargs)
        timer.post()
        return ret

    return wraps(f)(do_timing)


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

    def add_timings(self, name_obj_proc_iter, method_names):
        """
        Add FuncTimers for all instances in name_obj_iter and all methods in method_names.

        Parameters
        ----------
        name_obj_proc_iter : iterator
            Yields (name, object, nprocs) tuples.
        method_names : list of str
            List of names of methods to wrap.
        """
        for name, obj, nprocs in name_obj_proc_iter:
            for _type, method_name in method_names:
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
            except AttributeError:
                obj_name = type(obj).__name__
            timer = FuncTimer(method_name, obj_name, self._call_stack)
            if isinstance(obj, ParallelGroup):
                self._par_groups.add(obj.pathname)
            parent = obj.pathname.rpartition('.')[0]
            is_par_child = parent in self._par_groups
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


def _setup_sys_timers(options, system, method_names):
    # decorate all specified System methods
    global _timing_managers

    probname = system._problem_meta['name']
    name_sys_procs = ((s.pathname, s, s.comm.size) for s in system.system_iter(include_self=True,
                                                                               recurse=True))
    if probname not in _timing_managers:
        _timing_managers[probname] = TimingManager(options)
    tmanager = _timing_managers[probname]
    tmanager.add_timings(name_sys_procs, method_names)


def _setup_timers(options, system):
    # hook called after _setup_procs to decorate all specified System methods
    global _timing_managers

    timed_methods = options.funcs

    tmanager = _timing_managers.get(system._problem_meta['name'])
    if tmanager is not None and not tmanager._timers:
        _setup_sys_timers(options, system, method_names=timed_methods)


def _set_timer_setup_hook(options, problem):
    # This just sets a hook into the top level system of the model after we know it exists.
    # Note that this means that no timings can happen until AFTER _setup_procs is done.
    global _timing_managers

    probname = problem._name
    if probname not in _timing_managers:
        _timing_managers[probname] = TimingManager(options)
        hooks._register_hook('_setup_procs', 'System', inst_id='',
                             post=partial(_setup_timers, options))
        hooks._setup_hooks(problem.model)
