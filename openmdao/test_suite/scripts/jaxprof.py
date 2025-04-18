"""
This script is used to compare performance of JAX sparsity comp vs. plain sparsity comp.

Cmd line args control the following:

color: use coloring
prof: use profiler
rev: use reverse mode
check: check partials
jax: use JAX sparsity comp
group: use group of sparsity comps
sparse: use sparse partials
fd: use finite difference partials
show: show sparsity

"""

import time
import sys
from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np

import openmdao.api as om
from openmdao.devtools.debug import profiling
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.general_utils import do_nothing_context


class PerfTestCompFD(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        size = self.options['size']
        self.add_input('a', shape=(size,))
        self.add_input('b', shape=(size,))
        self.add_output('x', shape=(size,))
        self.add_output('y', shape=(size,))

    def setup_partials(self):
        self.declare_partials('x', 'a', rows=np.arange(size), cols=np.arange(size), method='fd')
        self.declare_partials('x', 'b', rows=np.arange(size), cols=np.arange(size), method='fd')
        self.declare_partials('y', 'b', rows=np.arange(size), cols=np.arange(size), method='fd')

    def compute_primal(self, a, b):
        x = a * b
        y = b * b
        return x, y


class PerfTestCompAnalytic(PerfTestCompFD):
    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        size = self.options['size']
        self.add_input('a', shape=(size,))
        self.add_input('b', shape=(size,))
        self.add_output('x', shape=(size,))
        self.add_output('y', shape=(size,))

    def setup_partials(self):
        self.declare_partials('x', 'a', rows=np.arange(size), cols=np.arange(size))
        self.declare_partials('x', 'b', rows=np.arange(size), cols=np.arange(size))
        self.declare_partials('y', 'b', rows=np.arange(size), cols=np.arange(size))

    def compute_primal(self, a, b):
        x = a * b
        y = b * b
        return x, y

    def compute_partials(self, inputs, partials):
        partials['x', 'a'] = inputs['b']
        partials['x', 'b'] = inputs['a']
        partials['y', 'b'] = 2 * inputs['b']


class JaxPerfTestComp(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        size = self.options['size']
        self.add_input('a', shape=(size,))
        self.add_input('b', shape=(size,))
        self.add_output('x', shape=(size,))
        self.add_output('y', shape=(size,))

    def compute_primal(self, a, b):
        x = a * b
        y = b * b
        return x, y



class SimpleJaxPerfTestComp(om.JaxExplicitComponent):
    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        size = self.options['size']
        self.add_input('a', shape=(size,))
        self.add_input('b', shape=(size,))
        self.add_output('x', shape=(size,))
        self.add_output('y', shape=(size,))

    @staticmethod
    def compute_primal(a, b):
        x = a * b
        y = b * b
        return x, y


def do_timing(meta):
    size = meta['size']
    reps = meta['reps']
    use_sparse = meta['sparse']
    use_group = meta['group']
    use_jax = meta['jax']
    use_jit = meta['jit']
    jax_comps = meta['jax_comps']
    use_coloring = meta['color']
    use_prof = meta['prof']
    use_jax_prof = meta['jax_prof']
    check = meta['check']
    simple = meta['simple']
    use_fd = meta['fd']
    ncomps = meta['ncomps']
    nsinks = meta['nsinks']
    show = meta['show']

    class MyJaxGroup(om.JaxExplicitGroup):
        def __init__(self, ncomps, klass, klass_kwargs, **kwargs):
            super().__init__(**kwargs)
            self.ncomps = ncomps
            self.klass = klass
            self.klass_kwargs = klass_kwargs

        def setup(self):
            for i in range(self.ncomps):
                self.add_subsystem('comp' + str(i), self.klass(**self.klass_kwargs))
                if i > 0:
                    self.connect('comp' + str(i - 1) + '.x', 'comp' + str(i) + '.a')
                    self.connect('comp' + str(i - 1) + '.y', 'comp' + str(i) + '.b')


    class MyGroup(om.Group):
        def __init__(self, ncomps, klass, klass_kwargs, **kwargs):
            super().__init__(**kwargs)
            self.ncomps = ncomps
            self.klass = klass
            self.klass_kwargs = klass_kwargs

        def setup(self):
            for i in range(self.ncomps):
                self.add_subsystem('comp' + str(i), self.klass(**self.klass_kwargs))
                if i > 0:
                    self.connect('comp' + str(i - 1) + '.x', 'comp' + str(i) + '.a')
                    self.connect('comp' + str(i - 1) + '.y', 'comp' + str(i) + '.b')


    p = om.Problem()
    model = p.model

    kwargs = {'use_jit': use_jit, 'size': size}

    if use_jax or jax_comps:
        if simple:
            klass = SimpleJaxPerfTestComp
        else:
            klass = JaxPerfTestComp
    else:
        if use_fd:
            klass = PerfTestCompFD
        else:
            klass = PerfTestCompAnalytic

    if use_group:
        G = model.add_subsystem('G', MyJaxGroup(ncomps, klass, kwargs) if use_jax
                                else MyGroup(ncomps, klass, kwargs))  # currently top group can't be a jax group
        model.add_subsystem('sink', om.ExecComp([f'y{i} = x{i}' for i in range(nsinks)]))
        system = G
    else:
        comp = p.model.add_subsystem('comp', klass(**kwargs))
        if use_fd:
            comp.options['derivs_method'] = 'fd'
            method = 'fd'
        else:
            method = 'jax' if use_jax else 'exact'
        if use_coloring:
            comp.declare_coloring(show_summary=True, show_sparsity=show, method=method)
        system = comp

    results = {
        'class': system.__class__.__name__,
    }

    if use_group:
        results['comp type'] = klass.__name__
        results['ncomps'] = ncomps
        results['nsinks'] = nsinks

    results.update({
        'size': size,
        'jit': use_jit,
        'sparse': use_sparse,
        'color': use_coloring,
        'fd': use_fd,
    })

    print("Performance for:")
    pprint(meta)

    t0 = time.perf_counter()
    p.setup()
    setup_time = time.perf_counter() - t0
    print(f'setup time: {setup_time}')
    results['setup time'] = setup_time

    if use_group:
        for i in range(nsinks):
            model.connect(f'G.comp{ncomps - 1}.y{i}', f'sink.x{i}')

    t0 = time.perf_counter()
    for i in range(reps):
        p.run_model()
    run_time = time.perf_counter() - t0
    print(f'run_model time: {run_time}')
    results['run time'] = run_time

    if check:
        t0 = time.perf_counter()
        if use_group:
            assert_check_partials(p.check_partials(method='fd', compact_print=True,
                                                   show_only_incorrect=True))
        else:
            assert_check_partials(system.check_partials(method='fd', compact_print=True,
                                                        show_only_incorrect=True))
        check_time = time.perf_counter() - t0
        print(f'check_partials time: {check_time}')
        results['check time'] = check_time

    if use_prof or use_jax_prof:
        profname = 'color' if use_coloring else 'nocolor'
        if use_jax:
            profname = 'jax_' + profname
        if use_group:
            profname = 'group_' + profname
        if use_fd:
            profname = profname + '_fd'
        if not use_sparse:
            profname = profname + '_dense'
        if simple:
            profname = profname + '_simple'
        profname = profname + f'_{size}'

        if use_jax_prof:
            jax_profile_dir = profname + '.jaxprof'
            ctx = jax.profiler.trace(jax_profile_dir, create_perfetto_link=True)
            jax.profiler.save_device_memory_profile(f"{jax_profile_dir}/memory0.prof")
        else:
            ctx = profiling(profname + '.prof')
    else:
        ctx = do_nothing_context()

    t0 = time.perf_counter()
    with ctx:
        for i in range(reps):
            system._linearize()
            if use_jax_prof:
                jax.profiler.save_device_memory_profile(f"{jax_profile_dir}/memory{i}.prof")

    t1 = time.perf_counter()
    print(f'linearize time: {t1 - t0} for {reps} reps')
    results['linearize time'] = t1 - t0

    return results


def read_args(args=None):
    if args is None:
        args = sys.argv[1:]

    meta = {
        'color': 'color' in args,
        'prof': 'prof' in args,
        'jax_prof': 'jaxprof' in args,
        'check': 'check' in args,
        'jax': 'jax' in args,
        'jit': 'jit' in args,
        'jax_comps': 'jaxcomps' in args,
        'show': 'show' in args,
        'fd': 'fd' in args,
        'sparse': 'sparse' in args,
        'group': 'group' in args,
        'simple': 'simple' in args,
        'nsinks': int('nsinks' in args)
    }
    ncomps = 0
    if not meta['group']:
        for arg in args:
            if arg.startswith('group='):
                meta['group'] = True
                ncomps = int(arg.rpartition('=')[-1])
                break
    meta['ncomps'] = ncomps

    nsinks = 0
    if not meta['nsinks']:
        for arg in args:
            if arg.startswith('nsinks='):
                meta['nsinks'] = int(arg.rpartition('=')[-1])
                break
    meta['nsinks'] = nsinks

    return meta


if __name__ == '__main__':
    reps = 100
    size = 500

    meta = read_args()
    for name, val in meta.items():
        if val and name != 'group':
            # don't do loop tests because specific args were passed in
            meta['reps'] = reps
            meta['size'] = size
            results = do_timing(meta)
            print(f"\nResults:\n")
            pprint(results)
            sys.exit()

    meta['reps'] = reps
    meta['size'] = size

    reslist = []
    metalist = []

    if meta['group']:
        meta['ncomps'] = 5
        for nsinks in [1, 5]:
            meta['nsinks'] = nsinks
            for with_jax in [True, False]:
                meta['jax'] = with_jax
                meta['group'] = True
                if with_jax:
                    meta['jax_comps'] = False
                    for with_jit in [True, False]:
                        meta['jit'] = with_jit
                        if with_jit:
                            meta['color'] = True
                        results = do_timing(meta)
                        meta['color'] = False
                        reslist.append(results)
                else:
                    for jaxcomps in [True, False]:
                        meta['jax_comps'] = jaxcomps
                        meta['jit'] = jaxcomps
                        for simple in [True, False]:
                            meta['simple'] = simple
                            results = do_timing(meta)
                            reslist.append(results)
    else:  # single component tests
        for with_jax, with_jit in [(True, True), (True, False), (False, False)]:
            meta['jax'] = with_jax
            meta['jit'] = with_jit
            if with_jax:
                for simple in [True, False]:
                    meta['simple'] = simple
                    if not simple:
                        for coloring in [True, False]:
                            meta['color'] = coloring
                            results = do_timing(meta)
                            reslist.append(results)
                    else:
                        results = do_timing(meta)
                        reslist.append(results)
            else:
                for do_fd in [True, False]:
                    meta['fd'] = do_fd
                    results = do_timing(meta)
                    reslist.append(results)

    om.generate_table(reslist, headers='keys', tablefmt='tabulator').display()

