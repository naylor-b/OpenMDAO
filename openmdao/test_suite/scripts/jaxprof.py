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

import jax
import jax.numpy as jnp
import numpy as np

import openmdao.api as om
from openmdao.devtools.debug import profiling
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.test_suite.components.sparsity_comp import SparsityComp, JaxSparsityComp
from openmdao.utils.array_utils import rand_sparsity
from openmdao.utils.general_utils import do_nothing_context

from jax.profiler import start_trace, stop_trace


class JaxMultiSparsityComp(om.JaxExplicitComponent):
    def __init__(self, sparsities, **kwargs):
        super().__init__(**kwargs)
        self.sparsities = [jnp.array(sparsity) for sparsity in sparsities]

    def setup(self):
        self.add_input('x', shape=self.sparsities[0].shape[1])
        self.add_output('y', shape=self.sparsities[0].shape[0])

    def compute_primal(self, x):
        print("computing primal", self.pathname, type(x))
        y = None
        for sparsity in self.sparsities:
            if y is None:
                y = sparsity @ x
            else:
                y = sparsity @ y
        return y


args = sys.argv[1:]

use_coloring = 'color' in args
use_prof = 'prof' in args
use_jax_prof = 'jaxprof' in args
rev = 'rev' in args
check = 'check' in args
use_jax = 'jax' in args
use_jit = 'jit' in args
show = 'show' in args
use_fd = 'fd' in args
use_sparse = 'sparse' in args
use_group = 'group' in args
ncomps = 1
if not use_group:
    for arg in args:
        if arg.startswith('group='):
            use_group = True
            ncomps = int(arg.rpartition('=')[-1])
            break

if use_group:
    nrows = ncols = 500
else:
    if rev:
        nrows = 100
        ncols = 1000
    else:
        nrows = 1000
        ncols = 100


def main():
    rng = np.random.default_rng(66)
    p = om.Problem()

    klass = JaxSparsityComp if use_jax else SparsityComp
    sparsity = rand_sparsity((nrows, ncols), 0.01, rng=rng)
    if not use_sparse:
        sparsity = sparsity.toarray()

    if use_group:
        # comp = p.model.add_subsystem('comp', JaxMultiSparsityComp(sparsities=[sparsity]*ncomps, use_jit=use_jit))
        # system = comp
        model = p.model
        G = model.add_subsystem('G', om.JaxExplicitGroup() if use_jax else om.Group())  # currently top group can't be a jax group
        for i in range(ncomps):
            G.add_subsystem('comp' + str(i), klass(sparsity=sparsity, use_jit=use_jit))
            if i > 0:
                G.connect('comp' + str(i - 1) + '.y', 'comp' + str(i) + '.x')
        system = G
    else:
        comp = p.model.add_subsystem('comp', klass(sparsity=sparsity, use_jit=use_jit))
        if use_coloring:
            comp.declare_coloring(show_summary=True, show_sparsity=show)
        if use_fd:
            comp.options['derivs_method'] = 'fd'
        system = comp

    print("Performance for args: ", args)

    t0 = time.perf_counter()
    p.setup()
    setup_time = time.perf_counter() - t0
    print(f'setup time: {setup_time}')

    t0 = time.perf_counter()
    p.run_model()
    run_time = time.perf_counter() - t0
    print(f'run_model time: {run_time}')

    if check:
        t0 = time.perf_counter()
        if use_group:
            assert_check_partials(p.check_partials(method='fd', show_only_incorrect=True))
        else:
            assert_check_partials(system.check_partials(method='fd', show_only_incorrect=True))
        check_time = time.perf_counter() - t0
        print(f'check_partials time: {check_time}')

    if use_prof or use_jax_prof:
        profname = 'color' if use_coloring else 'nocolor'
        if use_jax:
            profname = 'jax_' + profname
        if use_group:
            profname = 'group_' + profname
        if rev:
            profname = profname + '_rev'
        if use_fd:
            profname = profname + '_fd'
        if not use_sparse:
            profname = profname + '_dense'

        if use_jax_prof:
            start_trace(profname + '.jaxprof')
            ctx = do_nothing_context()
        else:
            ctx = profiling(profname + '.prof')
    else:
        ctx = do_nothing_context()

    reps = 1000
    t0 = time.perf_counter()
    with ctx:
        for i in range(reps):
            system._linearize()  # force coloring to be computed

    if use_jax_prof:
        stop_trace()

    t1 = time.perf_counter()
    print(f'linearize time: {t1 - t0} for {reps} reps')


if __name__ == '__main__':
    main()

