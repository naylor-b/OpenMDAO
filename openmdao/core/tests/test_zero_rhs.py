import unittest
import itertools
from collections.abc import Iterable

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals
from openmdao.core.constants import _UNDEFINED

from openmdao.utils.mpi import MPI

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class Combiner(om.ExplicitComponent):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def setup(self):
        for i in range(self.size):
            self.add_input(f'x{i+1}', shape=1)

        self.declare_partials('*', '*')
        self.add_output('y', shape=self.size)

    def compute(self, inputs, outputs):
        for i in range(self.size):
            outputs['y'][i] = inputs[f'x{i+1}'] * (i+1)

    def compute_partials(self, inputs, partials):
        for i in range(self.size):
            v = np.zeros(self.size)
            v[i] = i + 1
            partials[f'y', f'x{i+1}'] = v


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        if isinstance(p, str):
            p = [p]
        elif not isinstance(p, Iterable):
            p = [p]
        for item in p:
            try:
                arg = item.__name__
            except:
                arg = str(item)
            args.append(arg)
    return func.__name__ + '_' + '_'.join(args)


class TestRHSZero(unittest.TestCase):

    def setup_model(self, size, mode, linsolver):
        p = om.Problem()
        model = p.model

        model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size)))
        G = model.add_subsystem('G', om.ParallelGroup())
        for i in range(size):
            sub = G.add_subsystem(f'sub{i+1}', om.Group())
            sub.linear_solver = linsolver()
            sub.add_subsystem(f'C{i+1}', om.ExecComp(f'y={i+1}.0*x'))
            model.connect('indep.x', f'G.sub{i+1}.C{i+1}.x', src_indices=[i])

        model.add_subsystem('final', Combiner(size))

        for i in range(size):
            model.connect(f'G.sub{i+1}.C{i+1}.y', f'final.x{i+1}')

        model.add_design_var('indep.x')
        model.add_constraint('final.y', lower=-10.0, upper=10.0)

        p.setup(mode=mode, force_alloc_complex=True)
        p.run_model()

        return p

    @parameterized.expand(itertools.product(['fwd', 'rev'],
                                            [om.DirectSolver, om.LinearBlockGS,
                                             om.ScipyKrylov, om.PETScKrylov]),
                          name_func=_test_func_name)
    def test_zero_rhs(self, mode, linsolver):
        p = self.setup_model(3, mode, linsolver)

        assert_check_totals(p.check_totals(method='cs', show_only_incorrect=True),
                            atol=1e-6, rtol=1e-6)


class TestRHSZeroMPI3(TestRHSZero):

    N_PROCS = 3


class TestRHSZeroMPI5(TestRHSZero):

    N_PROCS = 5


