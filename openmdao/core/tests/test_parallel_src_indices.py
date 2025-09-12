import unittest
import numpy as np
from openmdao.utils.mpi import MPI
from pydantic import Field

from openmdao.api import Problem
from openmdao.api import ExplicitComponent
from openmdao.api import NonlinearRunOnce, LinearRunOnce
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class CompOptions(ExplicitComponentOptions):
    node_size: int = Field(default=0, desc='Node size')


class Comp(ExplicitComponent):

    def setup(self):
        node_size = self.options['node_size']

        self.add_input( 'x', shape=node_size, distributed=True)
        self.add_output('y', shape=node_size, distributed=True)

    def compute(self,inputs,outputs):
        outputs['y'] = inputs['x'] + 1.0


@dmm.register(Comp)
class CompModel(ExplicitComponentModel):
    options: CompOptions = Field(default_factory=CompOptions)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestSrcIndices(unittest.TestCase):
    N_PROCS = 4

    def test_zero_src_indices(self):
        prob = Problem()
        model = prob.model
        model.nonlinear_solver = NonlinearRunOnce()
        model.linear_solver = LinearRunOnce()

        irank = prob.comm.Get_rank()
        if irank == 1:
            node_size = 0
        else:
            node_size = 3

        model.add_subsystem('comp1', Comp(node_size=node_size))
        model.add_subsystem('comp2', Comp(node_size=node_size))

        n_list = prob.comm.allgather(node_size)
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        model.connect('comp1.y','comp2.x', src_indices=np.arange(n1, n2, dtype=int))
        model.connect('comp2.y','comp1.x')

        prob.setup()
        prob.run_model()

        if prob.comm.rank == 1:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.array([]))
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.array([]))
        else:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.ones(3) * 2.)
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.ones(3) * 3.)

    def test_zero_src_indices_flat(self):
        prob = Problem()
        model = prob.model
        model.nonlinear_solver = NonlinearRunOnce()
        model.linear_solver = LinearRunOnce()

        irank = prob.comm.Get_rank()
        if irank == 1:
            node_size = 0
        else:
            node_size = 3

        model.add_subsystem('comp1', Comp(node_size=node_size))
        model.add_subsystem('comp2', Comp(node_size=node_size))

        n_list = prob.comm.allgather(node_size)
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        model.connect('comp1.y','comp2.x', src_indices=np.arange(n1, n2, dtype=int), flat_src_indices=True)
        model.connect('comp2.y','comp1.x')

        prob.setup()
        prob.run_model()

        if prob.comm.rank == 1:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.array([]))
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.array([]))
        else:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.ones(3) * 2.)
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.ones(3) * 3.)
