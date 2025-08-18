import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.configuration import process_config, set_config
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2
from openmdao.utils.assert_utils import assert_near_equal


testdir = os.path.dirname(os.path.abspath(__file__))


class TestConfiguration(unittest.TestCase):
    def test_simple(self):
        cfg = """
        problem:
          type: openmdao.core.problem.Problem
          kwargs:
            name: simple
          model:
            type: openmdao.core.group.Group
            subsystems:
              - C1:
                  type: openmdao.components.exec_comp.ExecComp
                  kwargs:
                    exprs: y = 3.0 * x
        """
        prob = process_config(cfg)
        prob.setup()
        prob.run_model()

        prob2 = om.Problem()
        prob2.model.add_subsystem('C1', om.ExecComp('y=3.0*x'))
        prob2.setup()
        prob2.run_model()

        assert_near_equal(prob.get_val('C1.y'), prob2.get_val('C1.y'))

    def test_conn(self):
        cfg = """
        problem:
          type: openmdao.core.problem.Problem
          kwargs:
            name: conn
          model:
            type: openmdao.core.group.Group
            subsystems:
              - C1:
                  type: openmdao.components.exec_comp.ExecComp
                  kwargs:
                    exprs: y = 3.0 * x
              - C2:
                  type: openmdao.components.exec_comp.ExecComp
                  kwargs:
                    exprs: y = 4.0 * x
            connections:
              - src: C1.y
                tgt: C2.x
        """
        prob = process_config(cfg)
        prob.setup()
        prob.run_model()

        prob2 = om.Problem()
        prob2.model.add_subsystem('C1', om.ExecComp('y=3.0*x'))
        prob2.model.add_subsystem('C2', om.ExecComp('y=4.0*x'))
        prob2.model.connect('C1.y', 'C2.x')
        prob2.setup()
        prob2.run_model()

        assert_near_equal(prob.get_val('C2.y'), prob2.get_val('C2.y'))

    def test_nested(self):
        cfg = """
        problem:
          type: openmdao.core.problem.Problem
          kwargs:
            name: nested
          model:
            type: openmdao.core.group.Group
            subsystems:
              - G1:
                  type: openmdao.core.group.Group
                  subsystems:
                    - C1:
                        type: openmdao.components.exec_comp.ExecComp
                        kwargs:
                          exprs: y = 3.0 * x
                    - C2:
                        type: openmdao.components.exec_comp.ExecComp
                        kwargs:
                          exprs: y = 4.0 * x
                  connections:
                    - src: C1.y
                      tgt: C2.x
        """
        prob = process_config(cfg)
        prob.setup()
        prob.run_model()

        prob2 = om.Problem()
        G1 = prob2.model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y=3.0*x'))
        G1.add_subsystem('C2', om.ExecComp('y=4.0*x'))
        G1.connect('C1.y', 'C2.x')
        prob2.setup()
        prob2.run_model()

        assert_near_equal(prob.get_val('G1.C2.y'), prob2.get_val('G1.C2.y'))

    def test_set_config(self):
        prob = om.Problem()
        prob.model.add_subsystem('C1', om.ExecComp('y=3.0*x'))
        prob.model.add_subsystem('C2', om.ExecComp('y=4.0*x'))
        prob.model.connect('C1.y', 'C2.x')

        cfg = """
        nonlinear_solver:
          type: openmdao.solvers.nonlinear.newton.NewtonSolver
          kwargs:
            iprint: 0
            atol: 1.e-6
            solve_subsystems: True
          linesearch:
            type: openmdao.solvers.linesearch.backtracking.BoundsEnforceLS
        """
        set_config(prob.model, cfg)
        prob.setup()

        self.assertTrue(isinstance(prob.model.nonlinear_solver, om.NewtonSolver))
        self.assertTrue(isinstance(prob.model.nonlinear_solver.linesearch, om.BoundsEnforceLS))

        prob.run_model()

    def test__sellar(self):
        prob = process_config(os.path.join(testdir, 'simple_sellar_config.yml'))

        # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
        prob.model.approx_totals()

        prob.setup()
        prob.set_solver_print(level=0)

        prob.set_val('x', 2.0)
        prob.set_val('z', [-1., -1.])

        prob.run_model()

        prob2 = om.Problem()
        model = prob2.model

        cycle = model.add_subsystem('cycle', om.Group(), promotes_inputs=['x', 'z'])
        cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'],
                            promotes_outputs=['y1'])
        cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'],
                            promotes_outputs=['y2'])

        # cycle.set_input_defaults('x', 1.0)
        cycle.set_input_defaults('z', src_shape=(2, ))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.add_design_var('x', lower=-1.0, upper=10.0)
        model.add_design_var('z', lower=-1.0, upper=10.0)
        model.add_objective('obj_cmp.obj')
        model.add_constraint('con_cmp1.con1', upper=0.0)
        model.add_constraint('con_cmp2.con2', upper=0.0)

        prob2.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1.0e-8)

        prob2.setup()
        prob2.set_solver_print(level=0)

        prob2.set_val('x', 2.0)
        prob2.set_val('z', [-1., -1.])

        prob2.run_model()

        for name in ['x', 'z', 'y1', 'y2', 'obj', 'con1', 'con2']:
            assert_near_equal(prob.get_val(name), prob2.get_val(name), 1e-5)

        prob.run_driver()
        prob2.run_driver()

        print('minimum found at')
        assert_near_equal(prob.get_val('x'), prob2.get_val('x'), 1e-5)
        assert_near_equal(prob.get_val('z'), prob2.get_val('z'), 1e-5)

        print('minumum objective')
        assert_near_equal(prob.get_val('obj'), prob2.get_val('obj'), 1e-5)
