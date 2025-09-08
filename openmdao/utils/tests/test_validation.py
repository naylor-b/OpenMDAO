import unittest

import os

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.validation import DataModelManager as dmm
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

testdir = os.path.dirname(os.path.abspath(__file__))


class TestValidation(unittest.TestCase):
    def test_typestr_to_class(self):
        from openmdao.solvers.solver import Solver
        self.assertEqual(dmm.type_to_class('openmdao.solvers.solver.Solver'), Solver)

    def test_dict_to_instance_simple(self):
        cfg = {
            'type': 'openmdao.core.problem.Problem',
            'name': 'simple',
            'model': {
                'type': 'openmdao.core.group.Group',
                'subsystems': [
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'C1',
                        'exprs': 'y = 3.0 * x'
                    }
                ]
            }
        }
        randval = np.random.random(1)[0]

        prob = dmm.from_dict(cfg)
        prob.setup()
        prob.set_val('C1.x', randval)
        prob.setup()
        prob.run_model()

        prob2 = om.Problem()
        prob2.model.add_subsystem('C1', om.ExecComp('y=3.0*x'))
        prob2.setup()
        prob2.set_val('C1.x', randval)
        prob2.setup()
        prob2.run_model()

        assert_near_equal(prob.get_val('C1.y'), prob2.get_val('C1.y'))

    def test_dict_to_instance_with_conn(self):
        cfg = {
            'type': 'openmdao.core.problem.Problem',
            'name': 'simple',
            'model': {
                'type': 'openmdao.core.group.Group',
                'subsystems': [
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'C1',
                        'exprs': 'y = 3.0 * x'
                    },
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'C2',
                        'exprs': 'y = 2.0 * x'
                    }
                ],
                'connections': [
                    {
                        'src': 'C1.y',
                        'tgt': 'C2.x'
                    }
                ]
            },
        }
        randval = np.random.random(1)[0]

        prob = dmm.from_dict(cfg)
        prob.setup()
        prob.set_val('C1.x', randval)
        prob.run_model()

        prob2 = om.Problem()
        prob2.model.add_subsystem('C1', om.ExecComp('y=3.0*x'))
        prob2.model.add_subsystem('C2', om.ExecComp('y=2.0*x'))
        prob2.model.connect('C1.y', 'C2.x')
        prob2.setup()
        prob2.set_val('C1.x', randval)
        prob2.run_model()

        assert_near_equal(prob.get_val('C1.y'), prob2.get_val('C1.y'))
        assert_near_equal(prob.get_val('C2.y'), prob2.get_val('C2.y'))

    def test_dict_to_instance_with_promotion(self):
        cfg = {
            'type': 'openmdao.core.problem.Problem',
            'name': 'simple',
            'model': {
                'type': 'openmdao.core.group.Group',
                'subsystems': [
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'C1',
                        'exprs': 'y = 3.0 * x',
                        'promotes': ['y']
                    },
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'C2',
                        'exprs': 'y = 2.0 * x',
                        'promotes': [('x','y')]
                    }
                ],
            },
        }
        randval = np.random.random(1)[0]

        prob = dmm.from_dict(cfg)
        prob.setup()
        prob.set_val('C1.x', randval)
        prob.run_model()

        prob2 = om.Problem()
        prob2.model.add_subsystem('C1', om.ExecComp('y=3.0*x'), promotes=['y'])
        prob2.model.add_subsystem('C2', om.ExecComp('y=2.0*x'), promotes=[('x', 'y')])
        prob2.setup()
        prob2.set_val('C1.x', randval)
        prob2.run_model()

        assert_near_equal(prob.get_val('C1.y'), prob2.get_val('C1.y'))
        assert_near_equal(prob.get_val('C2.y'), prob2.get_val('C2.y'))

    def test_sellar(self):
        cfg = {
            'type': 'openmdao.core.problem.Problem',
            'name': 'Sellar_MDA',

            'driver': {
                'type': 'openmdao.drivers.scipy_optimizer.ScipyOptimizeDriver',
                'options': {
                    'invalid_desvar_behavior': 'warn',
                    'optimizer': 'SLSQP',
                    'tol': 1.0e-8
                }
            },
            'model': {
                'type': 'openmdao.core.group.Group',
                'subsystems': [
                    {
                        'type': 'openmdao.core.group.Group',
                        'name': 'cycle',
                        'subsystems': [
                            {
                                'type': 'openmdao.test_suite.components.sellar.SellarDis1',
                                'name': 'd1',
                                'promotes_inputs': ['x', 'z', 'y2'],
                                'promotes_outputs': ['y1']
                            },
                            {
                                'type': 'openmdao.test_suite.components.sellar.SellarDis2',
                                'name': 'd2',
                                'promotes_inputs': ['z', 'y1'],
                                'promotes_outputs': ['y2']
                            }
                        ],
                        'input_defaults': [
                            {
                                'name': 'z',
                                'src_shape': (2,)
                            }
                        ],
                        'nonlinear_solver': {
                            'type': 'openmdao.solvers.nonlinear.nonlinear_block_gs.NonlinearBlockGS'
                        },
                        'promotes_inputs': ['x', 'z']
                    },
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'obj_cmp',
                        'exprs': 'obj = x**2 + z[1] + y1 + exp(-y2)',
                        'kwargs': {
                            'z': [0.0, 0.0],
                            'x': 0.0
                        },
                        'promotes': ['x', 'z', 'y1', 'y2', 'obj']
                    },
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'con_cmp1',
                        'exprs': 'con1 = 3.16 - y1',
                        'promotes': ['con1', 'y1']
                    },
                    {
                        'type': 'openmdao.components.exec_comp.ExecComp',
                        'name': 'con_cmp2',
                        'exprs': 'con2 = y2 - 24.0',
                        'promotes': ['con2', 'y2']
                    }
                ],
                # 'connections': [
                #     {
                #         'src': 'cycle.d1.y1',
                #         'tgt': ['obj_cmp.y1', 'con_cmp1.y1']
                #     },
                #     {
                #         'src': 'cycle.d2.y2',
                #         'tgt': ['obj_cmp.y2', 'con_cmp2.y2']
                #     }
                # ],
                'design_variables': [
                    {
                        'name': 'x',
                        'lower': -1.0,
                        'upper': 10.0
                    },
                    {
                        'name': 'z',
                        'lower': -1.0,
                        'upper': 10.0
                    }
                ],
                'objectives': [
                    {
                        'name': 'obj_cmp.obj',
                    }
                ],
                'constraints': [
                    {
                        'name': 'con_cmp1.con1',
                        'upper': 0.0
                    },
                    {
                        'name': 'con_cmp2.con2',
                        'upper': 0.0
                    }
                ],
            },
        }

        prob = dmm.from_dict(cfg)
        prob.model.approx_totals()
        prob.setup()
        prob.set_val('x', 2.0)
        prob.set_val('z', [0.0, 0.0])
        prob.set_solver_print(level=0)
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
                                                  z=np.array([0.0, 0.0])),#, x=0.0),
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
        prob2.set_val('z', [0.0, 0.0])

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


if __name__ == '__main__':
    unittest.main()