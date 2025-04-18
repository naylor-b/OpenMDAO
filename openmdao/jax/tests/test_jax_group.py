import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.api as om

from openmdao.jax.jax_utils import jnp
from openmdao.jax.tests.test_jax_implicit import JaxQuadraticCompPrimal
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.testing_utils import parameterized_name

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


class JaxExplicitComp1(om.JaxExplicitComponent):
    def setup(self):
        self.options['default_to_dyn_shapes'] = True

        self.add_input('x')
        self.add_input('y')
        self.add_output('z')

    def compute_primal(self, x, y):
        return jnp.dot(x, y)


class JaxExplicitComp1Shaped(om.JaxExplicitComponent):
    def __init__(self, xshape, yshape, **kwargs):
        super().__init__(**kwargs)
        self.xshape = xshape
        self.yshape = yshape

    def setup(self):
        self.add_input('x', shape=self.xshape)
        self.add_input('y', shape=self.yshape)
        self.add_output('z', shape=(self.xshape[0], self.yshape[1]))

    def compute_primal(self, x, y):
        return jnp.dot(x, y)


class JaxExplicitComp2(om.JaxExplicitComponent):
    def setup(self):
        self.options['default_to_dyn_shapes'] = True

        self.add_input('x')
        self.add_input('y')
        self.add_output('z')
        self.add_output('zz')

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * 2.5
        return z, zz


class JaxExplicitComp2Shaped(om.JaxExplicitComponent):
    def __init__(self, xshape, yshape, **kwargs):
        super().__init__(**kwargs)
        self.xshape = xshape
        self.yshape = yshape

    def setup(self):
        self.add_input('x', shape=self.xshape)
        self.add_input('y', shape=self.yshape)
        self.add_output('z', shape=(self.xshape[0], self.yshape[1]))
        self.add_output('zz', shape=self.yshape)

    def compute_primal(self, x, y):
        z = jnp.dot(x, y)
        zz = y * 2.5
        return z, zz


class SellarDis1Primal(om.JaxExplicitComponent):
    def __init__(self, units=None, ref=1.):
        super().__init__()
        self._units = units
        self._ref = ref

    def setup(self):
        self.add_input('z', val=np.zeros(2), units=self._units)
        self.add_input('x', val=0., units=self._units)
        self.add_input('y2', val=1.0, units=self._units)
        self.add_output('y1', val=1.0, lower=0.1, upper=1000., units=self._units, ref=self._ref)

    def compute_primal(self, z, x, y2):
        y1 = z[0]**2 + z[1] + x - 0.2*y2
        return y1


class SellarDis2Primal(om.JaxExplicitComponent):
    def __init__(self, units=None, ref=1.):
        super().__init__()
        self._units = units
        self._ref = ref

    def setup(self):
        self.add_input('z', val=np.zeros(2), units=self._units)
        self.add_input('y1', val=1.0, units=self._units)
        self.add_output('y2', val=1.0, lower=0.1, upper=1000., units=self._units, ref=self._ref)

    def compute_primal(self, z, y1):
        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        newy1 = jnp.select(condlist=[y1.real > 0.0], choicelist=[y1], default=-y1)
        y2 = newy1**.5 + z[0] + z[1]
        return y2


class SellarPrimalGrouped(om.Group):
    def setup(self):
        self.mda = mda = self.add_subsystem('mda', om.Group(derivs_method='jax'), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d1', SellarDis1Primal(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2Primal(), promotes=['z', 'y1', 'y2'])
        self.mda.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.mda.linear_solver = om.ScipyKrylov()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))


class TestJaxGroup(unittest.TestCase):
    @parameterized.expand(['fwd', 'rev'], name_func=parameterized_name)
    def test_jax_group_outer_ivc(self, mode):
        x_shape = (2, 3)
        y_shape = (3, 4)

        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        ivc.add_output('y', val=np.ones(y_shape))
        G = p.model.add_subsystem('G', om.JaxExplicitGroup())
        G.add_subsystem('comp2', JaxExplicitComp2())
        G.add_subsystem('comp', JaxExplicitComp1())

        p.model.connect('ivc.x', ['G.comp.x', 'G.comp2.x'])
        p.model.connect('ivc.y', 'G.comp2.y')
        G.connect('comp2.zz', 'comp.y')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('ivc.x', x)
        p.set_val('ivc.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)
        assert_check_partials(p.check_partials(show_only_incorrect=True))
        assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             wrt=['G.comp2.x', 'G.comp2.y'], method='fd', show_only_incorrect=True))

    @parameterized.expand(['fwd', 'rev'], name_func=parameterized_name)
    def test_jax_group_auto_ivc(self, mode):
        x_shape = (2, 3)
        y_shape = (3, 4)

        p = om.Problem()
        G = p.model.add_subsystem('G', om.JaxExplicitGroup())

        G.add_subsystem('comp2', JaxExplicitComp2Shaped(x_shape, y_shape))
        G.add_subsystem('comp', JaxExplicitComp1Shaped(x_shape, y_shape))

        G.connect('comp2.zz', 'comp.y')

        p.setup(mode=mode)

        x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        p.set_val('G.comp.x', x)
        p.set_val('G.comp2.x', x)
        p.set_val('G.comp2.y', y)
        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)
        assert_near_equal(p.get_val('G.comp.z'), np.dot(x, y * 2.5))
        assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                           wrt=['G.comp2.x', 'G.comp2.y'], method='fd', show_only_incorrect=True))
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    @parameterized.expand(['fwd', 'rev'], name_func=parameterized_name)
    def test_cycle_error(self, mode):
        x_shape = (2, 3)
        y_shape = (3, 4)

        p = om.Problem()
        G = p.model.add_subsystem('G', om.JaxExplicitGroup())

        G.add_subsystem('comp2', JaxExplicitComp2Shaped(x_shape, y_shape))
        G.add_subsystem('comp', JaxExplicitComp1Shaped(x_shape, y_shape))

        G.connect('comp2.zz', 'comp.y')
        G.connect('comp.z', 'comp2.x')

        with self.assertRaises(Exception) as ctx:
            p.setup(mode=mode)

        self.assertEqual(ctx.exception.args[0],
                         "'G' <class JaxExplicitGroup>: JAX mode is currently supported for explicit groups only. They must contain no implicit components or cycles."
)

    def test_jax_implicit_comp_group(self):
        raise unittest.SkipTest("Skipping this test until implicit AD support works.")
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('a'))
        ivc.add_output('b')
        ivc.add_output('c')
        G = p.model.add_subsystem('G', om.Group(derivs_method='jax'))
        #G = p.model.add_subsystem('G', om.Group())
        G.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        G.linear_solver = om.ScipyKrylov()
        G.add_subsystem('comp', JaxQuadraticCompPrimal())
        p.model.connect('ivc.a', 'G.comp.a')
        p.model.connect('ivc.b', 'G.comp.b')
        p.model.connect('ivc.c', 'G.comp.c')
        p.setup()

        p.set_val('ivc.a', 1.0)
        p.set_val('ivc.b', -4.0)
        p.set_val('ivc.c', 3.0)

        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('G.comp.x'), 3.0)
        assert_check_partials(p.check_partials(show_only_incorrect=True), atol=2e-6)

        assert_check_totals(p.check_totals(of=['G.comp.x'],
                                           wrt=['G.comp.a', 'G.comp.b', 'G.comp.c'],
                                           method='fd',
                                           show_only_incorrect=True,
                                           abs_err_tol=3e-5,
                                           rel_err_tol=5e-6),
                            atol=3e-5, rtol=5e-6)

    def test_jax_sellar_primal_grouped(self):
        raise unittest.SkipTest("Skipping this test until implicit AD support works.")
        p = om.Problem()
        p.model.add_subsystem('G', SellarPrimalGrouped(), promotes=['*'])
        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(p.get_val('y2'), 12.05848819, .00001)

        assert_check_partials(p.check_partials(show_only_incorrect=True))

        p.run_model()
        assert_check_totals(p.check_totals(of=['obj', 'con1', 'con2'],
                                           wrt=['x', 'z'],
                                           method='fd',
                                           show_only_incorrect=True))

    def test_sellar_grouped(self):
        # Tests basic Newton solution on Sellar in a subgroup without jax

        prob = om.Problem(model=SellarDerivativesGrouped(nonlinear_solver=om.NewtonSolver(solve_subsystems=False),
                                                         linear_solver=om.ScipyKrylov))

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

        assert_check_partials(prob.check_partials(show_only_incorrect=True, abs_err_tol=2e-6), atol=2e-6)

        prob.run_model()
        assert_check_totals(prob.check_totals(of=['obj', 'con1', 'con2'],
                                           wrt=['x', 'z'],
                                           method='fd',
                                           show_only_incorrect=True), atol=2e-6)


if __name__ == '__main__':
    unittest.main()
