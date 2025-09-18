import unittest

import numpy as np
from pydantic import Field, field_validator

import openmdao.api as om
from openmdao.test_suite.components.options_feature_array import ArrayMultiplyComp
from openmdao.test_suite.components.options_feature_function import UnitaryFunctionComp
from openmdao.test_suite.components.options_feature_lincomb import LinearCombinationComp
from openmdao.test_suite.components.options_feature_vector import VectorDoublingComp
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class TestOptions(unittest.TestCase):

    def test_simple(self):
        prob = om.Problem()
        prob.model.add_subsystem('double', VectorDoublingComp(size=3))  # 'size' is an option

        prob.setup()

        prob.set_val('double.x', [1., 2., 3.])

        prob.run_model()
        assert_near_equal(prob.get_val('double.y'), [2., 4., 6.])

    def test_simple_fail(self):
        prob = om.Problem()
        prob.model.add_subsystem('double', VectorDoublingComp())  # 'size' not specified

        try:
            prob.setup()
        except RuntimeError as err:
            self.assertEqual(str(err), "'double' <class VectorDoublingComp>: Option 'size' is required but has not been set.")

    def test_with_default(self):
        prob = om.Problem()
        prob.model.add_subsystem('linear', LinearCombinationComp(a=2.))  # 'b' not specified

        prob.setup()

        prob.set_val('linear.x', 3)

        prob.run_model()
        self.assertEqual(prob.get_val('linear.y'), 7.)

    def test_simple_array(self):
        prob = om.Problem()
        prob.model.add_subsystem('a_comp', ArrayMultiplyComp(array=np.array([1, 2, 3])))

        prob.setup()

        prob.set_val('a_comp.x', 5.)

        prob.run_model()
        assert_near_equal(prob.get_val('a_comp.y'), [5., 10., 15.])

    def test_simple_function(self):

        def my_func(x):
            return x*2

        prob = om.Problem()
        prob.model.add_subsystem('f_comp', UnitaryFunctionComp(func=my_func))

        prob.setup()

        prob.set_val('f_comp.x', 5.)

        prob.run_model()
        assert_near_equal(prob.get_val('f_comp.y'), 10.)

    def test_simple_values(self):

        class VectorDoublingComp(om.ExplicitComponent):

            def setup(self):
                size = self.options['size']

                self.add_input('x', shape=size)
                self.add_output('y', shape=size)
                self.declare_partials('y', 'x', val=2.,
                                      rows=np.arange(size),
                                      cols=np.arange(size))

            def compute(self, inputs, outputs):
                outputs['y'] = 2 * inputs['x']

        class VectorDoublingCompOptions(_ExplicitComponentOptions):
            size: int = Field(default=2, desc='Size of vector')
            
            @field_validator('size')
            @classmethod
            def _validate_size(cls, v):
                nset =  {2, 4, 6, 8}
                if v not in nset:
                    raise ValueError(f"Option 'size' is not one of {sorted(nset)}.")
                return v

        @dmm.register(VectorDoublingComp)
        class VectorDoublingCompModel(_ExplicitComponentModel):
            options: VectorDoublingCompOptions = Field(default_factory=VectorDoublingCompOptions)

        prob = om.Problem()
        prob.model.add_subsystem('double', VectorDoublingComp(size=4))

        prob.setup()

        prob.set_val('double.x', [1., 2., 3., 4.])

        prob.run_model()
        assert_near_equal(prob.get_val('double.y'), [2., 4., 6., 8.])

    def test_simple_bounds_valid(self):

        def check_even(name, value):
            if value % 2 != 0:
                raise ValueError(f"Option '{name}' with value {value} must be an even number.")

        class VectorDoublingCompOptions2(_ExplicitComponentOptions):
            size: int = Field(default=2, desc='Size of vector (must be even)')

            @field_validator('size')
            @classmethod
            def check_even(cls, value):
                if value % 2 != 0:
                    raise ValueError(f"Option 'size' with value {value} must be an even number.")

        class VectorDoublingComp(om.ExplicitComponent):

            def setup(self):
                size = self.options['size']

                self.add_input('x', shape=size)
                self.add_output('y', shape=size)
                self.declare_partials('y', 'x', val=2.,
                                      rows=np.arange(size),
                                      cols=np.arange(size))

            def compute(self, inputs, outputs):
                outputs['y'] = 2 * inputs['x']

        @dmm.register(VectorDoublingComp)
        class VectorDoublingCompModel(_ExplicitComponentModel):
            options: VectorDoublingCompOptions2 = Field(default_factory=VectorDoublingCompOptions2)

        try:
            VectorDoublingComp(size=5)
        except Exception as err:
            self.assertTrue("size\n  Value error, Option 'size' with value 5 must be an even number." in str(err))


if __name__ == "__main__":
    unittest.main()
