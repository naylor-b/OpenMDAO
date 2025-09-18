import unittest
from pydantic import Field, field_validator

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.units import convert_units
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class AviaryComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('x', 3.0)
        self.add_output('y', 3.0)

    def compute(self, inputs, outputs):
        length = self.options['length'][0]

        x = inputs['x']
        outputs['y'] = length * x


class AviaryCompOptions(_ExplicitComponentOptions):
    length: tuple = Field(default=(12.0, 'inch'), desc='Length with units')
    
    @field_validator('length')
    @classmethod
    def _validate_length(cls, v):
        _, units = cls.model_fields['length'].default
        new_val, new_units = v
        return (convert_units(new_val, new_units, units), units)



@dmm.register(AviaryComp)
class AviaryCompModel(_ExplicitComponentModel):
    options: AviaryCompOptions = Field(default_factory=AviaryCompOptions)



class Fakeviary(om.Group):

    def setup(self):
        self.add_subsystem('mass', AviaryComp())



class TestOptionsUnits(unittest.TestCase):

    def test_simple(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('statics', Fakeviary())

        prob.model_options['*'] = {'length': (2.0, 'ft')}
        prob.setup()

        prob.run_model()

        y = prob.get_val('statics.mass.y')
        assert_near_equal(y, 72, 1e-6)




if __name__ == "__main__":
    unittest.main()
