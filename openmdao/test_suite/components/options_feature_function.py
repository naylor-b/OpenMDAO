"""
A component that computes y = func(x), where func
is a function given as an option.
"""

from types import FunctionType
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class UnitaryFunctionComp(om.ExplicitComponent):


    def setup(self):
        self.add_input('x')
        self.add_output('y')

    def setup_partials(self):
        self.declare_partials('y', 'x', method='fd')

    def compute(self, inputs, outputs):
        func = self.options['func']
        outputs['y'] = func(inputs['x'])


class UnitaryFunctionCompOptions(ExplicitComponentOptions):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    func: FunctionType = Field(default=None, desc='Function to apply to input')


@dmm.register(UnitaryFunctionComp)
class UnitaryFunctionCompModel(ExplicitComponentModel):
    options: UnitaryFunctionCompOptions = Field(default_factory=UnitaryFunctionCompOptions)
