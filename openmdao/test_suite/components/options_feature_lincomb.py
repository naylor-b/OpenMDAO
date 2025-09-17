"""
A component that computes y = a*x + b, where a and b
are given as an option of type 'numpy.ScalarType'.
"""
import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class LinearCombinationComp(om.ExplicitComponent):


    def setup(self):
        self.add_input('x')
        self.add_output('y')

    def setup_partials(self):
        self.declare_partials('y', 'x', val=self.options['a'])

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['a'] * inputs['x'] + self.options['b']


class LinearCombinationCompOptions(_ExplicitComponentOptions):
    a: float = Field(default=1.0, desc='Linear coefficient')
    b: float = Field(default=1.0, desc='Constant offset')


@dmm.register(LinearCombinationComp)
class LinearCombinationCompModel(_ExplicitComponentModel):
    options: LinearCombinationCompOptions = Field(default_factory=LinearCombinationCompOptions)
