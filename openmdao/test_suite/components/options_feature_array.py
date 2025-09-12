"""
A component that multiplies an array by an input value, where
the array is given as an option of type 'numpy.ndarray'.
"""
import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class ArrayMultiplyComp(om.ExplicitComponent):


    def setup(self):
        array = self.options['array']

        self.add_input('x', 1.)
        self.add_output('y', shape=array.shape)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['array'] * inputs['x']


class ArrayMultiplyCompOptions(ExplicitComponentOptions):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray = Field(default=np.zeros(0), desc='Array to multiply by input')


@dmm.register(ArrayMultiplyComp)
class ArrayMultiplyCompModel(ExplicitComponentModel):
    options: ArrayMultiplyCompOptions = Field(default_factory=ArrayMultiplyCompOptions)
