import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class VolumeComp(om.ExplicitComponent):


    def setup(self):
        num_elements = self.options['num_elements']
        b = self.options['b']
        L = self.options['L']
        L0 = L / num_elements

        self.add_input('h', shape=num_elements)
        self.add_output('volume')

        self.declare_partials('volume', 'h', val=b * L0)

    def compute(self, inputs, outputs):
        L0 = self.options['L'] / self.options['num_elements']

        outputs['volume'] = np.sum(inputs['h'] * self.options['b'] * L0)


class VolumeCompOptions(ExplicitComponentOptions):
    num_elements: int = Field(default=0, desc='Number of beam elements')
    b: float = Field(default=1.0, desc='Width of the beam')
    L: float = Field(default=1.0, desc='Length of the beam')


@dmm.register(VolumeComp)
class VolumeCompModel(ExplicitComponentModel):
    options: VolumeCompOptions = Field(default_factory=VolumeCompOptions)
