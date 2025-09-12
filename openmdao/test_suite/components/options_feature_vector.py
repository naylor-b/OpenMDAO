"""
A component that multiplies a vector by 2, where the
size of the vector is given as an option of type 'int'.
"""
import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class VectorDoublingComp(om.ExplicitComponent):


    def setup(self):
        size = self.options['size']

        self.add_input('x', shape=size)
        self.add_output('y', shape=size)

    def setup_partials(self):
        size = self.options['size']
        self.declare_partials('y', 'x', val=2.,
                              rows=np.arange(size),
                              cols=np.arange(size))

    def compute(self, inputs, outputs):
        outputs['y'] = 2 * inputs['x']


class VectorDoublingCompOptions(ExplicitComponentOptions):
    size: int = Field(default=1, desc='Size of the vector')


@dmm.register(VectorDoublingComp)
class VectorDoublingCompModel(ExplicitComponentModel):
    options: VectorDoublingCompOptions = Field(default_factory=VectorDoublingCompOptions)
