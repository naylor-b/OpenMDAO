import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class MomentOfInertiaComp(om.ExplicitComponent):


    def setup(self):
        num_elements = self.options['num_elements']

        self.add_input('h', shape=num_elements)
        self.add_output('I', shape=num_elements)

    def setup_partials(self):
        rows = cols = np.arange(self.options['num_elements'])
        self.declare_partials('I', 'h', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['I'] = 1./12. * self.options['b'] * inputs['h'] ** 3

    def compute_partials(self, inputs, partials):
        partials['I', 'h'] = 1./4. * self.options['b'] * inputs['h'] ** 2


class MomentOfInertiaCompOptions(ExplicitComponentOptions):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_elements: int = Field(default=5, desc='Number of beam elements')
    b: float = Field(default=0.1, desc='Width of the beam')


@dmm.register(MomentOfInertiaComp)
class MomentOfInertiaCompModel(ExplicitComponentModel):
    options: MomentOfInertiaCompOptions = Field(default_factory=MomentOfInertiaCompOptions)
