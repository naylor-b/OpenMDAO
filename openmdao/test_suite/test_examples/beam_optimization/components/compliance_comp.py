import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class ComplianceComp(om.ExplicitComponent):


    def setup(self):
        num_nodes = self.options['num_elements'] + 1

        self.add_input('displacements', shape=2 * num_nodes)
        self.add_output('compliance')

    def setup_partials(self):
        num_nodes = self.options['num_elements'] + 1
        force_vector = self.options['force_vector']
        self.declare_partials('compliance', 'displacements',
                              val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        outputs['compliance'] = np.dot(self.options['force_vector'], inputs['displacements'])


class ComplianceCompOptions(ExplicitComponentOptions):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_elements: int = Field(default=0, desc='Number of beam elements')
    force_vector: np.ndarray = Field(default=np.zeros(0), desc='Force vector')


@dmm.register(ComplianceComp)
class ComplianceCompModel(ExplicitComponentModel):
    options: ComplianceCompOptions = Field(default_factory=ComplianceCompOptions)
