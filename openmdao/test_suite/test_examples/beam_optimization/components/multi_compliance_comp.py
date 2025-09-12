import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class MultiComplianceComp(om.ExplicitComponent):


    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_rhs = self.options['num_rhs']

        for j in range(num_rhs):
            self.add_input('displacements_%d' % j, shape=2 * num_nodes)
            self.add_output('compliance_%d' % j)
            force_vector = self.options['force_vector'][:, j]

            self.declare_partials('compliance_%d' % j, 'displacements_%d' % j,
                                  val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        num_rhs = self.options['num_rhs']

        for j in range(num_rhs):
            force_vector = self.options['force_vector'][:, j]
            outputs['compliance_%d' % j] = np.dot(force_vector, inputs['displacements_%d' % j])


class MultiComplianceCompOptions(ExplicitComponentOptions):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_elements: int = Field(default=0, desc='Number of beam elements')
    force_vector: np.ndarray = Field(default=np.zeros(0), desc='Force vector')
    num_rhs: int = Field(default=0, desc='Number of right-hand sides')


@dmm.register(MultiComplianceComp)
class MultiComplianceCompModel(ExplicitComponentModel):
    options: MultiComplianceCompOptions = Field(default_factory=MultiComplianceCompOptions)
