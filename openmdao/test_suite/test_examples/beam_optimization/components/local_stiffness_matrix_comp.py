import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class LocalStiffnessMatrixComp(om.ExplicitComponent):


    def setup(self):
        num_elements = self.options['num_elements']
        E = self.options['E']
        L = self.options['L']

        self.add_input('I', shape=num_elements)
        self.add_output('K_local', shape=(num_elements, 4, 4))

        L0 = L / num_elements
        coeffs = np.empty((4, 4))
        coeffs[0, :] = [12, 6 * L0, -12, 6 * L0]
        coeffs[1, :] = [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2]
        coeffs[2, :] = [-12, -6 * L0, 12, -6 * L0]
        coeffs[3, :] = [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]
        coeffs *= E / L0 ** 3

        self.mtx = np.zeros((num_elements, 4, 4, num_elements))
        for ind in range(num_elements):
            self.mtx[ind, :, :, ind] = coeffs

        self.declare_partials('K_local', 'I',
            val=self.mtx.reshape(16 * num_elements, num_elements))

    def compute(self, inputs, outputs):
        outputs['K_local'] = 0
        for ind in range(self.options['num_elements']):
            outputs['K_local'][ind, :, :] = self.mtx[ind, :, :, ind] * inputs['I'][ind]


class LocalStiffnessMatrixCompOptions(_ExplicitComponentOptions):
    num_elements: int = Field(default=5, desc='Number of beam elements')
    E: float = Field(default=1.0, desc='Young\'s modulus of the beam material')
    L: float = Field(default=1.0, desc='Length of the beam')


@dmm.register(LocalStiffnessMatrixComp)
class LocalStiffnessMatrixCompModel(_ExplicitComponentModel):
    options: LocalStiffnessMatrixCompOptions = Field(default_factory=LocalStiffnessMatrixCompOptions)
