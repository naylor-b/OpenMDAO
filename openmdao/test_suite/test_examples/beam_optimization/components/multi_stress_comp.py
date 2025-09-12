"""
Stress calculation component.

Simple calculation of beam bending stress assuming small angular displacements.
Vectorized for multiple load cases.
"""

import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponentOptions, ExplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm


class MultiStressComp(om.ExplicitComponent):


    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_rhs = self.options['num_rhs']

        self.add_input('h', shape=num_elements)

        for j in range(num_rhs):
            self.add_input('displacements_%d' % j, shape=2 * num_nodes)
            self.add_output('stress_%d' % j, shape=num_elements)

            self.declare_partials(of='stress_%d' % j, wrt='displacements_%d' % j)
            self.declare_partials(of='stress_%d' % j, wrt='h')

    def compute(self, inputs, outputs):
        num_rhs = self.options['num_rhs']
        tk = inputs['h'] * 0.5
        E = self.options['E']

        for j in range(num_rhs):
            ang = inputs['displacements_%d' % j][1::2]
            d_ang = ang[1:] - ang[0:-1]
            outputs['stress_%d' % j] = tk * E * d_ang

    def compute_partials(self, inputs, partials):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_states = num_nodes * 2
        num_rhs = self.options['num_rhs']
        tk = inputs['h'] * 0.5
        E = self.options['E']

        for j in range(num_rhs):
            ang = inputs['displacements_%d' % j][1::2]
            d_ang = ang[1:] - ang[0:-1]
            partials['stress_%d' % j, 'h'] = np.diag(0.5 * E * d_ang)

            J = np.zeros((num_elements, num_states))
            J[:, 1:-1:2] = -np.diag(tk * E)
            J[:, 3::2] += np.diag(tk * E)
            partials['stress_%d' % j, 'displacements_%d' % j] = J


class MultiStressCompOptions(ExplicitComponentOptions):
    num_elements: int = Field(default=0, desc='Number of beam elements')
    num_rhs: int = Field(default=0, desc='Number of right-hand sides')
    E: float = Field(default=0.0, desc='Young\'s modulus of the beam material')


@dmm.register(MultiStressComp)
class MultiStressCompModel(ExplicitComponentModel):
    options: MultiStressCompOptions = Field(default_factory=MultiStressCompOptions)
