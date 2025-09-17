"""
Distributed components.

Components that are used in multiple places for testing distributed components.
"""
import numpy as np
from pydantic import Field, ConfigDict

import openmdao.api as om
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.validation import DataModelManager as dmm


class DistribComp(om.ExplicitComponent):
    """Simple Distributed Component."""


    def setup(self):
        comm = self.comm
        rank = comm.rank

        size = self.options['size']

        # if comm.size is 2 and size is 15, this results in
        # 8 entries for proc 0 and 7 entries for proc 1
        sizes, _ = evenly_distrib_idxs(comm.size, size)
        mysize = sizes[rank]

        self.add_input('invec', np.ones(mysize, float), distributed=True)
        self.add_output('outvec', np.ones(mysize, float), distributed=True,)

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * -3.0

class DistribCompOptions(_ExplicitComponentOptions):
    size: int = Field(default=1, desc="Size of input and output vectors.")


@dmm.register(DistribComp)
class DistribCompModel(_ExplicitComponentModel):
    options: DistribCompOptions = Field(default_factory=DistribCompOptions)


class Summer(om.ExplicitComponent):
    """Sums an input array."""

    def setup(self):
        self.add_input('invec', np.ones(self.options['size'], float))

        self.add_output('sum', 0.0, shape=1)

    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['invec'])


class SummerOptions(_ExplicitComponentOptions):
    size: int = Field(default=1, desc="Size of input and output vectors.")

@dmm.register(Summer)
class SummerModel(_ExplicitComponentModel):
    options: SummerOptions = Field(default_factory=SummerOptions)



class DistribCompDerivs(om.ExplicitComponent):
    """Simple Distributed Component with Derivatives."""

    def setup(self):
        comm = self.comm
        rank = comm.rank

        size = self.options['size']

        # if comm.size is 2 and size is 15, this results in
        # 8 entries for proc 0 and 7 entries for proc 1
        sizes, _ = evenly_distrib_idxs(comm.size, size)
        self.mysize = mysize = sizes[rank]

        # don't set src_indices on the input, just use default behavior
        self.add_input('invec', np.ones(mysize, float), distributed=True)
        self.add_output('outvec', np.ones(mysize, float), distributed=True)

    def setup_partials(self):
        # declare partial derivatives (diagonal of mysize)
        self.declare_partials('outvec', 'invec',
                              rows=np.arange(0, self.mysize),
                              cols=np.arange(0, self.mysize))

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * -3.0

    def compute_partials(self, inputs, J):
        # get mysize from the input vector for this process
        mysize = inputs['invec'].size

        if self.comm.rank == 0:
            J['outvec', 'invec'] = np.ones((mysize,)) * 2.0
        else:
            J['outvec', 'invec'] = np.ones((mysize,)) * -3.0

class DistribCompDerivsOptions(_ExplicitComponentOptions):
    size: int = Field(default=1, desc="Size of input and output vectors.")


@dmm.register(DistribCompDerivs)
class DistribCompDerivsModel(_ExplicitComponentModel):
    options: DistribCompDerivsOptions = Field(default_factory=DistribCompDerivsOptions)


class SummerDerivs(om.ExplicitComponent):
    """Sums an input array."""

    def setup(self):
        self.add_input('invec', np.ones(self.options['size'], float))

        self.add_output('sum', 0.0, shape=1)

    def setup_partials(self):
        # the derivative is constant
        self.declare_partials('sum', 'invec', val=1.)

    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['invec'])


class SummerDerivsOptions(_ExplicitComponentOptions):
    size: int = Field(default=1, desc="Size of input and output vectors.")

@dmm.register(SummerDerivs)
class SummerDerivsModel(_ExplicitComponentModel):
    options: SummerDerivsOptions = Field(default_factory=SummerDerivsOptions)


if __name__ == "__main__":
    s = SummerDerivs(size=10)