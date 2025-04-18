
import numpy as np
from scipy.sparse import coo_matrix

try:
    import jax
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = np

from openmdao.core.explicitcomponent import ExplicitComponent


class SparsityComp(ExplicitComponent):
    def __init__(self, sparsity, **kwargs):
        super(SparsityComp, self).__init__(**kwargs)
        if isinstance(sparsity, np.ndarray):
            self.use_sparse = False
            self.sparsity = sparsity
            self.nzrows, self.nzcols = np.nonzero(self.sparsity)
        else:
            self.use_sparse = True
            self.sparsity = sparsity.tocoo()
            self.nzrows, self.nzcols = self.sparsity.row, self.sparsity.col

    def setup(self):
        self.add_input('x', shape=self.sparsity.shape[1])
        self.add_output('y', shape=self.sparsity.shape[0])

    def setup_partials(self):
        if self.use_sparse:
            self.declare_partials('y', 'x', rows=self.nzrows, cols=self.nzcols)
        else:
            self.declare_partials('y', 'x')

    def compute(self, inputs, outputs):
        outputs['y'] = self.sparsity @ inputs['x']

    def compute_partials(self, inputs, partials):
        if self.use_sparse:
            partials['y', 'x'] = self.sparsity.data
        else:
            partials['y', 'x'] = self.sparsity  # [self.nzrows, self.nzcols]




if __name__ == '__main__':
    from openmdao.test_suite.comp_tester import ComponentTester

    sparsity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sparsity_coo = coo_matrix(sparsity)
    jax_sparsity = jnp.array(sparsity)

    ComponentTester(SparsityComp, (sparsity,)).run()
    ComponentTester(SparsityComp, (sparsity_coo,)).run()

