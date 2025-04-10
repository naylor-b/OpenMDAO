"""
An ExplicitComponent that uses JAX for derivatives.
"""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.jax.jax_mixin import JaxExplicitMixin


class JaxExplicitComponent(JaxExplicitMixin, ExplicitComponent):
    """
    Base class for explicit components when using JAX for derivatives.

    Parameters
    ----------
    matrix_free : bool
        If True, this component will compute derivatives using matrix vector products.
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    pass
