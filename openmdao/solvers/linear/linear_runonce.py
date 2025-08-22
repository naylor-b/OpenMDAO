"""Define the LinearRunOnce class."""

from openmdao.core.constants import _UNDEFINED
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS, _NonIterLinearBlockGSOptions


class LinearRunOnce(LinearBlockGS):
    """
    Simple linear solver that performs a single iteration of Guass-Seidel.

    This is done without iteration or norm calculation.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.
    """

    SOLVER = 'LN: RUNONCE'

    options = _NonIterLinearBlockGSOptions

    def solve(self, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.  Deprecated.
        """
        self._mode = mode

        self._update_rhs_vec()

        # Single iteration of GS
        self._single_iteration()

        # reset after solve is done
        self._scope_in = self._scope_out = _UNDEFINED
