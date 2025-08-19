"""
Total Jacobian class.
"""

from openmdao.jacobians.jacobian import Jacobian


class TotalJacobian(Jacobian):
    """
    A Jacobian containing total derivatives.

    Parameters
    ----------
    of : list
        List of row names.
    wrt : list
        List of column names.
    driver : Driver
        The driver that owns this jacobian.

    Attributes
    ----------
    shape : tuple
        Full shape of the jacobian.
    """

    def __init__(self, system, of, wrt, driver):
        """
        Initialize the TotalJacobian.

        Parameters
        ----------
        of : list
            List of row names.
        wrt : list
            List of column names.
        driver : Driver
            The driver that owns this jacobian.
        """
        super().__init__(system)
        self.shape = 0
        # self.shape = (len(system._outputs), len(system._outputs) + len(system._inputs))
