"""
Total Jacobian class.
"""

from openmdao.jacobians.jacobian import Jacobian


class TotalJacobian(Jacobian):
    """
    A Jacobian containing total derivatives.

    Parameters
    ----------
    system : <System>
        The system that owns this jacobian.
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
        """
        super().__init__(system)
        self.shape = 0
        # self.shape = (len(system._outputs), len(system._outputs) + len(system._inputs))
