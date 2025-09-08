"""
Define the NonlinearRunOnce class.

This is a simple nonlinear solver that just runs the system once.
"""

from pydantic import Field

from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver, _NonIterSolverOptions
from openmdao.utils.mpi import multi_proc_fail_check
from openmdao.solvers.solver import NonlinearSolverModel
from openmdao.utils.validation import DataModelManager as dmm


class NonlinearRunOnce(NonlinearSolver):
    """
    Simple solver that runs the containing system once.

    This is done without iteration or norm calculation.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.
    """

    SOLVER = 'NL: RUNONCE'

    def _solve_with_cache_check(self):
        self.solve()  # don't use caching

    def solve(self):
        """
        Run the solver.
        """
        system = self._system()

        with Recording('NLRunOnce', 0, self) as rec:
            # If this is a parallel group, transfer all at once then run each subsystem.
            if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd')

                with multi_proc_fail_check(system.comm):
                    for subsys in system._relevance.filter(system._subsystems_myproc):
                        subsys._solve_nonlinear()

            # If this is not a parallel group, transfer for each subsystem just prior to running it.
            else:
                self._gs_iter()

            rec.abs = 0.0
            rec.rel = 0.0


class _NonlinearRunOnceOptions(_NonIterSolverOptions):
    pass


@dmm.register(NonlinearRunOnce)
class NonlinearRunOnceModel(NonlinearSolverModel):
    options: _NonlinearRunOnceOptions = Field(default_factory=_NonlinearRunOnceOptions)
