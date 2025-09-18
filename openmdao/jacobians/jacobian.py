"""Define the base Jacobian class."""
import numpy as np

from openmdao.utils.iter_utils import meta2range_iter
from openmdao.utils.rangemapper import TwoWayRangeMapper
from openmdao.utils.general_utils import do_nothing_context
from openmdao.utils.coloring import _ColSparsityJac


def _get_vec_slices(system, iotype, subset=None):
    return {
        name: slice(start, end) for name, start, end in
        meta2range_iter(system._var_abs2meta[iotype].items(), subset=subset)
    }


# Design Notes:
# - When Components declare partials, they are stored as metadata in the _subjacs_info dict.
# - These _subjacs_info entries may be used by multiple Jacobians at higher levels of the System
#   hierarchy.
# - Jacobian objects contain Subjac objects, which wrap the _subjacs_info metadata and add context
#   like row and column slices specific to a their owning Jacobian.


class Jacobian(object):
    """
    Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _subjacs : dict
        Dictionary of the relevant sub-Jacobian objects keyed by absolute names.
    _irrelevant_subjacs : dict
        Dictionary of the irrelevant sub-Jacobian objects keyed by absolute names.
    _under_complex_step : bool
        When True, this Jacobian is under complex step, using a complex jacobian.
    _col_mapper : TwoWayRangeMapper
        Maps variable names to column indices and vice versa.
    _problem_meta : dict
        Problem metadata.
    _resolver : <Resolver>
        Resolver for this system.
    _output_slices : dict
        Maps output names to slices of the output vector.
    _input_slices : dict
        Maps input names to slices of the input vector.
    _has_approx : bool
        Whether the system has an approximate jacobian.
    _ordered_subjac_keys : list
        List of subjac keys in order of appearance.
    _initialized : bool
        Whether the jacobian has been initialized.
    dtype : dtype
        The dtype of the jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        self._subjacs = None
        self._under_complex_step = False
        self._col_mapper = None
        self._problem_meta = system._problem_meta
        self._resolver = system._resolver
        self._output_slices = _get_vec_slices(system, 'output')
        self._input_slices = _get_vec_slices(system, 'input')
        self._has_approx = system._has_approx
        self._ordered_subjac_keys = None
        self._initialized = False
        self.dtype = system._outputs.dtype

    def _pre_update(self, dtype):
        """
        Pre-update the jacobian.

        Parameters
        ----------
        dtype : dtype
            The dtype of the jacobian.
        """
        if dtype.kind != self.dtype.kind:
            self.dtype = dtype

            # if _subjacs is None, our system hasn't been linearized
            if self._subjacs is not None:
                for subjac in self._subjacs.values():
                    subjac.set_dtype(dtype)

    def _post_update(self):
        """
        Post-update the jacobian.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        pass

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        pass

    def _get_abs_key(self, key):
        try:
            return self._abs_keys[key]
        except KeyError:
            abskey = self._resolver.any2abs_key(key)
            if abskey is not None:
                self._abs_keys[key] = abskey
            return abskey

    def get_metadata(self, key):
        """
        Get metadata for the given key.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        dict
            Metadata dict for the given key.
        """
        try:
            return self._subjacs_info[self._get_abs_key(key)]
        except KeyError:
            raise KeyError(f'Variable name pair {key} not found.')

    def __contains__(self, key):
        """
        Return whether there is a subjac for the given promoted or relative name pair.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        bool
            return whether sub-Jacobian has been defined.
        """
        return self._get_abs_key(key) in self._subjacs_info

    def __iter__(self):
        """
        Yield next name pair of sub-Jacobian.

        Yields
        ------
        str
        """
        yield from self._subjacs.keys()

    def keys(self):
        """
        Yield next name pair of sub-Jacobian.

        Yields
        ------
        str
        """
        yield from self._subjacs.keys()

    def items(self):
        """
        Yield name pair and value of sub-Jacobian.

        Yields
        ------
        str
        """
        for key, subjac in self._subjacs.items():
            yield key, subjac.info['val']

    def _apply(self, system, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        raise NotImplementedError(f"Class {type(self).__name__} does not implement _apply.")

    def _setup_index_maps(self, system):
        namesize_iter = [(n, end - start) for n, start, end, _, _, _ in system._get_jac_wrts()]
        self._col_mapper = TwoWayRangeMapper.create(namesize_iter)

    def set_col(self, system, icol, column):
        """
        Set a column of the jacobian.

        The column is assumed to be the same size as a column of the jacobian.

        This also assumes that the column does not attempt to set any nonzero values that are
        outside of specified sparsity patterns for any of the subjacs.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        icol : int
            Column index.
        column : ndarray
            Column value.
        """
        if self._col_mapper is None:
            self._setup_index_maps(system)

        wrt, loc_idx = self._col_mapper.get_key_rel(icol)  # local col index into subjacs

        subjacs = self._subjacs

        for of, start, end, _, _ in system._get_jac_ofs():
            key = (of, wrt)
            if key in subjacs:
                subjacs[key].set_col(loc_idx, column[start:end])

    def set_csc_jac(self, system, jac):
        """
        Assign a CSC jacobian to this jacobian.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        jac : csc_matrix
            CSC jacobian.
        """
        ofiter = list(system._get_jac_ofs())
        for wrt, wstart, wend, _, _, _ in system._get_jac_wrts():
            wjac = jac[:, wstart:wend]
            for of, start, end, _, _ in ofiter:
                key = (of, wrt)
                if key in self._subjacs_info:
                    subjac = self.get_metadata(key)
                    if subjac['cols'] is None:  # dense
                        subjac['val'][:, :] = wjac[start:end, :].toarray()
                    else:  # our COO format
                        subj = wjac[start:end, :]
                        subjac['val'][:] = subj[subjac['rows'], subjac['cols']]

    def set_dense_jac(self, system, jac):
        """
        Assign a dense jacobian to this jacobian.

        This assumes that any column does not attempt to set any nonzero values that are
        outside of specified sparsity patterns for any of the subjacs.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        jac : ndarray
            Dense jacobian.
        """
        if self._col_mapper is None:
            self._setup_index_maps(system)

        wrtiter = list(system._get_jac_wrts())
        for of, start, end, _, _ in system._get_jac_ofs():
            for wrt, wstart, wend, _, _, _ in wrtiter:
                key = (of, wrt)
                if key in self._subjacs_info:
                    subjac = self.get_metadata(key)
                    if subjac['cols'] is None:  # dense
                        subjac['val'][:, :] = jac[start:end, wstart:wend]
                    else:  # our COO format
                        subj = jac[start:end, wstart:wend]
                        subjac['val'][:] = subj[subjac['rows'], subjac['cols']]

    def _reset_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._initialized = False
        self._subjacs = None
        self._irrelevant_subjacs = {}
        self._get_subjacs(system)
        self._col_mapper = None  # force recompute of internal index maps on next set_col

    def _get_ordered_subjac_keys(self, system):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        list
            List of keys matching this jacobian for the current system.
        """
        relevance = None
        if self._ordered_subjac_keys is None:
            relevance = self._problem_meta['relevance']
            is_relevant = relevance.is_relevant
            active = system.linear_solver is None or system.linear_solver.use_relevance()
            if not active or not relevance._active:
                relevance = None

            subjacs_info = self._subjacs_info
            keys = []
            # determine the set of remote keys (keys where either of or wrt is remote somewhere)
            # only if we're under MPI with comm size > 1 and the given system is a Group that
            # computes its derivatives using finite difference or complex step.
            if system.pathname and system.comm.size > 1 and system._owns_approx_jac:
                ofnames = system._var_allprocs_abs2meta['output']
                wrtnames = system._var_allprocs_abs2meta
            else:
                ofnames = system._var_abs2meta['output']
                wrtnames = system._var_abs2meta

            with relevance.active(active) if relevance else do_nothing_context():
                with relevance.all_seeds_active() if relevance else do_nothing_context():
                    for of in ofnames:
                        for type_ in ('output', 'input'):
                            for wrt in wrtnames[type_]:
                                key = (of, wrt)
                                if key in subjacs_info:
                                    if relevance is not None and (not is_relevant(wrt) or
                                                                  not is_relevant(of)):
                                        continue
                                    keys.append(key)

            self._ordered_subjac_keys = keys

        return self._ordered_subjac_keys

    def todense(self):
        """
        Return a dense version of the full jacobian.

        This includes the combined dr/do and dr/di matrices.

        Returns
        -------
        ndarray
            Dense version of the full jacobian.
        """
        # get shapes of dr/do and dr/di
        drdo_shape = (self.shape[0], self.shape[0])
        drdi_shape = (self.shape[0], self.shape[1] - self.shape[0])

        J_dr_do = np.zeros(drdo_shape)
        J_dr_di = np.zeros(drdi_shape)

        lst = [J_dr_do, J_dr_di]

        for key, subjac in self._subjacs.items():
            if key[1] in self._output_slices:
                J_dr_do[subjac.row_slice, subjac.col_slice] = subjac.todense()
            else:
                J_dr_di[subjac.row_slice, subjac.col_slice] = subjac.todense()

        return np.hstack(lst)


class JacobianUpdateContext:
    """
    Within this context, the Jacobian may be updated.

    Ways to update:
        - __setitem__, during component compute_jacvec_product or linearize
        - set_col, during computation of approximate derivatives
        - set_dense_jac, during linearization of jax components

    Parameters
    ----------
    system : System
        The system that owns this jacobian.

    Attributes
    ----------
    system : System
        The system that owns this jacobian.
    jac : Jacobian
        The jacobian that is being updated.
    """

    def __init__(self, system):
        """
        Initialize the context.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        """
        self.system = system
        self.jac = None

    def __enter__(self):
        """
        Enter the context.

        Returns
        -------
        Jacobian
            The jacobian that is being updated.
        """
        self.jac = self.system._get_jacobian()

        if self.jac is not None:
            self.jac._pre_update(self.system._outputs.dtype)

        return self.jac

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.

        Parameters
        ----------
        exc_type : type
            The type of the exception.
        exc_val : Exception
            The exception object.
        exc_tb : traceback
            The traceback object.
        """
        if self.jac is not None:
            self.jac._update(self.system)
            self.jac._post_update()

        if exc_type:
            self.jac = self.system._jacobian = None
            return False  # Re-raise the exception after logging/handling


class GroupJacobianUpdateContext:
    """
    Within this context, the Jacobian may be updated.

    Ways to update:
        - set_col, during computation of approximate derivatives
        - full subjac update after recursive linearization of children

    Parameters
    ----------
    group : Group
        The group that owns this jacobian.

    Attributes
    ----------
    group : Group
        The group that owns this jacobian.
    jac : Jacobian
        The jacobian that is being updated.
    """

    def __init__(self, group):
        """
        Initialize the context.

        Parameters
        ----------
        group : Group
            The group that owns this jacobian.
        """
        self.group = group
        self.jac = None

    def __enter__(self):
        """
        Enter the context.

        Returns
        -------
        Jacobian
            The jacobian that is being updated.
        """
        if self.group._owns_approx_jac:
            if self.group._tot_jac is not None and not isinstance(self.group._jacobian,
                                                                  _ColSparsityJac):
                self.jac = self.group._jacobian = self.group._tot_jac
            else:
                self.jac = self.group._jacobian = self.group._get_jacobian()

        else:
            self.jac = self.group._get_assembled_jac()

        if self.jac is not None:
            self.jac._pre_update(self.group._outputs.dtype)

        return self.jac  # may be None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.

        Parameters
        ----------
        exc_type : type
            The type of the exception.
        exc_val : Exception
            The exception object.
        exc_tb : traceback
            The traceback object.
        """
        if self.jac is not None:
            self.jac._update(self.group)
            self.jac._post_update()

        if exc_type:
            self.jac = self.group._jacobian = None
            return False  # Re-raise the exception after logging/handling
