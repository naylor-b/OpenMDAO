"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import sys
import inspect
from itertools import chain
from types import MethodType
from functools import partial

import numpy as np
from scipy.sparse import coo_matrix

from openmdao.utils.code_utils import get_function_deps, get_return_names
from openmdao.utils.om_warnings import issue_warning
import openmdao.utils.coloring as coloring_mod
from openmdao.jax.jax_utils import jax, jit, _ensure_returns_tuple, _jax_register_pytree_class, \
    get_vmap_tangents, _update_subjac_sparsity, _jax2np, _compute_output_shapes


class JaxMixin(object):
    """
    Mixin class for components that use jax.

    Parameters
    ----------
    matrix_free : bool
        If True, this component will compute derivatives using matrix vector products.
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.

    Attributes
    ----------
    matrix_free : bool
        If True, this component will compute derivatives using matrix vector products.
    _tangents : dict
        The tangents for the inputs and outputs.
    _sparsity : coo_matrix or None
        The sparsity of the Jacobian.
    _jac_func_ : function or None
        The function that computes the jacobian.
    _jac_colored_ : function or None
        The function that computes the colored jacobian.
    _static_hash : tuple
        The hash of the static values.
    _orig_compute_primal : function
        The original compute_primal method.
    _ret_tuple_compute_primal : function
        The compute_primal method that returns a tuple.
    """

    def __init__(self, matrix_free=False, fallback_derivs_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxExplicitComponent requires Python 3.9 or newer.")

        super().__init__(**kwargs)
        self.matrix_free = matrix_free

        self._re_init_jax()

        if self.compute_primal is None:
            raise RuntimeError(f"{self.msginfo}: compute_primal is not defined for this component.")

        self._setup_compute_primal()

        # if derivs_method is explicitly passed in, just use it
        if 'derivs_method' in kwargs and kwargs['derivs_method'] != 'jax':
            return

        if jax:
            self.options['derivs_method'] = 'jax'
        else:
            issue_warning(f"{self.msginfo}: JAX is not available, so '{fallback_derivs_method}' "
                          "will be used for derivatives.")
            self.options['derivs_method'] = fallback_derivs_method

    def _setup_compute_primal(self):
        """
        Set up the compute_primal method.
        """
        self._orig_compute_primal = self.compute_primal
        self._ret_tuple_compute_primal = \
            MethodType(_ensure_returns_tuple(self.compute_primal.__func__), self)
        self.compute_primal = self._ret_tuple_compute_primal

    def _re_init_jax(self):
        """
        Re-initialize the component for a new run.
        """
        self._tangents = {'fwd': None, 'rev': None}
        self._do_sparsity = False
        self._sparsity = None
        self._jac_func_ = None
        self._jac_func_nojit = None
        self._static_hash = None
        self._jac_colored_ = None
        self._output_shapes = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        self.options.declare('default_to_dyn_shapes', types=bool, default=False,
                             desc='If True, use dynamic shaping for any variables whose value is '
                             'scalar and whose shape is not explicitly set. Inputs will use '
                             'shape_by_conn and outputs will use a compute_shape method based '
                             'on jax.eval_shape. Default is False.')

        self.options.undeclare("distributed")

    def _get_jax_compute_primal(self, discrete_inputs, need_jit):
        """
        Get the jax version of the compute_primal method.
        """
        compute_primal = self._ret_tuple_compute_primal.__func__

        if need_jit:
            # jit the compute_primal method
            static_argnums = self._get_static_argnums(discrete_inputs)
            compute_primal = jit(compute_primal, static_argnums=static_argnums)

        return MethodType(compute_primal, self)

    def _setup_check(self):
        """
        Check if inputs and outputs have been added, and if not, determine them from compute_primal.
        """
        self._re_init_jax()

        if len(self._var_rel_names['input']) > 0 or len(self._var_rel_names['output']) > 0:
            return

        if not self._var_rel_names['input']:
            for argname in inspect.signature(self._orig_compute_primal).parameters:
                self.add_input(argname)

        if not self._var_rel_names['output']:
            for i, name in enumerate(get_return_names(self._orig_compute_primal)):
                if name is None:
                    name = f'out_{i}'
                self.add_output(name)

    def add_input(self, name, **kwargs):
        """
        Add an input to the component.

        This overrides the base class method to update the kwargs to use dynamic shaping by
        default.

        Parameters
        ----------
        name : str
            The name of the input.
        **kwargs : dict
            The kwargs to pass to the base class method.
        """
        super().add_input(name, **self._update_add_input_kwargs(**kwargs))

    def add_output(self, name, **kwargs):
        """
        Add an output to the component.

        This overrides the base class method to update the kwargs to use dynamic shaping by
        default.

        Parameters
        ----------
        name : str
            The name of the output.
        **kwargs : dict
            The kwargs to pass to the base class method.
        """
        super().add_output(name, **self._update_add_output_kwargs(name, **kwargs))

    def _setup_jax(self):
        """
        Set up the jax interface for this component.

        This happens in final_setup after all var sizes and partials are set.
        """
        _jax_register_pytree_class(self.__class__)

        if not self._discrete_inputs and not self.get_self_statics():
            # avoid unnecessary statics checks
            self._statics_changed = self._statics_noop

        self.compute_primal = self._get_jax_compute_primal(self._discrete_inputs,
                                                           self.options['use_jit'])

    def _check_first_linearize(self):
        if self._first_call_to_linearize:
            self._first_call_to_linearize = False  # only do this once
            if not self.matrix_free and self._coloring_info.use_coloring() and \
                    coloring_mod._use_partial_sparsity:
                self._get_coloring()
                if self._jacobian is not None:
                    self._jacobian._restore_approx_sparsity()
            elif self._do_sparsity and self.options['derivs_method'] == 'jax':
                self.compute_sparsity()

    def _statics_changed(self, discrete_inputs):
        """
        Determine if jitting is needed based on changes in static values since the last call.

        Parameters
        ----------
        discrete_inputs : dict
            dict containing discrete input values.

        Returns
        -------
        bool
            Whether jitting is needed.
        """
        # if static values change, we need to rejit
        inhash = hash((tuple(discrete_inputs) if discrete_inputs else (), self.get_self_statics()))
        if inhash != self._static_hash:
            self._static_hash = inhash
            return True
        return False

    def _statics_noop(self, discrete_inputs):
        """
        Use this function if the component has no discrete inputs or self statics.

        Parameters
        ----------
        discrete_inputs : dict
            dict containing discrete input values.

        Returns
        -------
        bool
            Always returns False.
        """
        return False

    def _get_static_argnums(self, discrete_inputs):
        """
        Get the static argnums for the compute_primal method.
        """
        idx = self._get_num_differentiable_args() + 1
        if discrete_inputs:
            return list(range(idx, idx + len(discrete_inputs)))

    def _get_differentiable_argnums(self):
        """
        Get the argnums for the compute_primal method that are differentiable.
        """
        return list(range(self._get_num_differentiable_args()))

    def declare_coloring(self, **kwargs):
        """
        Declare coloring for this component.

        The 'method' argument is set to options['derivs_method'] and passed to the base class.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to the base class.
        """
        if 'method' in kwargs and kwargs['method'] != self.options['derivs_method']:
            raise ValueError(f"method must be '{self.options['derivs_method']}' for this component "
                             "but got '{kwargs['method']}'.")
        kwargs['method'] = self.options['derivs_method']
        super().declare_coloring(**kwargs)

    def _update_jac_functs(self, discrete_inputs):
        """
        Update the jax function that computes the jacobian for this component if necessary.

        An update is required if jitting is enabled and any static values have changed.

        Parameters
        ----------
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.

        Returns
        -------
        tuple
            The jax functions (jax_compute_primal, jax_compute_jac). Note that these are not
            methods, but rather functions. To make them methods you need to assign
            MethodType(function, self) to an attribute of the instance.
        """
        need_jit = self.options['use_jit']
        if need_jit and self._statics_changed(discrete_inputs):
            self._jac_func_ = None

        if self._jac_func_ is None:
            differentiable_cp = self._get_differentiable_compute_primal(discrete_inputs)

            if self._coloring_info.use_coloring():
                if self._coloring_info.coloring is None:
                    # need to dynamically compute the coloring first
                    self._compute_coloring()

                if self.best_partial_deriv_direction() == 'fwd':
                    self._get_tangents('fwd', self._coloring_info.coloring)

                    # here we'll use the same inputs and a single tangent vector from the vmap
                    # batch to compute a single jvp, which corresponds to a column of the
                    # jacobian (the compressed jacobian in the colored case).
                    def jvp_at_point(tangent, icontvals):
                        # [1] is the derivative, [0] is the primal (we don't need the primal)
                        return jax.jvp(differentiable_cp, icontvals, tangent)[1]

                    # vectorize over the last axis of the tangent vectors and use the same
                    # inputs for all cases.
                    self._jac_func_ = jax.vmap(jvp_at_point, in_axes=[-1, None], out_axes=-1)
                    self._jac_colored_ = self._jacfwd_colored
                else:  # rev
                    def vjp_at_point(cotangent, icontvals):
                        # Returns primal and a function to compute VJP so just take [1],
                        # the vjp function
                        return jax.vjp(differentiable_cp, *icontvals)[1](cotangent)

                    self._get_tangents('rev', self._coloring_info.coloring)

                    # Batch over last axis of cotangents
                    self._jac_func_ = jax.vmap(vjp_at_point, in_axes=[-1, None], out_axes=-1)
                    self._jac_colored_ = self._jacrev_colored
            else:
                self._jac_colored_ = None
                fjax = jax.jacfwd if self.best_partial_deriv_direction() == 'fwd' else jax.jacrev
                self._jac_func_ = fjax(differentiable_cp,
                                       argnums=self._get_differentiable_argnums())

            if need_jit:
                self._jac_func_ = jax.jit(self._jac_func_)

    def _get_differentiable_compute_primal(self, discrete_inputs):
        """
        Get the compute_primal function for the jacobian.

        This version of the compute primal should take no discrete inputs and return no discrete
        outputs. It will be called when computing the jacobian.

        Parameters
        ----------
        self : Component
            The component to get the compute_primal function for.
        discrete_inputs : iter of discrete values
            The discrete input values.

        Returns
        -------
        function
            The compute_primal function to be used to compute the jacobian.
        """
        # exclude the discrete inputs from the inputs and the discrete outputs from the outputs
        if discrete_inputs:
            if self._discrete_outputs:
                ncontouts = self._outputs.nvars()

                def differentiable_compute_primal(*contvals):
                    return self._ret_tuple_compute_primal(*contvals, *discrete_inputs)[:ncontouts]

            else:

                def differentiable_compute_primal(*contvals):
                    return self._ret_tuple_compute_primal(*contvals, *discrete_inputs)

            return differentiable_compute_primal

        elif self._discrete_outputs:
            ncontouts = self._outputs.nvars()

            def differentiable_compute_primal(*contvals):
                return self._ret_tuple_compute_primal(*contvals)[:ncontouts]

            return differentiable_compute_primal

        return self._ret_tuple_compute_primal

    def _get_tangents(self, direction, coloring=None):
        """
        Get the tangents for the inputs or outputs.

        If coloring is not None, then the tangents will be compressed based on the coloring.

        Parameters
        ----------
        direction : str
            The direction to get the tangents for.
        coloring : Coloring
            The coloring to use.

        Returns
        -------
        tuple
            The tangents.
        """
        if self._tangents[direction] is None:
            if direction == 'fwd':
                args = tuple(self._get_compute_primal_invals(include_discrete=False))
                self._tangents[direction] = get_vmap_tangents(args, direction, fill=1.,
                                                              coloring=coloring)
            else:
                self._tangents[direction] = get_vmap_tangents(tuple(self._outputs.values()),
                                                              direction, fill=1., coloring=coloring)
        return self._tangents[direction]

    def _compute_sparsity(self, direction=None, num_iters=1, perturb_size=1e-9, use_nan=False):
        """
        Compute the sparsity of the Jacobian using jvp/vjp with nans for the seeds.

        Parameters
        ----------
        self : Component
            The component to compute the sparsity for.
        direction : str or None
            The direction to compute the sparsity in.  If None, the best direction is chosen based
            on the number of inputs and outputs.  If a str, it must be 'fwd' or 'rev'.
        num_iters : int
            The number of times to run the perturbation iteration.
        perturb_size : float
            The size of the perturbation to use.
        use_nan : bool
            If True, use nans for the seeds.

        Returns
        -------
        coo_matrix, dict
            The boolean sparsity matrix and info.
        """
        if direction is None:
            direction = self.best_partial_deriv_direction()

        assert direction in ['fwd', 'rev']

        implicit = not self.is_explicit()

        if implicit:
            pvecs = (self._outputs, self._inputs)
            wrtsize = len(self._outputs) + len(self._inputs)
            save_vecs = (self._residuals,)
        else:
            pvecs = (self._inputs,)
            wrtsize = len(self._inputs)
            save_vecs = (self._outputs, self._residuals,)

        sparsity = None
        idiscvals = tuple(self._discrete_inputs.values())

        # exclude the discrete inputs from the inputs and the discrete outputs from the outputs
        differentiable_part = self._get_differentiable_compute_primal(idiscvals)

        # when computing tangents we only care about shapes of the values, not the values
        # themselves, so we can use the unperturbed values for the tangents
        icontvals = tuple(self._get_compute_primal_invals(include_discrete=False))
        if direction == 'fwd':
            tangents = get_vmap_tangents(icontvals, 'fwd', fill=np.nan if use_nan else 1.)

            def jvp_at_point(tangent, contvals):
                # [1] is the derivative, [0] is the primal (we don't need the primal)
                return jax.jvp(differentiable_part, contvals, tangent)[1]

            Jfunc = jax.vmap(jvp_at_point, in_axes=[-1, None], out_axes=-1)
        else:
            # these are really cotangents
            tangents = get_vmap_tangents(tuple(self._outputs.values()), 'rev',
                                         fill=np.nan if use_nan else 1.)

            def vjp_at_point(cotangent, contvals):
                return jax.vjp(differentiable_part, *contvals)[1](cotangent)

            # vectorize over last axis of cotangents
            Jfunc = jax.vmap(vjp_at_point, in_axes=[-1, None], out_axes=-1)

        if self.options['use_jit']:
            Jfunc = jax.jit(Jfunc)

        sparsity = np.zeros((len(self._outputs), wrtsize))

        for _ in self._perturbation_iter(num_iters=num_iters, perturb_size=perturb_size,
                                         perturb_vecs=pvecs, save_vecs=save_vecs):
            self._apply_nonlinear()

            J = Jfunc(tangents, tuple(self._get_compute_primal_invals(include_discrete=False)))

            if not isinstance(J, tuple):
                J = (J,)

            if len(J) == 1:
                J = J[0]
                if len(J.shape) > 2:
                    # flatten 'variable' dimensions.  Last dimension is the batching dimension.
                    J = J.reshape(np.prod(J.shape[:-1], dtype=int), J.shape[-1])
                elif len(J.shape) == 1:
                    J = np.atleast_2d(J)
            else:
                # flatten 'variable' dimensions for each variable.  Last dimension is the batching
                # dimension.  Then vertically stack all the flattened 'variable' arrays.
                J = np.vstack([j.reshape(np.prod(j.shape[:-1], dtype=int), j.shape[-1]) for j in J])

            if direction != 'fwd':
                J = J.T

            if sparsity is None:
                sparsity[:, :] = np.abs(J)
            else:
                sparsity[:, :] += np.abs(J)

        if implicit:
            # we need to swap input and output cols because OpenMDAO jacs have output wrts first
            # followed by input wrts but compute_primal takes inputs first followed by outputs
            sparsity = np.hstack((sparsity[:, -len(self._outputs):],
                                  sparsity[:, :len(self._inputs)]))

        nz = np.nonzero(sparsity)
        data = np.ones(len(nz[0]), dtype=bool)
        sparsity = coo_matrix((data, nz), shape=sparsity.shape)

        info = {
            'tol': 0.,
            'orders': None,
            'good_tol': 0.,
            'nz_matches': 0,
            'n_tested': 0,
            'nz_entries': len(nz[0]),
            'J_shape': sparsity.shape,
        }

        self._update_subjac_sparsity(self.subjac_sparsity_iter(sparsity=sparsity))

        return sparsity, info

    def _uncompress_jac(self, J, direction):
        """
        Uncompress the Jacobian using the coloring information.

        Parameters
        ----------
        self : Component
            The component to uncompress the Jacobian for.
        J : ndarray
            The Jacobian to uncompress.
        direction : str
            The direction to uncompress the Jacobian in.

        Returns
        -------
        ndarray
            The uncompressed Jacobian.
        """
        if self._coloring_info.coloring is not None:
            return self._coloring_info.coloring.expand_jac(J, direction)
        return J

    def _jax_derivs2partials(self, deriv_vals, partials, ofnames, wrtnames):
        """
        Copy JAX derivatives into partials.

        Parameters
        ----------
        self : Component
            The component to copy the derivatives into.
        deriv_vals : tuple
            The derivatives.
        partials : dict
            The partials to copy the derivatives into, keyed by (of_name, wrt_name).
        ofnames : list
            The output names.
        wrtnames : list
            The input names.
        """
        nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
            isinstance(deriv_vals[0], tuple)
        nof = len(ofnames)

        wrtnames = list(wrtnames)
        for ofidx, ofname in enumerate(ofnames):
            ofmeta = self._var_rel2meta[ofname]
            for wrtidx, wrtname in enumerate(wrtnames):
                key = (ofname, wrtname)
                if key not in partials:
                    # FIXME: this means that we computed a derivative that we didn't need
                    continue

                dvals = deriv_vals
                # if there's only one 'of' value, we only take the indexed value if the
                # return value of compute_primal is single entry tuple. If a single array or
                # scalar is returned, we don't apply the 'of' index.
                if nof > 1 or nested_tup:
                    dvals = dvals[ofidx]

                dvals = dvals[wrtidx].reshape(ofmeta['size'], self._var_rel2meta[wrtname]['size'])

                sjmeta = partials.get_metadata(key)
                rows = sjmeta['rows']
                if rows is None:
                    partials[ofname, wrtname] = dvals
                else:
                    partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]

    def _update_add_input_kwargs(self, **kwargs):
        if self.options['default_to_dyn_shapes']:
            if kwargs.get('val') is None and kwargs.get('shape') is None:
                if kwargs.get('copy_shape') is None and kwargs.get('compute_shape') is None:
                    if kwargs.get('shape_by_conn') is None:
                        kwargs['shape_by_conn'] = True

        return kwargs

    def _update_add_output_kwargs(self, name, **kwargs):
        if self.options['default_to_dyn_shapes']:
            if kwargs.get('val') is None and kwargs.get('shape') is None:
                if kwargs.get('copy_shape') is None and kwargs.get('compute_shape') is None:
                    # add our own compute_shape function
                    kwargs['compute_shape'] = self._get_compute_shape_func(name)

        return kwargs

    def compute_sparsity(self, direction=None, num_iters=1, perturb_size=1e-9):
        """
        Get the sparsity of the Jacobian.

        Parameters
        ----------
        direction : str
            The direction to compute the sparsity for.
        num_iters : int
            The number of times to run the perturbation iteration.
        perturb_size : float
            The size of the perturbation to use.

        Returns
        -------
        coo_matrix
            The sparsity of the Jacobian.
        """
        if self._sparsity is None:
            if self.options['derivs_method'] == 'jax':
                self._sparsity = self._compute_sparsity(direction, num_iters, perturb_size)
            else:
                self._sparsity = super().compute_sparsity(direction=direction,
                                                          num_iters=num_iters,
                                                          perturb_size=perturb_size)
        return self._sparsity

    def _update_subjac_sparsity(self, sparsity_iter):
        if self.options['derivs_method'] == 'jax':
            _update_subjac_sparsity(sparsity_iter, self.pathname, self._subjacs_info)
        else:
            super()._update_subjac_sparsity(sparsity_iter)

    def _get_compute_shape_func(self, name):
        return partial(self._compute_output_shape, name)

    def _compute_output_shape(self, name, input_shapes):
        if self._output_shapes is None:
            out_shapes = _compute_output_shapes(self._orig_compute_primal.__func__,
                                                input_shapes)
            self._output_shapes = {n: shp for n, shp in zip(self._var_rel_names['output'],
                                                            out_shapes)}
        return self._output_shapes[name]


class JaxExplicitMixin(JaxMixin):
    """
    Mixin class for ExplicitComponents that use JAX for derivatives.

    Parameters
    ----------
    matrix_free : bool
        If True, this component will compute derivatives using matrix vector products.
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    def _setup_partials(self):
        """
        Call setup_partials in components.
        """
        if self.options['derivs_method'] == 'jax':
            if self.matrix_free:
                if self._coloring_info.use_coloring():
                    issue_warning(f"{self.msginfo}: coloring has been set but matrix_free is True, "
                                  "so coloring will be ignored.")
                self._coloring_info.deactivate()
                self.compute_jacvec_product = self._compute_jacvec_product
            else:
                # if user hasn't declared partials, try to infer them from the compute_primal. If
                # that fails, declare all partials.
                if not self._declared_partials_patterns:
                    self._do_sparsity = True
                    try:
                        deps = list(get_function_deps(self._orig_compute_primal,
                                                      self._var_rel_names['output']))
                    except Exception:
                        self.declare_partials('*', '*')
                    else:
                        contvars = set(self._var_rel_names['input'])
                        contvars.update(self._var_rel_names['output'])
                        for of, wrt in deps:
                            if of in contvars and wrt in contvars:
                                self.declare_partials(of, wrt)

                self.compute_partials = self._compute_partials
                self._has_compute_partials = True

        super()._setup_partials()

    # we define _compute_partials here and possibly later rename it to compute_partials instead of
    # making this the base class version as we did with compute, because the existence of a
    # compute_partials method that is not the base class method is used to determine if a given
    # component computes its own partials.
    def _compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        self : ImplicitComponent
            The component instance.
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name]..
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        discrete_inputs = discrete_inputs.values() if discrete_inputs else ()
        self._update_jac_functs(discrete_inputs)

        if self._jac_colored_ is not None:
            return self._jac_colored_(inputs, partials)

        derivs = self._jac_func_(*inputs.values())

        # check to see if we even need this with jax.  A jax component doesn't need to map string
        # keys to partials.  We could just use the jacobian as an array to compute the derivatives.
        # Maybe make a simple JaxJacobian that is just a thin wrapper around the jacobian array.
        # The only issue is do higher level jacobians need the subjacobian info?
        self._jax_derivs2partials(derivs, partials, self._var_rel_names['output'],
                                  self._var_rel_names['input'])

    def _jacfwd_colored(self, inputs, partials):
        """
        Compute the forward jacobian using vmap with jvp and coloring.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        partials : dict
            The partials to compute.
        """
        J = self._jac_func_(self._tangents['fwd'], tuple(inputs.values()))
        partials.set_dense_jac(self, self._uncompress_jac(_jax2np(J), 'fwd'))

    def _jacrev_colored(self, inputs, partials):
        """
        Compute the reverse jacobian using vmap with vjp and coloring.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        partials : dict
            The partials to compute.
        """
        J = self._jac_func_(self._tangents['rev'], tuple(inputs.values()))
        partials.set_dense_jac(self, self._uncompress_jac(_jax2np(J).T, 'rev'))

    def _compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        r"""
        Compute jac-vector product (explicit). The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Parameters
        ----------
        self : ExplicitComponent
            The component instance.
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        d_inputs : Vector
            See inputs; product must be computed only if var_name in d_inputs.
        d_outputs : Vector
            See outputs; product must be computed only if var_name in d_outputs.
        mode : str
            Either 'fwd' or 'rev'.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        if mode == 'fwd':
            dx = tuple(d_inputs.values())
            full_invals = tuple(self._get_compute_primal_invals(inputs, discrete_inputs))
            x = full_invals[:len(dx)]
            other = full_invals[len(dx):]
            _, deriv_vals = jax.jvp(lambda *args: self.compute_primal(*args, *other),
                                    primals=x, tangents=dx)
            d_outputs.set_vals(deriv_vals)
        else:
            inhash = ((inputs.get_hash(),) + tuple(self._discrete_inputs.values()) +
                      self.get_self_statics())
            if inhash != self._static_hash:
                self._static_hash = inhash

                ncont_ins = d_inputs.nvars()
                full_invals = tuple(self._get_compute_primal_invals(inputs, discrete_inputs))
                x = full_invals[:ncont_ins]
                other = full_invals[ncont_ins:]
                # recompute vjp function if inputs have changed
                _, self._vjp_fun = jax.vjp(lambda *args: self.compute_primal(*args, *other), *x)

            deriv_vals = self._vjp_fun(tuple(d_outputs.values()) +
                                       tuple(self._discrete_outputs.values()))

            d_inputs.set_vals(deriv_vals)


class JaxImplicitMixin(JaxMixin):
    """
    Mixin class for ImplicitComponents that use JAX for derivatives.

    Parameters
    ----------
    matrix_free : bool
        If True, this component will compute derivatives using matrix vector products.
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    def _setup_partials(self):
        """
        Call setup_partials in components.
        """
        if self.options['derivs_method'] == 'jax':
            if self.matrix_free:
                if self._coloring_info.use_coloring():
                    issue_warning(f"{self.msginfo}: coloring has been set but matrix_free is True, "
                                  "so coloring will be ignored.")
                self._coloring_info.deactivate()
                self.apply_linear = self._jax_apply_linear
            else:
                # if user hasn't declared partials, try to infer them from the compute_primal. If
                # that fails, declare all partials.
                if not self._declared_partials_patterns:
                    self._do_sparsity = True
                    try:
                        deps = list(get_function_deps(self._orig_compute_primal,
                                                      self._var_rel_names['output']))
                    except Exception as err:
                        issue_warning(f"{self.msginfo}: Couldn't determine function graph for "
                                      f"compute_primal: {err}")
                        self.declare_partials('*', '*')
                    else:
                        contvars = set(self._var_rel_names['input'])
                        contvars.update(self._var_rel_names['output'])
                        for of, wrt in deps:
                            if of in contvars and wrt in contvars:
                                self.declare_partials(of, wrt)

                self.linearize = self._jax_linearize
                self._has_linearize = True

        super()._setup_partials()

    def _jax_linearize(self, inputs, outputs, partials, discrete_inputs=None,
                       discrete_outputs=None):
        """
        Compute sub-jacobian parts for an implicit component.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        partials : partial Jacobian
            Sub-jac components written to jacobian[output_name, input_name].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        discrete_inputs = discrete_inputs.values() if discrete_inputs else ()
        self._update_jac_functs(discrete_inputs)

        if self._jac_colored_ is not None:
            return self._jac_colored_(inputs, outputs, partials)

        derivs = self._jac_func_(*chain(inputs.values(), outputs.values()))
        self._jax_derivs2partials(derivs, partials, self._var_rel_names['output'],
                                  chain(self._var_rel_names['input'],
                                        self._var_rel_names['output']))

    def _jacfwd_colored(self, inputs, outputs, partials):
        """
        Compute the forward jacobian using vmap with jvp and coloring.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        outputs : dict
            The outputs to the component.
        partials : dict
            The partials to compute.
        """
        J = self._jac_func_(self._tangents['fwd'], tuple(chain(inputs.values(), outputs.values())))
        partials.set_dense_jac(self, self._uncompress_jac(_jax2np(J), 'fwd'))

    def _jacrev_colored(self, inputs, outputs, partials):
        """
        Compute the reverse jacobian using vmap with vjp and coloring.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        outputs : dict
            The outputs to the component.
        partials : dict
            The partials to compute.
        """
        J = self._jac_func_(self._tangents['rev'], tuple(chain(inputs.values(), outputs.values())))
        partials.set_dense_jac(self, self._uncompress_jac(_jax2np(J).T, 'rev'))

    def _jax_apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        r"""
        Compute jac-vector product (implicit). The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': (d_inputs, d_outputs) \|-> d_residuals

            'rev': d_residuals \|-> (d_inputs, d_outputs)

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        d_inputs : Vector
            See inputs; product must be computed only if var_name in d_inputs.
        d_outputs : Vector
            See outputs; product must be computed only if var_name in d_outputs.
        d_residuals : Vector
            See outputs.
        mode : str
            Either 'fwd' or 'rev'.
        """
        if mode == 'fwd':
            dx = tuple(chain(d_inputs.values(), d_outputs.values()))
            full_invals = tuple(self._get_compute_primal_invals(inputs, outputs,
                                                                self._discrete_inputs))
            x = full_invals[:len(dx)]
            other = full_invals[len(dx):]
            _, deriv_vals = jax.jvp(lambda *args: self.compute_primal(*args, *other),
                                    primals=x, tangents=dx)
            if isinstance(deriv_vals, tuple):
                d_residuals.set_vals(deriv_vals)
            else:
                d_residuals.asarray()[:] = deriv_vals.flatten()
        else:
            inhash = (inputs.get_hash(), outputs.get_hash()) + tuple(self._discrete_inputs.values())
            if inhash != self._vjp_hash:
                # recompute vjp function only if inputs or outputs have changed
                dx = tuple(chain(d_inputs.values(), d_outputs.values()))
                full_invals = tuple(self._get_compute_primal_invals(inputs, outputs,
                                                                    self._discrete_inputs))
                x = full_invals[:len(dx)]
                other = full_invals[len(dx):]
                _, self._vjp_fun = jax.vjp(lambda *args: self.compute_primal(*args, *other), *x)
                self._vjp_hash = inhash

                if self._compute_primals_out_shape is None:
                    shape = jax.eval_shape(lambda *args: self.compute_primal(*args, *other), *x)
                    if isinstance(shape, tuple):
                        shape = (tuple(s.shape for s in shape), True,
                                 len(self._var_rel_names['input']))
                    else:
                        shape = (shape.shape, False, len(self._var_rel_names['input']))
                    self._compute_primals_out_shape = shape

            shape, istup, ninputs = self._compute_primals_out_shape

            if istup:
                deriv_vals = (self._vjp_fun(tuple(d_residuals.values())))
            else:
                deriv_vals = self._vjp_fun(tuple(d_residuals.values())[0])

            d_inputs.set_vals(deriv_vals[:ninputs])
            d_outputs.set_vals(deriv_vals[ninputs:])


class JaxExplicitGroupMixin(JaxMixin):
    """
    Mixin class for ExplicitGroups that use JAX for derivatives.
    """

    def _setup_compute_primal(self):
        """
        Set up the compute_primal method.
        """
        pass  # do nothing now because this is called before the Group's compute_primal is set

    def _get_compute_primal_inputs(self):
        """
        Return a dict of 'inputs' that will be used to compute the primal.

        This includes any inputs connected to a source outside the group boundary.

        Returns
        -------
        dict
            A dict of names of inputs passed to compute_primal mapped to shape.
        """
        if self.pathname == '':
            raise RuntimeError(f"{self.msginfo}: JAX mode not currently supported for the top level"
                               " group.")
        else:
            boundary_ins = self.get_boundary_inputs(local=True)
            ins = {n: m['shape'] for n, m in self._var_abs2meta['input'].items()
                   if n in boundary_ins}

        return ins

    def _get_compute_primal_outputs(self):
        # return all outputs that are not indep vars
        return {n: m for n, m in self._var_abs2meta['output'].items()
                if 'openmdao:indep_var' not in m['tags']}

    def _setup_check(self):
        """
        Do any error checking on user's setup, before any other recursion happens.
        """
        pass

    def _setup_jax(self):
        """
        If jax is active, collect all compute_primal methods from subcomponents.
        Combine them into a single compute_primal method for the group.
        """
        if jax is None:
            return

        if self.options['derivs_method'] != 'jax':
            # recurse down the tree and setup
            # jax anywhere below where it's active, then return.
            for subgroup in self._subgroups_myproc:
                subgroup._setup_jax()
            return

        if self._contains_parallel_group or self._mpi_proc_allocator.parallel:
            raise RuntimeError(f"{self.msginfo}: JAX mode not currently supported for parallel "
                               "groups or groups that contain them.")

        if self._discrete_inputs or self._discrete_outputs:
            raise RuntimeError(f"{self.msginfo}: JAX mode not currently supported for groups that "
                               "contain discrete inputs or outputs.")

        self._compute_primal_ins = self._get_compute_primal_inputs()
        self._compute_primal_in_slices = None
        self._compute_primal_outs = self._get_compute_primal_outputs()

        pathlen = len(self.pathname) + 1 if self.pathname else 0

        # local var names within our compute_primal function
        goutput_map = {n: f'o{i}' for i, n in enumerate(self._var_abs2meta['output'])}
        ginput_map = {}
        for i, name in enumerate(self._compute_primal_ins):
            ginput_map[name] = f'v{i}'

        instrs = ", ".join(ginput_map.values())
        src = [''.join(["def compute_primal(self, ", instrs, "):"])]

        from openmdao.core.component import Component

        # this will call compute_primal on all Components directly or indirectly under this group
        for system in self.system_iter(recurse=True, include_self=False, typ=Component):
            if system.compute_primal is None:
                raise RuntimeError(f"{self.msginfo}: Can't generate a compute_primal method for "
                                   f"this Group because component {system.pathname} has no "
                                   "compute_primal method.")

            ins = [f"self.{system.pathname[pathlen:]}"]
            for n in system._var_abs2meta['input']:
                if n in self._conn_global_abs_in2out:
                    ins.append(goutput_map[self._conn_global_abs_in2out[n]])
                else:
                    ins.append(ginput_map[n])

            ins = ', '.join(ins)
            outs = ', '.join(goutput_map[n] for n, m in system._var_abs2meta['output'].items()
                             if 'openmdao:indep_var' not in m['tags'])
            if len(system._var_abs2meta['output']) == 1:
                outs += ','
            src.append(f"    {outs} = self.{system.pathname[pathlen:]}.compute_primal({ins})")

        src.append('    return ' + ', '.join([goutput_map[n] for n in self._compute_primal_outs]))
        if len(goutput_map) == 1:
            src[-1] += ','  # make the output a tuple

        src = '\n'.join(src)

        print(f"{self.msginfo} compute_primal:\n" + src)

        # create the function
        namespace = {}
        exec(compile(src, '<string>', 'exec'), namespace)  # nosec trusted input
        compute_primal = namespace['compute_primal']

        self._orig_compute_primal = MethodType(compute_primal, self)

        if self.options['use_jit']:
            compute_primal = jax.jit(compute_primal, static_argnums=[0])

        self.compute_primal = MethodType(compute_primal, self)
