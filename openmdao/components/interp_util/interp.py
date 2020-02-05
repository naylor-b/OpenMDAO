"""
Base class for interpolation methods that calculate values for each dimension independently.

Based on Tables in NPSS, and was added to bridge the gap between some of the slower scipy
implementations.
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np

from openmdao.components.interp_util.interp_akima import InterpAkima
from openmdao.components.interp_util.interp_bsplines import InterpBSplines
from openmdao.components.interp_util.interp_cubic import InterpCubic
from openmdao.components.interp_util.interp_lagrange2 import InterpLagrange2
from openmdao.components.interp_util.interp_lagrange3 import InterpLagrange3
from openmdao.components.interp_util.interp_scipy import InterpScipy
from openmdao.components.interp_util.interp_slinear import InterpLinear

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError

INTERP_METHODS = {
    'slinear': InterpLinear,
    'lagrange2': InterpLagrange2,
    'lagrange3': InterpLagrange3,
    'cubic': InterpCubic,
    'akima': InterpAkima,
    'scipy_cubic': InterpScipy,
    'scipy_slinear': InterpScipy,
    'scipy_quintic': InterpScipy,
    'bsplines': InterpBSplines,
}

TABLE_METHODS = ['slinear', 'lagrange2', 'lagrange3', 'cubic', 'akima', 'scipy_cubic',
                 'scipy_slinear', 'scipy_quintic']
SPLINE_METHODS = ['slinear', 'lagrange2', 'lagrange3', 'cubic', 'akima', 'bsplines']


class InterpND(object):
    """
    Interpolation on a regular grid of arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be uneven. Several
    interpolation methods are supported. These are defined in the child classes. Gradients are
    provided for all interpolation methods. Gradients with respect to grid values are also
    available optionally.

    Attributes
    ----------
    extrapolate : bool
        If False, when interpolated values are requested outside of the domain of the input data,
        a ValueError is raised. If True, then the methods are allowed to extrapolate.
        Default is True (raise an exception).
    grid : tuple
        Collection of points that determine the regular grid.
    table : <InterpTable>
        Table object that contains algorithm that performs the interpolation.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    x_interp : ndarray
        Cached non-decreasing vector of points to be interpolated when used as an order-reducing
        spline.
    _compute_d_dvalues : bool
        When set to True, compute gradients with respect to the grid values.
    _compute_d_dx : bool
        When set to True, compute gradients with respect to the interpolated point location.
    _d_dx : ndarray
        Cache of computed gradients with respect to evaluation point.
    _d_dgrid : ndarray
        Cache of computed gradients with respect to grid.
    _d_dvalues : ndarray
        Cache of computed gradients with respect to table values.
    _interp : class
        Class specified as interpolation algorithm, used to regenerate if needed.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _interp_options : dict
        Dictionary of cached interpolator-specific options.
    _xi : ndarray
        Cache of current evaluation point.
    """

    def __init__(self, method="slinear", points=None, values=None, x_interp=None, extrapolate=False,
                 **kwargs):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        method : str or list of str, optional
            Name of interpolation method(s).
        x_interp : ndarry or None
            If we are always interpolating at a fixed set of locations, then they can be
            specified here.
        extrapolate : bool
            If False, when interpolated values are requested outside of the domain of the input data,
            a ValueError is raised. If True, then the methods are allowed to extrapolate.
            Default is True (raise an exception).
        **kwargs : dict
            Interpolator-specific options to pass onward.
        """
        if not isinstance(method, str):
            msg = "Argument 'method' should be a string."
            raise ValueError(msg)
        elif method not in INTERP_METHODS:
            all_m = ', '.join(['"' + m + '"' for m in INTERP_METHODS])
            raise ValueError('Interpolation method "%s" is not defined. Valid methods are '
                             '%s.' % (method, all_m))
        self.extrapolate = extrapolate

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        if np.iscomplexobj(values[:]):
            msg = "Interpolation method '%s' does not support complex values." % method
            raise ValueError(msg)

        for i, p in enumerate(points):
            n_p = len(p)
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == n_p:
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))

        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self.x_interp = x_interp

        self._xi = None
        self._d_dx = None
        self._d_dgrid = None
        self._d_dvalues = None
        self._compute_d_dvalues = False

        # Cache spline coefficients.
        interp = INTERP_METHODS[method]

        if method.startswith('scipy'):
            kwargs['interp_method'] = method

        table = interp(self.grid, values, interp, **kwargs)
        table.check_config()
        self.table = table
        self._interp = interp
        self._interp_options = kwargs

    def interpolate(self, x, compute_derivatives=True):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        x : ndarray of shape (..., ndim)
            Location to provide interpolation.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        ndarray
            Value of derivative of interpolated output with respect to input x.
        ndarray
            Value of derivative of interpolated output with respect to values.
        """
        table = self.table

        xnew = self._interpolate(x)

        if compute_derivatives:
            return xnew,
        else:
            return xnew

    def _interpolate(self, xi):
        """
        Interpolate at the sample coordinates.

        This method is called from OpenMDAO

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        """
        # cache latest evaluation point for gradient method's use later
        self._xi = xi

        if not self.extrapolate:
            for i, p in enumerate(xi.T):
                if np.isnan(p).any():
                    raise OutOfBoundsError("One of the requested xi contains a NaN",
                                           i, np.NaN, self.grid[i][0], self.grid[i][-1])

                eps = 1e-14 * self.grid[i][-1]
                if not np.logical_and(np.all(self.grid[i][0] <= p + eps),
                                      np.all(p - eps <= self.grid[i][-1])):
                    p1 = np.where(self.grid[i][0] > p)[0]
                    p2 = np.where(p > self.grid[i][-1])[0]
                    # First violating entry is enough to direct the user.
                    violated_idx = set(p1).union(p2).pop()
                    value = p[violated_idx]
                    raise OutOfBoundsError("One of the requested xi is out of bounds",
                                           i, value, self.grid[i][0], self.grid[i][-1])

        if self._compute_d_dvalues:
            # If the table grid or values are component inputs, then we need to create a new table
            # each iteration.
            interp = self._interp
            self.table = interp(self.grid, self.values, interp, **self._interp_options)
            self.table._compute_d_dvalues = True

        table = self.table
        if table._vectorized:
            result, derivs_x, derivs_val, derivs_grid = table.evaluate_vectorized(xi)

        else:
            xi = np.atleast_2d(xi)
            n_nodes, nx = xi.shape
            result = np.empty((n_nodes, ), dtype=xi.dtype)
            derivs_x = np.empty((n_nodes, nx), dtype=xi.dtype)
            derivs_val = None

            # TODO: it might be possible to vectorize over n_nodes.
            for j in range(n_nodes):
                val, d_x, d_values, d_grid = table.evaluate(xi[j, :])
                result[j] = val
                derivs_x[j, :] = d_x.flatten()
                if d_values is not None:
                    if derivs_val is None:
                        dv_shape = [n_nodes]
                        dv_shape.extend(self.values.shape)
                        derivs_val = np.zeros(dv_shape, dtype=xi.dtype)
                    in_slice = table._full_slice
                    full_slice = [slice(j, j + 1)]
                    full_slice.extend(in_slice)
                    shape = derivs_val[tuple(full_slice)].shape
                    derivs_val[tuple(full_slice)] = d_values.reshape(shape)

        # Cache derivatives
        self._d_dx = derivs_x
        self._d_dvalues = derivs_val

        return result

    def _evaluate_spline(self, values):
        """
        Interpolate at all fixed output coordinates given the new table values.

        This method is called from OpenMDAO.

        Parameters
        ----------
        values : ndarray(n_nodes x n_points)
            The data on the regular grid in n dimensions.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        """
        xi = self.x_interp
        self.values = values

        # cache latest evaluation point for gradient method's use later
        self._xi = xi.copy()

        table = self.table
        if table._vectorized:

            if table._name == 'bsplines':
                table.values = values
            else:
                interp = self._interp
                table = interp(self.grid, values, interp, **self._interp_options)
                table._compute_d_dvalues = True
                table._compute_d_dx = False

            result, _, derivs_val, _ = table.evaluate_vectorized(xi)

        else:
            interp = self._interp
            n_nodes, _ = values.shape
            nx = np.prod(xi.shape)
            result = np.empty((n_nodes, nx), dtype=values.dtype)
            derivs_x = np.empty((n_nodes, nx), dtype=values.dtype)
            derivs_val = None

            # TODO: it might be possible to vectorize over n_nodes.
            for j in range(n_nodes):

                table = interp(self.grid, values[j, :], interp, **self._interp_options)
                table._compute_d_dvalues = True
                table._compute_d_dx = False

                for k in range(nx):
                    x_pt = np.atleast_2d(xi[k])
                    val, _, d_values, _ = table.evaluate(x_pt)
                    result[j, k] = val
                    if d_values is not None:
                        if derivs_val is None:
                            dv_shape = [n_nodes, nx]
                            dv_shape.extend(values.shape[1:])
                            derivs_val = np.zeros(dv_shape, dtype=values.dtype)
                        in_slice = table._full_slice
                        full_slice = [slice(j, j + 1), slice(k, k + 1)]
                        full_slice.extend(in_slice)
                        shape = derivs_val[tuple(full_slice)].shape
                        derivs_val[tuple(full_slice)] = d_values.reshape(shape)

        # Cache derivatives
        self._d_dvalues = derivs_val

        self.table = table
        return result

    def gradient(self, xi):
        """
        Compute the gradients at the specified point.

        Most of the gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

        If the point for evaluation differs from the point used to produce
        the currently cached gradient, the interpolation is re-performed in
        order to return the correct gradient.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            Vector of gradients of the interpolated values with respect to each value in xi.
        """
        if (self._xi is None) or (not np.array_equal(xi, self._xi)):
            # If inputs have changed since last computation, then re-interpolate.
            self.interpolate(xi)

        return self._d_dx.reshape(np.asarray(xi).shape)

    def training_gradients(self, pt):
        """
        Compute the training gradient for the vector of training points.

        Parameters
        ----------
        pt : ndarray
            Training point values.

        Returns
        -------
        ndarray
            Gradient of output with respect to training point values.
        """
        if self.table._vectorized:
            return self.table.training_gradients(pt)

        else:
            grid = self.grid
            interp = self._interp

            for i, axis in enumerate(self.grid):
                ngrid = axis.size
                values = np.zeros(ngrid)
                deriv_i = np.zeros(ngrid)

                for j in range(ngrid):
                    values[j] = 1.0
                    table = interp([grid[i]], values, self._interp, **self._interp_options)
                    table._compute_d_dvalues = False
                    deriv_i[j], _, _, _ = table.evaluate(pt[i:i + 1])
                    values[j] = 0.0

                if i == 0:
                    deriv_running = deriv_i.copy()
                else:
                    deriv_running = np.outer(deriv_running, deriv_i)

            return deriv_running
