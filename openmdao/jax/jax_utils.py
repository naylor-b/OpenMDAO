"""
Utilities for the use of jax in combination with OpenMDAO.
"""
import ast
import textwrap
import inspect

import numpy as np

from openmdao.visualization.tables.table_builder import generate_table
from openmdao.utils.code_utils import get_function_deps
from openmdao.utils.om_warnings import issue_warning


def jit_stub(f, *args, **kwargs):
    """
    Provide a dummy jit decorator for use if jax is not available.

    Parameters
    ----------
    f : Callable
        The function or method to be wrapped.
    *args : list
        Positional arguments.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    Callable
        The decorated function.
    """
    return f


try:
    import jax
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
    import jax.numpy as jnp
    from jax import jit, tree_util

except ImportError:

    jax = None
    jnp = np
    jit = jit_stub


def register_jax_component(comp_class):
    """
    Provide a class decorator that registers the given class as a pytree_node.

    This allows jax to use jit compilation on the methods of this class if they
    reference attributes of the class itself, such as `self.options`.

    Note that this decorator is not necessary if the given class does not reference
    `self` in any methods to which `jax.jit` is applied.

    Parameters
    ----------
    comp_class : class
        The decorated class.

    Returns
    -------
    object
        The same class given as an argument.

    Raises
    ------
    NotImplementedError
        If this class does not define the `_tree_flatten` and _tree_unflatten` methods.
    RuntimeError
        If jax is not available.
    """
    if jax is None:
        raise RuntimeError("jax is not available. "
                           "Try 'pip install openmdao[jax]' with Python>=3.8.")

    if not hasattr(comp_class, '_tree_flatten'):
        raise NotImplementedError(f'class {comp_class} does not implement method _tree_flatten.'
                                  f'\nCannot register {comp_class} as a jax jit-compatible '
                                  f'component.')

    if not hasattr(comp_class, '_tree_unflatten'):
        raise NotImplementedError(f'class {comp_class} does not implement method _tree_unflatten.'
                                  f'\nCannot register class {comp_class} as a jax jit-compatible '
                                  f'component.')

    _jax_register_pytree_class(comp_class)
    return comp_class


def dump_jaxpr(closed_jaxpr):
    """
    Print out the contents of a Jaxpr.

    Parameters
    ----------
    closed_jaxpr : jax.core.ClosedJaxpr
        The Jaxpr to be examined.
    """
    jaxpr = closed_jaxpr.jaxpr
    print("invars:", jaxpr.invars)
    print("in_avals", closed_jaxpr.in_avals, closed_jaxpr.in_avals[0].dtype)
    print("outvars:", jaxpr.outvars)
    print("out_avals:", closed_jaxpr.out_avals)
    print("constvars:", jaxpr.constvars)
    for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print()
    print("jaxpr:", jaxpr)


class ReturnChecker(ast.NodeVisitor):
    """
    An ast.NodeVisitor that determines if a method returns a tuple or not.

    Parameters
    ----------
    method : method
        The method to be analyzed.

    Attributes
    ----------
    _returns : list
        The list of boolean values indicating whether or not the method returns a tuple. One
        entry for each return statement in the method.
    _fstack : list
        The stack of function definitions being visited.
    """

    def __init__(self, method):  # noqa
        self._returns = []
        self._fstack = []
        self.visit(ast.parse(textwrap.dedent(inspect.getsource(method)), mode='exec'))

    def returns_tuple(self):
        """
        Return whether or not the method returns a tuple.

        Returns
        -------
        bool
            True if the method returns a tuple, False otherwise.
        """
        if self._returns:
            ret = type(self._returns[0])
            for r in self._returns[1:]:
                if type(r) is not ret:
                    raise RuntimeError(f"ReturnChecker can't handle a method with multiple return "
                                       f"statements that return different types. This method "
                                       f"returns a tuple of {ret.__name__} and {type(r).__name__}.")
            return issubclass(ret, ast.Tuple)
        return False

    def visit_Return(self, node):
        """
        Visit a Return node.

        Parameters
        ----------
        node : ASTnode
            The return node being visited.
        """
        self._returns.append(node.value)

    def visit_FunctionDef(self, node):
        """
        Visit a FunctionDef node.

        Parameters
        ----------
        node : ASTnode
            The function definition node being visited.
        """
        if self._fstack:
            return  # skip nested functions
        self._fstack.append(node)
        for stmt in node.body:
            self.visit(stmt)
        self._fstack.pop()


def benchmark_component(comp_class, methods=(None, 'cs', 'jax'), initial_vals=None, repeats=2,
                        mode='auto', table_format='simple_grid', **kwargs):
    """
    Benchmark the performance of a Component using different methods for computing derivatives.

    Parameters
    ----------
    comp_class : class
        The class of the Component to be benchmarked.
    methods : tuple of str
        The methods to be benchmarked. Options are 'cs', 'jax', and None.
    initial_vals : dict or None
        Initial values for the input variables.
    repeats : int
        The number of times to run compute/compute_partials.
    mode : str
        The preferred derivative direction for the Problem.
    table_format : str or None
        If not None, the format of the table to be displayed.
    **kwargs : dict
        Additional keyword arguments to be passed to the Component.

    Returns
    -------
    dict
        A dictionary containing the benchmark results.
    """
    import time

    from openmdao.core.problem import Problem
    from openmdao.devtools.memory import mem_usage

    verbose = table_format is not None
    results = []
    for method in methods:
        mem_start = mem_usage()
        p = Problem()
        comp = p.model.add_subsystem('comp', comp_class(**kwargs))
        comp.options['derivs_method'] = method
        if method in ('cs', 'fd'):
            comp._has_approx = True
            comp._get_approx_scheme(method)

        if initial_vals:
            for name, val in initial_vals.items():
                p.model.set_val('comp.' + name, val)

        p.setup(mode=mode, force_alloc_complex='cs' in methods)
        p.run_model()

        model_mem = mem_usage

        if verbose:
            print(f"\nModel memory usage: {model_mem} MB")
            print(f"\nTiming {repeats} compute calls for {comp_class.__name__} using "
                  f"{method} method.")
        start = time.perf_counter()
        for n in range(repeats):
            comp.compute(comp._inputs, comp._outputs)
            if verbose:
                print('.', end='', flush=True)
        results.append([method, 'compute', n, time.perf_counter() - start, None])

        diff_mem = mem_usage() - mem_start
        results[-1][-1] = diff_mem

        if verbose:
            print(f"\n\nTiming {repeats} compute_partials calls for {comp_class.__name__} using "
                  f"{method} method.")
        start = time.perf_counter()
        for n in range(repeats):
            p.model._linearize(None)
            if verbose:
                print('.', end='', flush=True)
        results.append([method, 'compute_partials', n, time.perf_counter() - start, None])

        diff_mem = mem_usage() - model_mem
        results[-1][-1] = diff_mem

        del p

    if verbose:
        print('\n')
        headers = ['Method', 'Function', 'Iterations', 'Time (s)', 'Memory (MB)']
        generate_table(results, tablefmt=table_format, headers=headers).display()

    return results


if jax is None:
    def _jax_register_pytree_class(cls):
        pass

else:

    _registered_classes = set()

    def _jax_register_pytree_class(cls):
        """
        Register a class with jax so that it can be used with jax.jit.

        This can be called after instantiating the class if necessary.

        Parameters
        ----------
        cls : class
            The class to be registered.
        """
        global _registered_classes
        if cls not in _registered_classes:
            # register with jax so we can flatten/unflatten self
            tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
            _registered_classes.add(cls)


def get_vmap_tangents(vals, direction, fill=1., coloring=None):
    """
    Return a tuple of tangents values for use with vmap.

    The batching dimension is the last axis of each tangent.

    Parameters
    ----------
    vals : list
        List of function input or output values.
    direction : str
        The direction to compute the sparsity in.  It must be 'fwd' or 'rev'.
    fill : float
        The value to fill nonzero entries in the tangent with.
    coloring : Coloring or None
        A Coloring object that contains coloring information including nonzero indices.

    Returns
    -------
    tuple of ndarray or ndarray
        The tangents values to be passed to vmap.
    """
    sizes = [np.size(a) for a in vals]
    totsize = np.sum(sizes)

    if coloring is None:
        # start with a full diagonal matrix, which allows us to set a seed for each input value
        # in parallel.
        arr = np.empty(totsize)
        arr[:] = fill
        tangent = np.diag(arr)
        ncols = totsize
    else:
        # using coloring, so 'compress' the diagonal matrix to one with ncolors columns.
        # columns are the batching dimension for vmap and each column also corresponds to a color.
        colors = list(coloring.color_iter(direction))
        tangent = np.zeros((totsize, len(colors)))
        for i, nzs in enumerate(colors):
            tangent[nzs, i] = 1.
        ncols = len(colors)
    # take the 2D tangent array and reshape it to match the shape of each input variable.
    # (with the additional batching dimension as the last axis)
    tangents = []
    start = end = 0
    for v in vals:
        end += np.size(v)
        tangents.append(jnp.array(tangent[start:end].reshape(np.shape(v) + (ncols,))))
        start = end

    tangents = tuple(tangents)

    return tangents


def _update_subjac_sparsity(sparsity_iter, pathname, subjacs_info):
    """
    Update subjac sparsity info based on the given sparsity iterator.

    Parameters
    ----------
    sparsity_iter : iter of tuple
        Tuple of the form (of, wrt, rows, cols, shape).
    pathname : str
        The pathname of the component.
    subjacs_info : dict
        The subjac sparsity info.
    """
    prefix = pathname + '.'
    for of, wrt, rows, cols, shape in sparsity_iter:
        # sparsity uses relative names, so convert to absolute
        abs_key = (prefix + of, prefix + wrt)
        if abs_key not in subjacs_info:
            if rows is not None and len(rows) == 0:
                continue

            subjacs_info[abs_key] = {
                'shape': shape,
                'dependent': True,
                'rows': rows,
                'cols': cols,
            }

        if rows is None:
            subjacs_info[abs_key]['val'] = np.zeros(shape)
        else:
            if len(rows) == 0:
                del subjacs_info[abs_key]
            else:
                subjacs_info[abs_key].update({
                    'rows': rows,
                    'cols': cols,
                    'val': np.zeros(len(rows))
                })


def _compute_output_shapes(func, input_shapes):
    """
    Compute the shapes of the outputs of the function.

    The function must be traceable by jax.

    Parameters
    ----------
    func : function
        The function to compute the output shapes for.
    input_shapes : list
        The shapes of the input variables, or None if the input isn't a scalar or array.
    """
    argnames = list(inspect.signature(func).parameters)
    traceargs = []
    for argname in argnames:
        inshape = input_shapes.get(argname)
        if inshape is not None:
            traceargs.append(jax.ShapeDtypeStruct(inshape, jnp.float64))
        else:
            traceargs.append(None)

    retvals = jax.eval_shape(func, *traceargs)
    if not isinstance(retvals, tuple):
        retvals = (retvals,)

    retshapes = []
    for val in retvals:
        try:
            retshapes.append(val.shape)
        except AttributeError:
            retshapes.append(None)

    return retshapes


def _ensure_returns_tuple(func):
    """
    Ensure that the function returns a tuple.

    If the function already returns a tuple, it is returned unchanged.
    Otherwise, a wrapper function is returned that returns a tuple.

    If for some reason the function cannot be parsed, it is returned unchanged.

    Parameters
    ----------
    func : function
        The function to ensure returns a tuple.

    Returns
    -------
    function
        The function that returns a tuple.
    """
    try:
        checker = ReturnChecker(func)
    except Exception:
        issue_warning(f"Failed to parse function {func.__name__} to check if it returns a tuple."
                      "Returning original function.")
        return func
    else:
        if checker.returns_tuple():
            return func
        else:
            def wrapper(*args, **kwargs):
                return (func(*args, **kwargs),)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper


def _jax2np(J):
    """
    Take the return of vmapped jvp/vjp and convert to a numpy array.

    Parameters
    ----------
    J : tuple or jax array
        The return of vmapped jvp/vjp.

    Returns
    -------
    ndarray
        The numpy array.
    """
    if isinstance(J, tuple):
        if len(J) == 1:
            J = np.asarray(J[0])
            # reshape(-1, ...) to flatten all but the last dimension
            return J.reshape(-1, J.shape[-1])
        else:
            return np.concatenate([np.asarray(a).reshape(-1, a.shape[-1]) for a in J])
    else:
        return np.asarray(J).reshape(-1, J.shape[-1])


if __name__ == '__main__':
    import openmdao.api as om

    def func(x, y):  # noqa: D103
        z = jnp.sin(x) * y
        q = x * 1.5
        zz = q + x * 1.5
        return z, zz

    print('partials are:\n', list(get_function_deps(func, ('z', 'zz'))))

    p = om.Problem()
    comp = p.model.add_subsystem('comp', om.ExecComp('y = 2.0*x', x=np.ones(3), y=np.ones(3)))
    comp.derivs_method = 'jax'
    p.setup()
    p.run_model()

    print(p.compute_totals(of=['comp.y'], wrt=['comp.x']))
