"""
Dump a simplified, generic description of an OpenMDAO model's structure as JSON.

The output is an om4 model spec built from generic groups and generic components, so a model's
structure can be inspected with om4 tooling without importing any of the model's own component
classes or their dependencies.

Recorded: all inputs and outputs with their shapes and units, all promotions (including
src_indices), all manual connections (including src_indices), all ``Group.set_input_defaults``
data, the class of each non-default solver, and all design variables, constraints and
objectives.  Variable values, solver options and optimizer scaling are not recorded.
"""

import json
import sys

import numpy as np

from openmdao.core.constants import _SetupStatus
from openmdao.core.group import Group
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.problem import Problem
from openmdao.core.system import System
import openmdao.utils.hooks as hooks
from openmdao.utils.file_utils import _load_and_exec
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning


# The om4 'type' keys for the generic classes this tool targets.
GENERIC_GROUP_TYPE = 'om4.utils.generic_systems:GenericGroup'
GENERIC_EXPLICIT_TYPE = 'om4.utils.generic_systems:GenericExplicitComponent'
GENERIC_IMPLICIT_TYPE = 'om4.utils.generic_systems:GenericImplicitComponent'

_DEFAULT_OUTFILE = 'generic_model.json'

# The name of the component OpenMDAO generates to source unconnected inputs.  om4 has no
# equivalent, so it is excluded from the dump.
_AUTO_IVC = '_auto_ivc'

# Name given to the single subsystem of the group generated to wrap a dumped Component.
_WRAPPED_COMP_NAME = 'comp'

# OpenMDAO solver class name -> the om4 'type' key for the equivalent solver.  Only the class
# is carried over; solver options are not, since this dump is structural.  A solver whose class
# is not here, or a subclass of one of these, is matched through its MRO.
_SOLVER_TYPES = {
    # nonlinear
    'NonlinearRunOnce': 'om4.solvers.nonlinear.nonlinear_runonce:NonlinearRunOnce',
    'NonlinearBlockGS': 'om4.solvers.nonlinear.nonlinear_block_gs:NonlinearBlockGS',
    'NewtonSolver': 'om4.solvers.nonlinear.newton.newton:NewtonSolver',
    # linear
    'LinearRunOnce': 'om4.solvers.linear.linear_runonce:LinearRunOnce',
    'LinearBlockGS': 'om4.solvers.linear.linear_block_gs:LinearBlockGS',
    'DirectSolver': 'om4.solvers.linear.direct:DirectSolver',
    'PETScDirectSolver': 'om4.solvers.linear.petsc_direct:PETScDirectSolver',
    'PETScKrylov': 'om4.solvers.linear.petsc_krylov:PETScKrylov',
    # OpenMDAO's ScipyKrylov only offers gmres, so it maps to om4's gmres class
    'ScipyKrylov': 'om4.solvers.linear.scipy_krylov:ScipyGMRES',
}

# The solvers a Group gets when nothing was assigned.  Omitting these keeps the spec minimal
# and lets om4 apply its own defaults.
_DEFAULT_SOLVERS = frozenset(('NonlinearRunOnce', 'LinearRunOnce'))


class _RemovedIvcs(object):
    """
    Bookkeeping for the IndepVarComps that are dropped from a dump.

    om4 has no IndepVarComp: an unconnected input is its own source, and a design variable has
    to sit on one (``om4/variables/design_var.py`` raises for a design variable on an output).
    So every IndepVarComp is removed, the connections from its outputs are dropped so that the
    inputs they fed become unconnected, and any design variable or response on one of its
    outputs is moved onto those inputs.

    Parameters
    ----------
    model : Group
        The root of the model being dumped.
    keep : str or None
        Pathname of a system to leave alone, used when the dump was asked for that system
        specifically and removing it would leave nothing to look at.

    Attributes
    ----------
    pathnames : set of str
        Pathnames of the IndepVarComps being removed.
    src2tgts : dict
        Absolute IndepVarComp output name mapped to the absolute inputs it feeds.
    """

    def __init__(self, model, keep=None, warnings=None):
        self.pathnames = set()
        self.src2tgts = {}
        self.promotions = {}
        self.shared = {}
        self._model = model

        for comp in model.system_iter(include_self=True, recurse=True, typ=IndepVarComp):
            # _auto_ivc is already excluded from the tree, and OpenMDAO already reports an
            # auto_ivc sourced design variable against the input it feeds, so it needs none of
            # this handling.
            if comp.name == _AUTO_IVC or comp.pathname == keep:
                continue
            self.pathnames.add(comp.pathname)

        for abs_in, abs_out in model._conn_global_abs_in2out.items():
            if self.owns(abs_out):
                self.src2tgts.setdefault(abs_out, []).append(abs_in)

        if warnings is not None:
            self._plan_promotions(warnings)

    def _plan_promotions(self, warnings):
        """
        Work out the promotions that keep each optimizer variable a single variable.

        An IndepVarComp output often fans out to several separate inputs, and in OpenMDAO that
        is one design variable driving all of them.  Splitting it into one design variable per
        input would change the optimization problem, so instead the fed inputs are promoted to
        a single shared name and the design variable stays on that name.  om4 supports one
        promoted input fanning out to many leaves, so this is the faithful translation.

        Parameters
        ----------
        warnings : list
            List that messages about lossy conversions are appended to.
        """
        for system in self._model.system_iter(include_self=True, recurse=True):
            for metadata in (system._design_vars, system._responses):
                for meta in metadata.values():
                    abs_out = meta.get('source', '')
                    if self.owns(abs_out) and abs_out not in self.shared:
                        self._unify(system, abs_out, meta['name'], warnings)

    def _unify(self, owner, abs_out, declared, warnings):
        """
        Plan promotions giving every input fed by one IndepVarComp output a shared name.

        Parameters
        ----------
        owner : System
            The system that declared the optimizer variable.
        abs_out : str
            Absolute name of the IndepVarComp output.
        declared : str
            The output's promoted name in the owner's scope, which the optimizer variable uses.
        warnings : list
            List that messages about lossy conversions are appended to.
        """
        model = self._model
        # An unpromoted output's name is dotted and cannot name a promoted input, so fall back
        # to its local name.
        desired = declared.rsplit('.', 1)[-1]

        # the shared name has to be free once the IndepVarComps are gone
        for other in owner._resolver.absnames(desired, 'output', report_error=False) or ():
            if not self.owns(other):
                warnings.append(f"Inputs fed by '{abs_out}' could not be promoted to a shared "
                                f"name, because '{desired}' is already the promoted name of "
                                f"output '{other}'.")
                return

        prefix = owner.pathname + '.' if owner.pathname else ''
        planned = []

        for abs_in in self.src2tgts[abs_out]:
            if not abs_in.startswith(prefix) or not owner._resolver.contains(abs_in, 'input'):
                warnings.append(f"Inputs fed by '{abs_out}' could not be promoted to a shared "
                                f"name, because '{abs_in}' is outside '{owner.pathname}'.")
                return

            if owner._resolver.abs2prom(abs_in, 'input') == desired:
                continue  # already shares the name, so nothing to synthesize

            child = owner._get_subsystem(abs_in[len(prefix):].split('.')[0])
            from_name = child._resolver.abs2prom(abs_in, 'input')

            # src_indices on the dropped connection have to move onto the promotion
            inds_list = model._var_abs2meta['input'].get(abs_in, {}).get('src_inds_list') or []
            if len(inds_list) > 1:
                warnings.append(f"Inputs fed by '{abs_out}' could not be promoted to a shared "
                                f"name, because '{abs_in}' has more than one level of "
                                "src_indices.")
                return

            src_shape = model._var_allprocs_abs2meta['output'][abs_out]['shape']
            planned.append((child.pathname, from_name, desired,
                            inds_list[0] if inds_list else None, src_shape))

        for pathname, from_name, name, indexer, src_shape in planned:
            self.promotions.setdefault(pathname, []).append((from_name, name, indexer,
                                                             src_shape))

        self.shared[abs_out] = desired

    def owns(self, abs_out):
        """
        Return True if an absolute output name belongs to an IndepVarComp being removed.

        Parameters
        ----------
        abs_out : str
            Absolute name of an output.

        Returns
        -------
        bool
            True if the output's component is being removed.
        """
        return '.' in abs_out and abs_out.rsplit('.', 1)[0] in self.pathnames

    def targets(self, system, abs_out):
        """
        Return the promoted names, in a system's scope, of the inputs an output feeds.

        Parameters
        ----------
        system : System
            The system whose scope the names are wanted in.
        abs_out : str
            Absolute name of the IndepVarComp output.

        Returns
        -------
        list of str
            Promoted input names, deduplicated and in connection order.
        """
        resolver = system._resolver
        names = []
        for abs_in in self.src2tgts.get(abs_out, ()):
            if not resolver.contains(abs_in, 'input'):
                continue
            prom = resolver.abs2prom(abs_in, 'input')
            if prom not in names:
                names.append(prom)

        return names


class _UnsupportedIndexError(Exception):
    """
    Raised internally when an index form has no om4 representation.
    """

    pass


def _json_val(val):
    """
    Convert a value to something the json module can serialize.

    Parameters
    ----------
    val : object
        The value to convert.

    Returns
    -------
    object
        A JSON serializable version of val.
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    return val


def _shape_json(shape):
    """
    Convert a shape to a JSON list.

    Parameters
    ----------
    shape : tuple or int or None
        The shape to convert.

    Returns
    -------
    list or None
        List version of shape, or None if shape is None.
    """
    if shape is None:
        return None
    if isinstance(shape, int):
        return [shape]
    return [int(s) for s in shape]


def _serialize_index(idx):
    """
    Convert a python index object to JSON using om4's IndexSpec rules.

    An int stays an int, a slice becomes a 'start:stop:step' string, an array becomes a list of
    ints, and a tuple becomes {'tuple': [...]} because each of its entries indexes a different
    dimension.

    Parameters
    ----------
    idx : int or slice or ndarray or tuple or list
        The index object to convert.

    Returns
    -------
    int or str or list or dict
        A JSON serializable version of idx.
    """
    if isinstance(idx, (bool, np.bool_)):
        raise _UnsupportedIndexError('boolean index')
    if isinstance(idx, (int, np.integer)):
        return int(idx)
    if isinstance(idx, slice):
        return f'{idx.start}:{idx.stop}:{idx.step}'
    if isinstance(idx, np.ndarray):
        return idx.tolist()
    if isinstance(idx, tuple):
        return {'tuple': [_serialize_index(i) for i in idx]}
    if isinstance(idx, list):
        return [_serialize_index(i) for i in idx]
    raise _UnsupportedIndexError(f'{idx!r}')


def _src_indices_json(idxr, where, warnings):
    """
    Convert an OpenMDAO Indexer to a JSON index spec and a flat_src flag.

    The structure preserving form is used when om4 can represent it.  Otherwise the indices are
    flattened into a resolved list of ints, which is semantically equivalent but loses the
    original slice or multi-dimensional form.

    Parameters
    ----------
    idxr : Indexer
        The indexer to convert.
    where : str
        Description of the declaration being converted, used in warning messages.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    tuple
        The JSON index spec and the flat_src flag.
    """
    try:
        return _serialize_index(idxr()), bool(idxr._flat_src)
    except _UnsupportedIndexError:
        warnings.append(f"src_indices of {where} have no om4 equivalent and were flattened "
                        "into a list of indices.")
        return idxr.to_json(), True


def _dyn_spec(meta, iotype, kind):
    """
    Build an om4 DynSpec for a dynamically shaped or dynamically united variable.

    om4 allows 'by_conn' only on an input and 'copy' only on an output, and 'compute' requires a
    method that a generic component does not have, so some OpenMDAO declarations cannot be
    represented.  In those cases the caller falls back to the resolved concrete value.

    Parameters
    ----------
    meta : dict
        Metadata for the variable.
    iotype : str
        Either 'input' or 'output'.
    kind : str
        Either 'shape' or 'units'.

    Returns
    -------
    tuple
        The DynSpec dict, or None, and a description of why no DynSpec could be built.
    """
    by_conn, copy, compute = f'{kind}_by_conn', f'copy_{kind}', f'compute_{kind}'

    if meta.get(by_conn):
        if iotype == 'input':
            return {'by_conn': True}, None
        return None, f"'{by_conn}' on an output"

    if meta.get(copy):
        if iotype == 'output':
            return {'copy': meta[copy]}, None
        return None, f"'{copy}' on an input"

    if meta.get(compute):
        return None, f"'{compute}'"

    return None, None


def _var_dict(meta, iotype, where, warnings):
    """
    Build the om4 spec for a single continuous variable.

    Parameters
    ----------
    meta : dict
        Metadata for the variable.
    iotype : str
        Either 'input' or 'output'.
    where : str
        Absolute name of the variable, used in warning messages.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    dict
        The om4 InputVar or OutputVar spec.
    """
    vdict = {}

    for kind in ('shape', 'units'):
        dyn, reason = _dyn_spec(meta, iotype, kind)
        if dyn is not None:
            vdict[kind] = dyn
            continue
        if reason is not None:
            warnings.append(f"'{where}' declares {reason}, which has no om4 equivalent, so its "
                            f"resolved {kind} was used instead.")
        # 'units' is always emitted, even when None.  om4 defaults it to UNSET, meaning 'not yet
        # known', while OpenMDAO's None means dimensionless.
        vdict[kind] = _shape_json(meta['shape']) if kind == 'shape' else meta['units']

    if meta.get('distributed'):
        vdict['distributed'] = True

    return vdict


def _component_vars(comp, warnings):
    """
    Build the variable specs for a Component.

    Parameters
    ----------
    comp : Component
        The component whose variables are being dumped.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    dict
        Mapping of om4 field name to a dict of variable specs, omitting empty ones.
    """
    out = {}
    resolver = comp._resolver

    for iotype, field in (('input', 'inputs'), ('output', 'outputs')):
        vdicts = {}
        for abs_name, meta in comp._var_allprocs_abs2meta[iotype].items():
            rel_name = resolver.abs2rel(abs_name, iotype)
            vdicts[rel_name] = _var_dict(meta, iotype, abs_name, warnings)
        if vdicts:
            out[field] = vdicts

    for iotype, field in (('input', 'discrete_inputs'), ('output', 'discrete_outputs')):
        # om4's DiscreteVar carries only val, desc and tags, none of which are structural, so a
        # discrete variable's name and existence are all that is recorded.
        vdicts = {resolver.abs2rel(abs_name, iotype): {}
                  for abs_name in comp._var_allprocs_discrete[iotype]}
        if vdicts:
            out[field] = vdicts

    return out


def _pattern_json(spec):
    """
    Convert a single OpenMDAO promotion spec to an om4 promotion pattern.

    Parameters
    ----------
    spec : str or tuple
        A name, a glob pattern, or an (old_name, new_name) rename tuple.

    Returns
    -------
    str or list
        The om4 pattern.
    """
    if isinstance(spec, tuple):
        return list(spec)
    return spec


def _promotes_entries(entries, where, warnings):
    """
    Convert (spec, prominfo) pairs into a list of om4 PromotesSpec dicts.

    Duplicate patterns are collapsed, preferring the entry that carries src_indices, because
    ``Group.promotes`` extends ``_var_promotes`` rather than replacing entries made by
    ``add_subsystem``.  Patterns are then grouped by their src_indices, since one om4
    PromotesSpec carries one index spec for a list of patterns.

    Parameters
    ----------
    entries : list
        List of (spec, _PromotesInfo or None) pairs.
    where : str
        Pathname of the system being promoted, used in warning messages.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    list
        List of om4 PromotesSpec dicts.
    """
    chosen = {}
    for spec, prominfo in entries:
        key = tuple(spec) if isinstance(spec, tuple) else spec
        if key not in chosen or prominfo is not None:
            chosen[key] = (spec, prominfo)

    groups = {}
    order = []
    for spec, prominfo in chosen.values():
        if prominfo is None or prominfo.src_indices is None:
            sig, extra = None, {}
        else:
            inds, flat = _src_indices_json(prominfo.src_indices,
                                           f"promotion of '{where}'", warnings)
            extra = {'src_indices': inds, 'flat_src': flat}
            if prominfo.src_shape is not None:
                extra['src_shape'] = _shape_json(prominfo.src_shape)
            sig = json.dumps(extra, sort_keys=True)

        if sig not in groups:
            groups[sig] = (extra, [])
            order.append(sig)
        groups[sig][1].append(_pattern_json(spec))

    specs = []
    for sig in order:
        extra, patterns = groups[sig]
        specs.append({'patterns': patterns, **extra})

    return specs


def _synthesized_promotes(subsys, ivcs, warnings):
    """
    Build the promotion specs added to give inputs fed by one IndepVarComp a shared name.

    Parameters
    ----------
    subsys : System
        The subsystem whose promotions are being dumped.
    ivcs : _RemovedIvcs
        The IndepVarComps being removed.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    list
        List of om4 PromotesSpec dicts.
    """
    groups = {}
    order = []

    for from_name, name, indexer, src_shape in ivcs.promotions.get(subsys.pathname, ()):
        pattern = name if from_name == name else [from_name, name]

        if indexer is None:
            sig, extra = None, {}
        else:
            inds, flat = _src_indices_json(indexer,
                                           f"promotion of '{from_name}' on "
                                           f"'{subsys.pathname}'", warnings)
            extra = {'src_indices': inds, 'flat_src': flat}
            if src_shape is not None:
                extra['src_shape'] = _shape_json(src_shape)
            sig = json.dumps(extra, sort_keys=True)

        if sig not in groups:
            groups[sig] = (extra, [])
            order.append(sig)
        groups[sig][1].append(pattern)

    return [{'patterns': patterns, **extra}
            for extra, patterns in (groups[sig] for sig in order)]


def _promotes_specs(subsys, warnings, ivcs):
    """
    Build the promotes_inputs and promotes_outputs specs for a subsystem.

    OpenMDAO's bare ``promotes`` has no om4 equivalent, so each of its patterns is routed to
    promotes_inputs, promotes_outputs, or both, depending on which it actually matches.  It
    cannot simply be copied to both, because om4 raises an error on a pattern that matches
    nothing.  The pattern strings themselves are kept as written.

    Parameters
    ----------
    subsys : System
        The subsystem whose promotions are being dumped.
    warnings : list
        List that messages about lossy conversions are appended to.
    ivcs : _RemovedIvcs
        The IndepVarComps being removed, which may add promotions of their own.

    Returns
    -------
    tuple
        The promotes_inputs and promotes_outputs spec lists.
    """
    var_promotes = subsys._var_promotes
    entries = {'input': list(var_promotes['input']), 'output': list(var_promotes['output'])}

    if var_promotes['any']:
        maps = subsys._get_promotion_maps()
        matched = {io: {tup[1] for tup in maps[io].values()} for io in ('input', 'output')}

        for spec, prominfo in var_promotes['any']:
            key = spec[0] if isinstance(spec, tuple) else spec
            hits = [io for io in ('input', 'output') if key in matched[io]]
            if hits:
                for io in hits:
                    entries[io].append((spec, prominfo))
            else:
                warnings.append(f"Promotion pattern '{key}' on '{subsys.pathname}' matched no "
                                "variables and was omitted.")

    proms_in = (_promotes_entries(entries['input'], subsys.pathname, warnings) +
                _synthesized_promotes(subsys, ivcs, warnings))

    return proms_in, _promotes_entries(entries['output'], subsys.pathname, warnings)


def _solvers(system, warnings):
    """
    Build the solver specs for a system.

    Only the solver class is translated.  Solver options are not carried over, so the dumped
    model says which kind of solver a group needs without claiming anything about how it was
    tuned.  A solver OpenMDAO has and om4 does not is skipped with a warning rather than
    silently replaced by a different algorithm.

    Parameters
    ----------
    system : System
        The system whose solvers are being dumped.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    dict
        Mapping of om4 field name to an om4 solver spec, omitting unset and default solvers.
    """
    specs = {}

    for field in ('nonlinear_solver', 'linear_solver'):
        solver = getattr(system, field, None)
        if solver is None or type(solver).__name__ in _DEFAULT_SOLVERS:
            continue

        # a user subclass of a known solver is recorded as its nearest mapped ancestor
        for klass in type(solver).__mro__:
            if klass.__name__ in _SOLVER_TYPES:
                specs[field] = {'type': _SOLVER_TYPES[klass.__name__]}
                break
        else:
            warnings.append(f"The {field} on '{system.pathname}' is a "
                            f"{type(solver).__name__}, which has no om4 equivalent, so it was "
                            "omitted.")

    return specs


def _opt_var(meta, key, where, warnings):
    """
    Build the om4 spec for one design variable, constraint or objective.

    Bounds, indices and units are carried over because they are what the declaration means.
    Scaling is not: OpenMDAO's scaler/adder/ref/ref0 have no field on om4's OptimizerVar, where
    scaling is a Transform, and it is optimizer tuning rather than model structure.

    Parameters
    ----------
    meta : dict
        OpenMDAO metadata for the design variable or response.
    key : str
        The name the variable is keyed by, which is its alias if it has one.
    where : str
        Description of the declaration, used in warning messages.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    dict
        The om4 DesignVar, Constraint or Objective spec.
    """
    spec = {}

    name = meta['name']
    if name != key:
        # the key is an alias, so om4 needs the promoted name spelled out
        spec['path'] = name

    for bound in ('lower', 'upper', 'equals'):
        if meta.get(bound) is not None:
            spec[bound] = _json_val(meta[bound])

    if meta.get('units') is not None:
        spec['units'] = meta['units']

    flat = bool(meta.get('flat_indices'))
    if meta.get('indices') is not None:
        inds, forced_flat = _src_indices_json(meta['indices'], where, warnings)
        spec['indices'] = inds
        flat = flat or forced_flat
    if flat:
        spec['flat_src'] = True

    for flag in ('linear', 'cache_linear_solution'):
        if meta.get(flag):
            spec[flag] = True

    if meta.get('parallel_deriv_color') is not None:
        spec['parallel_deriv_color'] = meta['parallel_deriv_color']

    scaling = [k for k in ('ref', 'ref0', 'scaler', 'adder') if meta.get(k) is not None]
    if scaling:
        warnings.append(f"The {where} declares {', '.join(sorted(scaling))}, which om4 expresses "
                        "as a Transform rather than a field, so the scaling was omitted.")

    return spec


def _retargeted_opt_vars(system, key, meta, where, ivcs, warnings):
    """
    Move an optimizer variable off a removed IndepVarComp output onto the inputs it feeds.

    om4 only allows a design variable on an unconnected input, and the IndepVarComp that used
    to source these inputs is not in the dump, so the variable is redeclared against each
    distinct promoted input name it fed.

    Parameters
    ----------
    system : System
        The system that declared the optimizer variable.
    key : str
        The name the variable is keyed by.
    meta : dict
        OpenMDAO metadata for the design variable or response.
    where : str
        Description of the declaration, used in warning messages.
    ivcs : _RemovedIvcs
        The IndepVarComps being removed.
    warnings : list
        List that messages about lossy conversions are appended to.

    Returns
    -------
    list
        List of (key, metadata) pairs to emit in place of the original.
    """
    abs_out = meta['source']

    shared = ivcs.shared.get(abs_out)
    if shared is not None:
        # the inputs it fed were promoted to one shared name, so this stays a single variable
        new_meta = dict(meta)
        new_meta['name'] = shared
        return [(key if meta['name'] != key else shared, new_meta)]

    names = ivcs.targets(system, abs_out)

    if not names:
        warnings.append(f"The {where} is on the IndepVarComp output '{abs_out}', which feeds no "
                        "input visible here, so it could not be moved onto an input and was "
                        "omitted.")
        return []

    is_alias = meta['name'] != key
    if len(names) > 1:
        warnings.append(f"The {where} is on the IndepVarComp output '{abs_out}', which feeds "
                        f"{len(names)} separately promoted inputs, so it became that many "
                        f"optimizer variables: {', '.join(repr(n) for n in names)}.")
        if is_alias:
            warnings.append(f"The alias '{key}' could not be kept, since it cannot name more "
                            "than one variable.")
            is_alias = False

    retargeted = []
    for name in names:
        new_meta = dict(meta)
        new_meta['name'] = name
        # a single aliased variable keeps its alias as the key, so om4 still needs the
        # promoted name, which _opt_var emits as 'path'
        retargeted.append((key if is_alias else name, new_meta))

    return retargeted


def _opt_vars(system, warnings, ivcs):
    """
    Build the design variable, constraint and objective specs declared on a system.

    Parameters
    ----------
    system : System
        The system whose optimizer variables are being dumped.
    warnings : list
        List that messages about lossy conversions are appended to.
    ivcs : _RemovedIvcs
        The IndepVarComps being removed.

    Returns
    -------
    dict
        Mapping of om4 field name to a dict of specs, omitting empty ones.
    """
    specs = {}
    groups = {'design_vars': {}, 'constraints': {}, 'objectives': {}}

    sources = [('design variable', system._design_vars, 'design_vars'),
               ('response', system._responses, None)]

    for label, metadata, fixed_field in sources:
        for key, meta in metadata.items():
            where = f"{label} '{key}' on '{system.pathname}'"

            if meta.get('discrete'):
                warnings.append(f"The {where} is discrete, which om4 does not support for "
                                "optimizer variables, so it was omitted.")
                continue

            field = fixed_field
            if field is None:
                field = 'objectives' if meta.get('type') == 'obj' else 'constraints'

            if ivcs.owns(meta.get('source', '')):
                entries = _retargeted_opt_vars(system, key, meta, where, ivcs, warnings)
            else:
                entries = [(key, meta)]

            for new_key, new_meta in entries:
                groups[field][new_key] = _opt_var(new_meta, new_key, where, warnings)

    for field, entries in groups.items():
        if entries:
            specs[field] = entries

    return specs


def _connections(group, warnings, ivcs):
    """
    Build the connection specs for the manual connections owned by a group.

    Parameters
    ----------
    group : Group
        The group whose connections are being dumped.
    warnings : list
        List that messages about lossy conversions are appended to.
    ivcs : _RemovedIvcs
        The IndepVarComps being removed.

    Returns
    -------
    list
        List of om4 Connection dicts.
    """
    conns = []
    resolver = group._resolver
    auto_prefix = _AUTO_IVC + '.'
    for tgt, (src, src_indices) in group._manual_connections.items():
        if resolver.contains(src, 'output'):
            absnames = resolver.absnames(src, 'output', report_error=False)
            # a connection from an IndepVarComp that is not in the dump has no source to come
            # from; dropping it is what leaves the target as an unconnected input, which is how
            # om4 sources it
            if absnames and ivcs.owns(absnames[0]):
                continue

        if src.startswith(auto_prefix):
            # An input-to-input connect whose chain reaches no real output is rewritten by
            # OpenMDAO into a connection from the auto_ivc output that sources it
            # (Group.transform_input_input_connections).  om4 has no auto_ivc, so there is no
            # such connection to make; the target is simply an unconnected input.  The two
            # inputs no longer share a source, which in om4 would be expressed by promoting
            # them to the same name instead.
            warnings.append(f"'{tgt}' is connected to the auto_ivc output '{src}', which has "
                            "no om4 equivalent, so the connection was omitted and the input "
                            "is left unconnected.")
            continue

        conn = {'src': src, 'tgt': tgt}
        if src_indices is not None:
            inds, flat = _src_indices_json(src_indices,
                                           f"connection '{src}' to '{tgt}'", warnings)
            conn['src_indices'] = inds
            conn['flat_src'] = flat
        conns.append(conn)

    return conns


def _input_defaults(group):
    """
    Build the input_defaults specs for a group.

    Parameters
    ----------
    group : Group
        The group whose input defaults are being dumped.

    Returns
    -------
    dict
        Mapping of promoted input name to an om4 InputDefault dict.
    """
    defaults = {}
    for name, meta in group._group_inputs.items():
        # Unlike variable values, a set_input_defaults value is a declaration about a promoted
        # input rather than model state, so it is always carried over.
        entry = {}
        if meta.get('val') is not None:
            entry['val'] = _json_val(meta['val'])
        if meta.get('units') is not None:
            entry['units'] = meta['units']
        if meta.get('src_shape') is not None:
            entry['src_shape'] = _shape_json(meta['src_shape'])
        defaults[name] = entry

    return defaults


def _system_dict(system, types, warnings, ivcs, is_root=False):
    """
    Build the om4 spec for a System and, recursively, everything under it.

    Parameters
    ----------
    system : System
        The system being dumped.
    types : dict
        Mapping of 'group', 'explicit' and 'implicit' to om4 type strings.
    warnings : list
        List that messages about lossy conversions are appended to.
    ivcs : _RemovedIvcs
        The IndepVarComps being removed.
    is_root : bool
        If True, this system is the root of the dump and has no promotions to record.

    Returns
    -------
    dict
        The om4 System spec.
    """
    if isinstance(system, Group):
        sdict = {'type': types['group']}
    elif isinstance(system, ImplicitComponent):
        sdict = {'type': types['implicit']}
    else:
        sdict = {'type': types['explicit']}

    if not is_root:
        proms_in, proms_out = _promotes_specs(system, warnings, ivcs)
        if proms_in:
            sdict['promotes_inputs'] = proms_in
        if proms_out:
            sdict['promotes_outputs'] = proms_out

    # Design vars and responses are recorded on the system that declared them, which is where
    # om4 keeps them too, so names stay in that system's scope on both sides.
    sdict.update(_opt_vars(system, warnings, ivcs))

    if not isinstance(system, Group):
        sdict.update(_component_vars(system, warnings))
        if isinstance(system, ImplicitComponent):
            # only om4's implicit components can own solvers
            sdict.update(_solvers(system, warnings))
        return sdict

    children = [(s.name, _system_dict(s, types, warnings, ivcs))
                for s in system._subsystems_myproc
                if s.name != _AUTO_IVC and s.pathname not in ivcs.pathnames]

    if system.comm.size > 1:
        # Only one rank per subsystem reports it, then all ranks are merged so that subsystems
        # that are remote on this rank are still in the tree.
        if system._subsystems_myproc and system._subsystems_myproc[0].comm.rank != 0:
            children = []
        merged = {}
        for gathered in system.comm.allgather(children):
            merged.update(gathered)
        # allgather loses declaration order, so restore it from the subsystem indices.
        order = {name: info.index for name, info in system._subsystems_allprocs.items()}
        children = sorted(merged.items(), key=lambda item: order.get(item[0], len(order)))

    sdict['subsystems'] = dict(children)

    conns = _connections(system, warnings, ivcs)
    if conns:
        sdict['connections'] = conns

    defaults = _input_defaults(system)
    if defaults:
        sdict['input_defaults'] = defaults

    solvers = _solvers(system, warnings)
    sdict.update(solvers)

    if 'nonlinear_solver' in solvers:
        # In OpenMDAO a group's nonlinear solver covers everything inside the group.  om4
        # defaults to collapsing only cycles and hoisting the rest, which would leave a
        # non-cyclic implicit child uncovered by the solver we just recorded, so the whole
        # group is collapsed into one execution node to match OpenMDAO.
        sdict['collapse_by'] = 'group'

    if isinstance(system, ParallelGroup):
        sdict['parallel'] = True

    # om4 defaults auto_order to True and would re-sort the subsystems, so ordering is pinned.
    sdict['auto_order'] = False

    return sdict


def generic_model_dict(source, system=None, group_type=GENERIC_GROUP_TYPE,
                       explicit_type=GENERIC_EXPLICIT_TYPE,
                       implicit_type=GENERIC_IMPLICIT_TYPE):
    """
    Build a generic om4 Group spec dict describing the structure of a set up model.

    Parameters
    ----------
    source : Problem or System
        A Problem, or a System belonging to one, on which final_setup has been called.
    system : str or None
        Pathname, relative to source, of the system to dump.  If None, the whole model is
        dumped.  If it names a Component, the component is wrapped in a generated group.
    group_type : str
        om4 'type' string to use for groups.
    explicit_type : str
        om4 'type' string to use for explicit components.
    implicit_type : str
        om4 'type' string to use for implicit components.

    Returns
    -------
    dict
        A JSON serializable om4 Group spec.
    """
    if isinstance(source, Problem):
        model = source.model
    elif isinstance(source, System):
        model = source
    else:
        raise TypeError(f"Expected a Problem or a System, but got {type(source).__name__}.")

    prob_meta = model._problem_meta
    if prob_meta is None or prob_meta['setup_status'] < _SetupStatus.POST_SETUP:
        raise RuntimeError("Can't dump a generic model before setup has been called. "
                           "Call Problem.setup() first.")

    if prob_meta['setup_status'] < _SetupStatus.POST_SETUP2:
        # Dynamic shapes are resolved and the auto_ivc component is created in setup_part2, so
        # it has to have run.  Values are not needed, so final_setup does not.
        if isinstance(source, Problem):
            source.set_setup_status(_SetupStatus.POST_SETUP2)
        else:
            prob_meta['model_ref']()._setup_part2()

    if system:
        target = model._get_subsystem(system)
        if target is None:
            raise RuntimeError(f"Can't find system with pathname '{system}'.")
    else:
        target = model

    types = {'group': group_type, 'explicit': explicit_type, 'implicit': implicit_type}
    warnings = []

    # The model root owns the resolved connections, so the IndepVarComps are worked out from
    # there even when only a subtree is being dumped.  A component asked for by name is kept,
    # since removing it would leave nothing to look at.
    ivcs = _RemovedIvcs(prob_meta['model_ref'](),
                        keep=target.pathname if isinstance(target, IndepVarComp) else None,
                        warnings=warnings)

    tree = _system_dict(target, types, warnings, ivcs, is_root=True)

    if not isinstance(target, Group):
        # om4 only wraps a Group in a Problem when loading a spec, so a lone component is
        # wrapped here to keep it viewable.
        tree = {
            'type': group_type,
            'subsystems': {_WRAPPED_COMP_NAME: tree},
            'auto_order': False,
        }

    for msg in warnings:
        issue_warning(msg)

    return tree


def dump_generic_model(source, outfile=_DEFAULT_OUTFILE, indent=4, **kwargs):
    """
    Write a generic om4 Group spec describing the structure of a set up model to a JSON file.

    Parameters
    ----------
    source : Problem or System
        A Problem, or a System belonging to one, on which final_setup has been called.
    outfile : str
        Name of the JSON file to write.
    indent : int or None
        Indentation to use in the JSON file.
    **kwargs : dict
        Additional arguments to pass to generic_model_dict.

    Returns
    -------
    dict
        The om4 Group spec that was written.
    """
    tree = generic_model_dict(source, **kwargs)

    if MPI is None or MPI.COMM_WORLD.rank == 0:
        with open(outfile, 'w') as f:
            json.dump(tree, f, indent=indent)

    return tree


def _dump_generic_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao dump_generic' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=_DEFAULT_OUTFILE, action='store', dest='outfile',
                        help='Name of the JSON output file.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name.')
    parser.add_argument('--system', action='store', dest='system', default='',
                        help='Pathname of the system to dump. Defaults to the whole model.')


def _dump_generic_cmd(options, user_args):
    """
    Return the post final_setup hook function for 'openmdao dump_generic'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _is_target(prob):
        # a sub-problem sets up before the top level one, so without this check the dump would
        # happen on the wrong problem
        return options.problem is not None or '/' not in prob._get_inst_id()

    def _dump(prob):
        dump_generic_model(prob, outfile=options.outfile, system=options.system)
        print(f"Generic model written to '{options.outfile}'.")

    def _dump_after_setup(prob):
        if _is_target(prob):
            _dump(prob)
            sys.exit(0)

    def _dump_with_errors(prob):
        # A model that failed setup is exactly the case where seeing its structure is most
        # useful, so dump it rather than letting the errors stop everything.
        if not _is_target(prob) or not prob._metadata['saved_errors']:
            return
        # setup_part2 has finished by the time this hook runs, the status just hasn't been
        # advanced yet.  Advance it so the dump isn't blocked; _check_collected_errors raises
        # immediately after this, so nothing else runs against the modified status.
        if prob._metadata['setup_status'] < _SetupStatus.POST_SETUP2:
            prob._metadata['setup_status'] = _SetupStatus.POST_SETUP2
        _dump(prob)

    hooks._register_hook('_check_collected_errors', class_name='Problem',
                         inst_id=options.problem, pre=_dump_with_errors)
    hooks._register_hook('setup', class_name='Problem', inst_id=options.problem,
                         post=_dump_after_setup)

    _load_and_exec(options.file[0], user_args)
