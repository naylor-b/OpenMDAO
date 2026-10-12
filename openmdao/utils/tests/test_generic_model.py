"""
Tests for the generic model dump.
"""

import json
import os
import subprocess
import textwrap
import unittest
from fnmatch import fnmatchcase

import numpy as np

import openmdao.api as om
from openmdao.core.constants import _SetupStatus
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.generic_model import (GENERIC_EXPLICIT_TYPE, GENERIC_GROUP_TYPE,
                                          GENERIC_IMPLICIT_TYPE, _SOLVER_TYPES,
                                          dump_generic_model, generic_model_dict)
from openmdao.utils.indexer import indexer
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import use_tempdirs

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class Comp(om.ExplicitComponent):
    """
    A simple component with two inputs, one output and one discrete input.
    """

    def setup(self):
        self.add_input('x', shape=(3,), units='m')
        self.add_input('a', val=2.0)
        self.add_output('y', shape=(3,), units='m')
        self.add_discrete_input('di', val=3)


class ImpComp(om.ImplicitComponent):
    """
    A simple implicit component.
    """

    def setup(self):
        self.add_input('x', val=1.0)
        self.add_output('y', val=1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['y'] = outputs['y'] - inputs['x']


def _spec_is_group(node):
    """
    Return True if a spec node describes a group.

    Parameters
    ----------
    node : dict
        A node of a generic model spec.

    Returns
    -------
    bool
        True if the node is a group.
    """
    return 'subsystems' in node


def _match_promotions(prom_specs, child_proms):
    """
    Map a child's promoted names to the names they are promoted to in the parent.

    This reimplements promotion matching from the emitted spec alone, so that the spec can be
    checked against what OpenMDAO itself resolved.

    Parameters
    ----------
    prom_specs : list
        List of om4 PromotesSpec dicts.
    child_proms : iterable of str
        The child's promoted names.

    Returns
    -------
    dict
        Mapping of child promoted name to parent promoted name, for promoted names only.
    """
    names, globs, renames = set(), [], {}
    for spec in prom_specs:
        for pat in spec['patterns']:
            if isinstance(pat, list):
                renames[pat[0]] = pat[1]
            elif any(c in pat for c in '*?['):
                globs.append(pat)
            else:
                names.add(pat)

    matched = {}
    for prom in child_proms:
        if prom in renames:
            matched[prom] = renames[prom]
        elif prom in names:
            matched[prom] = prom
        else:
            for pat in globs:
                if pat == '*' or fnmatchcase(prom, pat):
                    matched[prom] = prom
                    break

    return matched


def _resolve_spec(node, path=''):
    """
    Resolve promoted names and connections from a generic model spec.

    Parameters
    ----------
    node : dict
        A node of a generic model spec.
    path : str
        Absolute pathname of the node.

    Returns
    -------
    tuple
        A dict of {iotype: {promoted name: [absolute names]}} and a dict of
        {absolute input: absolute output}.
    """
    if not _spec_is_group(node):
        # discrete variables are promoted and connected just like continuous ones
        fields = {'input': ('inputs', 'discrete_inputs'),
                  'output': ('outputs', 'discrete_outputs')}
        proms = {}
        for iotype, names in fields.items():
            proms[iotype] = {name: [f'{path}.{name}' if path else name]
                             for field in names for name in node.get(field, {})}
        return proms, {}

    proms = {'input': {}, 'output': {}}
    conns = {}

    for cname, child in node['subsystems'].items():
        cpath = f'{path}.{cname}' if path else cname
        cproms, cconns = _resolve_spec(child, cpath)
        conns.update(cconns)

        for iotype, field in (('input', 'promotes_inputs'), ('output', 'promotes_outputs')):
            matched = _match_promotions(child.get(field, []), cproms[iotype])
            for cprom, absnames in cproms[iotype].items():
                name = matched.get(cprom, f'{cname}.{cprom}')
                proms[iotype].setdefault(name, []).extend(absnames)

    # an input and an output promoted to the same name in a group are implicitly connected
    for name, absins in proms['input'].items():
        if name in proms['output']:
            for absin in absins:
                conns.setdefault(absin, proms['output'][name][0])

    for conn in node.get('connections', []):
        absouts = proms['output'].get(conn['src'])
        for absin in proms['input'].get(conn['tgt'], ()):
            conns[absin] = absouts[0] if absouts else None

    return proms, conns


def _om_truth(model):
    """
    Collect the promoted input names and connection sources that OpenMDAO resolved.

    Parameters
    ----------
    model : Group
        The model to collect from.

    Returns
    -------
    tuple
        A dict of {absolute input: promoted name} and a dict of {absolute input: absolute
        output}, with auto_ivc sources reported as None.
    """
    resolver = model._resolver
    prefix = model.pathname + '.' if model.pathname else ''

    abs2prom = {}
    conns = {}
    for absname in resolver.abs_iter('input'):
        abs2prom[absname[len(prefix):]] = resolver.abs2prom(absname, 'input')
        src = resolver.source(absname)
        conns[absname[len(prefix):]] = None if src.startswith('_auto_ivc.') \
            else src[len(prefix):]

    return abs2prom, conns


class TestGenericModelDict(unittest.TestCase):

    def _build(self):
        p = om.Problem(name='t')
        model = p.model
        g = model.add_subsystem('g', om.Group(), promotes=['*'])
        g.add_subsystem('c1', Comp(), promotes_inputs=['x'], promotes_outputs=[('y', 'yy')])
        g.add_subsystem('c2', Comp(), promotes_inputs=['x', 'a'])
        g.promotes('c1', inputs=['x'], src_indices=[0, 1, 2], src_shape=(3, 3),
                   flat_src_indices=True)
        g.promotes('c2', inputs=['x'], src_indices=[3, 4, 5], src_shape=(3, 3),
                   flat_src_indices=True)
        model.add_subsystem('d', Comp(), promotes_inputs=['a*'])
        model.connect('yy', 'd.x', src_indices=[0, 1, 2], flat_src_indices=True)
        model.set_input_defaults('x', src_shape=(3, 3), units='m')
        p.setup()
        p.final_setup()
        return p

    def test_types(self):
        p = om.Problem()
        p.model.add_subsystem('c', Comp())
        p.model.add_subsystem('i', ImpComp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['type'], GENERIC_GROUP_TYPE)
        self.assertEqual(spec['subsystems']['c']['type'], GENERIC_EXPLICIT_TYPE)
        self.assertEqual(spec['subsystems']['i']['type'], GENERIC_IMPLICIT_TYPE)

    def test_type_overrides(self):
        p = om.Problem()
        p.model.add_subsystem('c', Comp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p, group_type='G', explicit_type='E', implicit_type='I')
        self.assertEqual(spec['type'], 'G')
        self.assertEqual(spec['subsystems']['c']['type'], 'E')

    def test_vars_shape_and_units(self):
        spec = generic_model_dict(self._build())
        c1 = spec['subsystems']['g']['subsystems']['c1']

        self.assertEqual(c1['inputs']['x'], {'shape': [3], 'units': 'm'})
        self.assertEqual(c1['inputs']['a'], {'shape': [1], 'units': None})
        self.assertEqual(c1['outputs']['y'], {'shape': [3], 'units': 'm'})
        self.assertEqual(c1['discrete_inputs'], {'di': {}})
        self.assertNotIn('discrete_outputs', c1)

    def test_scalar_shape_is_empty_list(self):
        class Scalar(om.ExplicitComponent):
            def setup(self):
                self.add_output('y', shape=())

        p = om.Problem()
        p.model.add_subsystem('c', Scalar())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['subsystems']['c']['outputs']['y']['shape'], [])

    def test_no_values_emitted(self):
        p = self._build()
        p.set_val('x', np.arange(9, dtype=float).reshape((3, 3)))
        spec = generic_model_dict(p)

        def check(node):
            if _spec_is_group(node):
                for child in node['subsystems'].values():
                    check(child)
                return
            for field in ('inputs', 'outputs'):
                for name, var in node.get(field, {}).items():
                    for forbidden in ('val', 'lower', 'upper', 'global_shape', 'global_size',
                                      'ref', 'ref0', 'res_ref'):
                        self.assertNotIn(forbidden, var,
                                         f"'{forbidden}' should not be emitted for '{name}'")

        check(spec)

    def test_units_always_emitted(self):
        # om4 defaults units to UNSET, meaning 'not yet known', while OpenMDAO's None means
        # dimensionless, so None must be emitted explicitly rather than omitted.
        spec = generic_model_dict(self._build())
        var = spec['subsystems']['g']['subsystems']['c1']['inputs']['a']
        self.assertIn('units', var)
        self.assertIsNone(var['units'])

    def test_promotion_rename(self):
        spec = generic_model_dict(self._build())
        c1 = spec['subsystems']['g']['subsystems']['c1']
        self.assertEqual(c1['promotes_outputs'], [{'patterns': [['y', 'yy']]}])

    def test_promotion_glob_kept_verbatim(self):
        spec = generic_model_dict(self._build())
        self.assertEqual(spec['subsystems']['d']['promotes_inputs'],
                         [{'patterns': ['a*']}])

    def test_promotion_src_indices(self):
        spec = generic_model_dict(self._build())
        c2 = spec['subsystems']['g']['subsystems']['c2']

        self.assertEqual(c2['promotes_inputs'], [
            {'patterns': ['x'], 'src_indices': [3, 4, 5], 'flat_src': True,
             'src_shape': [3, 3]},
            {'patterns': ['a']},
        ])

    def test_duplicate_promotion_collapsed(self):
        # 'x' is promoted by add_subsystem and again by Group.promotes with src_indices, so
        # _var_promotes holds two entries for it.  Only the indexed one takes effect.
        p = self._build()
        c1 = p.model._get_subsystem('g.c1')
        self.assertEqual(len(c1._var_promotes['input']), 2)

        spec = generic_model_dict(p)
        proms = spec['subsystems']['g']['subsystems']['c1']['promotes_inputs']
        self.assertEqual(len(proms), 1)
        self.assertEqual(proms[0]['patterns'], ['x'])
        self.assertEqual(proms[0]['src_indices'], [0, 1, 2])

    def test_bare_promotes_split_by_io(self):
        class InComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('xin', val=1.0)
                self.add_input('both', val=1.0)
                self.add_output('yout', val=1.0)

        class OutComp(om.ExplicitComponent):
            def setup(self):
                self.add_output('both', val=1.0)
                self.add_output('zout', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('c', InComp(), promotes=['x*', 'y*', ('both', 'bb')])
        p.model.add_subsystem('d', OutComp(), promotes=['z*', ('both', 'bb')])
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        c, d = spec['subsystems']['c'], spec['subsystems']['d']

        # 'x*' matches only inputs, 'y*' only outputs, and the identical rename lands on
        # inputs for one component and outputs for the other.
        self.assertEqual(c['promotes_inputs'], [{'patterns': ['x*', ['both', 'bb']]}])
        self.assertEqual(c['promotes_outputs'], [{'patterns': ['y*']}])
        self.assertNotIn('promotes_inputs', d)
        self.assertEqual(d['promotes_outputs'], [{'patterns': ['z*', ['both', 'bb']]}])

    def test_bare_promotes_star_goes_to_both(self):
        spec = generic_model_dict(self._build())
        g = spec['subsystems']['g']
        self.assertEqual(g['promotes_inputs'], [{'patterns': ['*']}])
        self.assertEqual(g['promotes_outputs'], [{'patterns': ['*']}])

    def test_root_promotions_dropped(self):
        spec = generic_model_dict(self._build())
        self.assertNotIn('promotes_inputs', spec)
        self.assertNotIn('promotes_outputs', spec)

    def test_manual_connection(self):
        spec = generic_model_dict(self._build())
        self.assertEqual(spec['connections'], [
            {'src': 'yy', 'tgt': 'd.x', 'src_indices': [0, 1, 2], 'flat_src': True},
        ])

    def test_connection_without_src_indices(self):
        p = om.Problem()
        p.model.add_subsystem('c1', Comp())
        p.model.add_subsystem('c2', Comp())
        p.model.connect('c1.y', 'c2.x')
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['connections'], [{'src': 'c1.y', 'tgt': 'c2.x'}])

    def test_multi_target_connect(self):
        p = om.Problem()
        p.model.add_subsystem('c1', Comp())
        p.model.add_subsystem('c2', Comp())
        p.model.add_subsystem('c3', Comp())
        p.model.connect('c1.y', ['c2.x', 'c3.x'])
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['connections'], [
            {'src': 'c1.y', 'tgt': 'c2.x'},
            {'src': 'c1.y', 'tgt': 'c3.x'},
        ])

    def test_input_defaults(self):
        spec = generic_model_dict(self._build())
        self.assertEqual(spec['input_defaults'], {'x': {'units': 'm', 'src_shape': [3, 3]}})

    def test_input_defaults_val_is_kept(self):
        p = om.Problem()
        p.model.add_subsystem('c1', Comp(), promotes_inputs=['a'])
        p.model.add_subsystem('c2', Comp(), promotes_inputs=['a'])
        p.model.set_input_defaults('a', val=7.0, units='m')
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['input_defaults'], {'a': {'val': 7.0, 'units': 'm',
                                                       'src_shape': [1]}})

    def test_input_defaults_array_val(self):
        class ArrComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('v', shape=(2,))
                self.add_output('o', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('c1', ArrComp(), promotes_inputs=['v'])
        p.model.add_subsystem('c2', ArrComp(), promotes_inputs=['v'])
        p.model.set_input_defaults('v', val=np.array([3., 4.]))
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['input_defaults']['v']['val'], [3., 4.])

    def test_auto_ivc_excluded(self):
        spec = generic_model_dict(self._build())
        self.assertNotIn('_auto_ivc', spec['subsystems'])
        self.assertEqual(list(spec['subsystems']), ['g', 'd'])

    def test_no_auto_ivc_anywhere_in_spec(self):
        # nothing in the emitted spec may name auto_ivc, since om4 has no equivalent
        spec = generic_model_dict(self._build())
        self.assertNotIn('_auto_ivc', json.dumps(spec))

    def test_auto_ivc_sourced_connection_dropped(self):
        # An input-to-input connect with no real source upstream is rewritten by OpenMDAO into
        # a connection from the auto_ivc output that feeds it.  That connection cannot exist in
        # om4, so it is dropped and the input is left unconnected.
        class Simple(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('c1', Simple())
        p.model.add_subsystem('c2', Simple())
        p.model.connect('c1.x', 'c2.x')
        p.setup()
        p.final_setup()

        self.assertEqual(p.model._manual_connections, {'c1.x': ('_auto_ivc.v0', None)})

        msg = ("'c1.x' is connected to the auto_ivc output '_auto_ivc.v0', which has no om4 "
               "equivalent, so the connection was omitted and the input is left unconnected.")
        with assert_warning(UserWarning, msg, contains_msg=True):
            spec = generic_model_dict(p)

        self.assertNotIn('connections', spec)
        self.assertNotIn('_auto_ivc', json.dumps(spec))

    def test_no_auto_ivc_model(self):
        # a fully connected model has no _auto_ivc at all
        class Src(om.ExplicitComponent):
            def setup(self):
                self.add_output('y', val=1.0)

        class Snk(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('z', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('src', Src())
        p.model.add_subsystem('snk', Snk())
        p.model.connect('src.y', 'snk.x')
        p.setup()
        p.final_setup()

        # the _auto_ivc object is still created, but it is never added as a subsystem
        self.assertNotIn('_auto_ivc', p.model._subsystems_allprocs)

        spec = generic_model_dict(p)
        self.assertEqual(list(spec['subsystems']), ['src', 'snk'])

    def test_subsystem_order_preserved(self):
        p = om.Problem()
        for name in ('zeta', 'alpha', 'mu'):
            p.model.add_subsystem(name, Comp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(list(spec['subsystems']), ['zeta', 'alpha', 'mu'])

    def test_auto_order_pinned_false(self):
        spec = generic_model_dict(self._build())
        self.assertIs(spec['auto_order'], False)
        self.assertIs(spec['subsystems']['g']['auto_order'], False)

    def test_parallel_group(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('c1', Comp())
        par.add_subsystem('c2', Comp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertIs(spec['subsystems']['par']['parallel'], True)
        self.assertNotIn('parallel', spec)

    def test_nested_groups(self):
        p = om.Problem()
        outer = p.model.add_subsystem('outer', om.Group())
        inner = outer.add_subsystem('inner', om.Group())
        inner.add_subsystem('c', Comp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        inner_spec = spec['subsystems']['outer']['subsystems']['inner']
        self.assertEqual(inner_spec['type'], GENERIC_GROUP_TYPE)
        self.assertIn('c', inner_spec['subsystems'])

    def test_empty_group(self):
        p = om.Problem()
        p.model.add_subsystem('empty', om.Group())
        p.model.add_subsystem('c', Comp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['subsystems']['empty']['subsystems'], {})

    def test_json_serializable(self):
        spec = generic_model_dict(self._build())
        # no default= hook, so any leaked set, tuple, ndarray or Indexer raises
        reloaded = json.loads(json.dumps(spec))
        self.assertEqual(reloaded, spec)

    def test_system_option_group(self):
        spec = generic_model_dict(self._build(), system='g')
        self.assertEqual(spec['type'], GENERIC_GROUP_TYPE)
        self.assertEqual(sorted(spec['subsystems']), ['c1', 'c2'])
        # the subtree root has no parent in the dump, so its own promotions are dropped
        self.assertNotIn('promotes_inputs', spec)

    def test_system_option_component(self):
        spec = generic_model_dict(self._build(), system='g.c1')
        self.assertEqual(spec['type'], GENERIC_GROUP_TYPE)
        self.assertEqual(list(spec['subsystems']), ['comp'])

        comp = spec['subsystems']['comp']
        self.assertEqual(comp['type'], GENERIC_EXPLICIT_TYPE)
        self.assertIn('x', comp['inputs'])
        self.assertNotIn('promotes_inputs', comp)

    def test_system_not_found(self):
        with self.assertRaises(RuntimeError) as cm:
            generic_model_dict(self._build(), system='nope')
        self.assertEqual(str(cm.exception), "Can't find system with pathname 'nope'.")

    def test_source_is_group(self):
        p = self._build()
        spec = generic_model_dict(p.model._get_subsystem('g'))
        self.assertEqual(sorted(spec['subsystems']), ['c1', 'c2'])

    def test_bad_source_type(self):
        with self.assertRaises(TypeError) as cm:
            generic_model_dict('not a problem')
        self.assertEqual(str(cm.exception), 'Expected a Problem or a System, but got str.')

    def test_before_setup(self):
        p = om.Problem()
        p.model.add_subsystem('c', Comp())

        with self.assertRaises(RuntimeError) as cm:
            generic_model_dict(p)
        self.assertTrue(str(cm.exception).startswith(
            "Can't dump a generic model before setup has been called."))

    def test_final_setup_not_required(self):
        # setup_part2 is needed for dynamic shapes and the auto_ivc, but values are not, so
        # final_setup is run only as far as necessary.
        class Dyn(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True)
                self.add_output('y', val=1.0)

        p = om.Problem()
        src = p.model.add_subsystem('src', om.IndepVarComp())
        src.add_output('o', shape=(4,))
        p.model.add_subsystem('c', Dyn())
        p.model.connect('src.o', 'c.x')
        p.setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['subsystems']['c']['inputs']['x'],
                         {'shape': {'by_conn': True}, 'units': None})
        self.assertLess(p._metadata['setup_status'], _SetupStatus.POST_FINAL_SETUP)

    def test_source_group_before_setup_part2(self):
        p = om.Problem()
        p.model.add_subsystem('c', Comp())
        p.setup()

        spec = generic_model_dict(p.model)
        self.assertIn('c', spec['subsystems'])


@use_tempdirs
class TestDumpGenericModelFile(unittest.TestCase):

    def test_writes_file(self):
        p = om.Problem()
        p.model.add_subsystem('c', Comp())
        p.setup()
        p.final_setup()

        spec = dump_generic_model(p, outfile='out.json')
        self.assertTrue(os.path.isfile('out.json'))

        with open('out.json') as f:
            self.assertEqual(json.load(f), spec)

    def test_default_filename(self):
        p = om.Problem()
        p.model.add_subsystem('c', Comp())
        p.setup()
        p.final_setup()

        dump_generic_model(p)
        self.assertTrue(os.path.isfile('generic_model.json'))

    def test_cmd_dumps_model_with_setup_errors(self):
        # a model that fails setup is the case where seeing its structure matters most
        script = textwrap.dedent("""
            import openmdao.api as om

            class A(om.ExplicitComponent):
                def setup(self):
                    self.add_input('x', val=1.0, units='m')
                    self.add_output('ya', val=1.0)

            class B(om.ExplicitComponent):
                def setup(self):
                    self.add_input('x', val=1.0, units='ft')
                    self.add_output('yb', val=1.0)

            p = om.Problem(name='broken')
            p.model.add_subsystem('a', A(), promotes_inputs=['x'])
            p.model.add_subsystem('b', B(), promotes_inputs=['x'])
            p.setup()
            p.final_setup()
            """)
        with open('broken.py', 'w') as f:
            f.write(script)

        proc = subprocess.run(['openmdao', 'dump_generic', '-o', 'broken.json', 'broken.py'],
                              capture_output=True, text=True)

        # the collected setup errors still stop the script
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('ambiguity', proc.stdout + proc.stderr)

        # but the structure was written first
        self.assertTrue(os.path.isfile('broken.json'))
        with open('broken.json') as f:
            spec = json.load(f)
        self.assertEqual(sorted(spec['subsystems']), ['a', 'b'])
        self.assertEqual(spec['subsystems']['a']['inputs']['x']['units'], 'm')
        self.assertEqual(spec['subsystems']['b']['inputs']['x']['units'], 'ft')


class TestSolvers(unittest.TestCase):
    """
    Only the solver class is carried over, mapped to the om4 equivalent's type key.
    """

    def _group_spec(self, nonlinear=None, linear=None):
        p = om.Problem()
        g = p.model.add_subsystem('g', om.Group())
        g.add_subsystem('c', Comp())
        if nonlinear is not None:
            g.nonlinear_solver = nonlinear
        if linear is not None:
            g.linear_solver = linear
        p.setup()
        p.final_setup()

        return generic_model_dict(p)['subsystems']['g']

    def test_map_keys_are_real_solver_classes(self):
        # guards against a typo in the OpenMDAO half of the mapping table
        for name in _SOLVER_TYPES:
            self.assertTrue(hasattr(om, name), f"openmdao.api has no '{name}'")

    def test_nonlinear_block_gs(self):
        spec = self._group_spec(nonlinear=om.NonlinearBlockGS())
        self.assertEqual(spec['nonlinear_solver'],
                         {'type': 'om4.solvers.nonlinear.nonlinear_block_gs:NonlinearBlockGS'})

    def test_newton(self):
        spec = self._group_spec(nonlinear=om.NewtonSolver(solve_subsystems=False))
        self.assertEqual(spec['nonlinear_solver'],
                         {'type': 'om4.solvers.nonlinear.newton.newton:NewtonSolver'})

    def test_direct(self):
        spec = self._group_spec(linear=om.DirectSolver())
        self.assertEqual(spec['linear_solver'],
                         {'type': 'om4.solvers.linear.direct:DirectSolver'})

    def test_linear_block_gs(self):
        spec = self._group_spec(linear=om.LinearBlockGS())
        self.assertEqual(spec['linear_solver'],
                         {'type': 'om4.solvers.linear.linear_block_gs:LinearBlockGS'})

    def test_scipy_krylov_maps_to_gmres(self):
        # OpenMDAO's ScipyKrylov only offers gmres
        spec = self._group_spec(linear=om.ScipyKrylov())
        self.assertEqual(spec['linear_solver'],
                         {'type': 'om4.solvers.linear.scipy_krylov:ScipyGMRES'})

    def test_no_options_carried_over(self):
        spec = self._group_spec(nonlinear=om.NonlinearBlockGS(maxiter=42, atol=1e-3))
        self.assertEqual(list(spec['nonlinear_solver']), ['type'])

    def test_default_solvers_omitted(self):
        spec = self._group_spec()
        self.assertNotIn('nonlinear_solver', spec)
        self.assertNotIn('linear_solver', spec)
        self.assertNotIn('collapse_by', spec)

    def test_explicit_run_once_omitted(self):
        spec = self._group_spec(nonlinear=om.NonlinearRunOnce(), linear=om.LinearRunOnce())
        self.assertNotIn('nonlinear_solver', spec)
        self.assertNotIn('linear_solver', spec)

    def test_subclass_maps_through_mro(self):
        class MyNewton(om.NewtonSolver):
            pass

        spec = self._group_spec(nonlinear=MyNewton(solve_subsystems=False))
        self.assertEqual(spec['nonlinear_solver'],
                         {'type': 'om4.solvers.nonlinear.newton.newton:NewtonSolver'})

    def test_unmapped_solvers_warn_and_are_omitted(self):
        # BroydenSolver requires a DirectSolver alongside it, hence the per-case kwargs
        cases = [
            ('nonlinear_solver', {'nonlinear': om.NonlinearBlockJac()}),
            ('nonlinear_solver', {'nonlinear': om.BroydenSolver(),
                                  'linear': om.DirectSolver()}),
            ('linear_solver', {'linear': om.LinearBlockJac()}),
            ('linear_solver', {'linear': om.LinearUserDefined()}),
        ]
        for field, kwargs in cases:
            solver = kwargs['nonlinear' if field == 'nonlinear_solver' else 'linear']
            with self.subTest(solver=type(solver).__name__):
                msg = (f"The {field} on 'g' is a {type(solver).__name__}, which has no om4 "
                       "equivalent, so it was omitted.")
                with assert_warning(UserWarning, msg, contains_msg=True):
                    spec = self._group_spec(**kwargs)
                self.assertNotIn(field, spec)

    def test_collapse_by_group_with_nonlinear_solver(self):
        # a group's nonlinear solver covers the whole group in OpenMDAO, so the group has to
        # collapse into one om4 execution node for the solver to cover a non-cyclic implicit
        # child the same way
        spec = self._group_spec(nonlinear=om.NonlinearBlockGS())
        self.assertEqual(spec['collapse_by'], 'group')

    def test_no_collapse_by_for_linear_solver_alone(self):
        spec = self._group_spec(linear=om.DirectSolver())
        self.assertIn('linear_solver', spec)
        self.assertNotIn('collapse_by', spec)

    def test_implicit_component_solvers(self):
        p = om.Problem()
        imp = p.model.add_subsystem('imp', ImpComp())
        imp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        imp.linear_solver = om.DirectSolver()
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)['subsystems']['imp']
        self.assertEqual(spec['nonlinear_solver'],
                         {'type': 'om4.solvers.nonlinear.newton.newton:NewtonSolver'})
        self.assertEqual(spec['linear_solver'],
                         {'type': 'om4.solvers.linear.direct:DirectSolver'})

    def test_explicit_component_solvers_never_emitted(self):
        # om4's ExplicitComponent declares no solver fields, so emitting one would not
        # validate.  OpenMDAO forbids assigning them in the first place, which is asserted
        # here so that a future relaxation on either side does not slip through.
        p = om.Problem()
        comp = p.model.add_subsystem('c', Comp())
        for field in ('nonlinear_solver', 'linear_solver'):
            with self.assertRaises(RuntimeError):
                setattr(comp, field, om.NonlinearBlockGS() if field == 'nonlinear_solver'
                        else om.DirectSolver())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)['subsystems']['c']
        self.assertNotIn('nonlinear_solver', spec)
        self.assertNotIn('linear_solver', spec)

    def test_nested_group_solvers(self):
        p = om.Problem()
        outer = p.model.add_subsystem('outer', om.Group())
        inner = outer.add_subsystem('inner', om.Group())
        inner.add_subsystem('c', Comp())
        inner.nonlinear_solver = om.NonlinearBlockGS()
        p.model.linear_solver = om.DirectSolver()
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['linear_solver'],
                         {'type': 'om4.solvers.linear.direct:DirectSolver'})
        self.assertNotIn('nonlinear_solver', spec['subsystems']['outer'])
        self.assertIn('nonlinear_solver', spec['subsystems']['outer']['subsystems']['inner'])


class TestSrcIndices(unittest.TestCase):
    """
    The JSON form of src_indices must follow om4's IndexSpec rules so om4 can read it back.
    """

    def _conn_spec(self, src_indices, flat_src_indices=False, shape=(2, 3)):
        class Src(om.ExplicitComponent):
            def setup(self):
                self.add_output('y', shape=shape)

        class Snk(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True)
                self.add_output('z', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('src', Src())
        p.model.add_subsystem('snk', Snk())
        p.model.connect('src.y', 'snk.x', src_indices=src_indices,
                        flat_src_indices=flat_src_indices)
        p.setup()
        p.final_setup()

        return generic_model_dict(p)['connections'][0]

    def test_int(self):
        conn = self._conn_spec(3, flat_src_indices=True)
        self.assertEqual(conn['src_indices'], 3)

    def test_array(self):
        conn = self._conn_spec([0, 2, 4], flat_src_indices=True)
        self.assertEqual(conn['src_indices'], [0, 2, 4])
        self.assertIs(conn['flat_src'], True)

    def test_slice(self):
        conn = self._conn_spec(om.slicer[0:4], flat_src_indices=True)
        self.assertEqual(conn['src_indices'], '0:4:1')

    def test_tuple_of_slices(self):
        conn = self._conn_spec(om.slicer[0:2, 1:3])
        self.assertEqual(conn['src_indices'], {'tuple': ['0:2:1', '1:3:1']})

    def test_tuple_with_int(self):
        conn = self._conn_spec(om.slicer[1, 0:2])
        self.assertEqual(conn['src_indices'], {'tuple': [1, '0:2:1']})

    def test_flat_src_false(self):
        conn = self._conn_spec(om.slicer[0:2, 1:3])
        self.assertIs(conn['flat_src'], False)

    def test_ellipsis_falls_back_to_flat_list(self):
        msg = ("src_indices of connection 'src.y' to 'snk.x' have no om4 equivalent and were "
               "flattened into a list of indices.")
        with assert_warning(UserWarning, msg, contains_msg=True):
            conn = self._conn_spec(om.slicer[..., 1])

        self.assertEqual(conn['src_indices'], [1, 4])
        self.assertIs(conn['flat_src'], True)

    def test_serialization_matches_indexer_forms(self):
        # the python index objects om4's IndexSpec accepts, as OpenMDAO's indexer returns them
        from openmdao.utils.generic_model import _serialize_index

        cases = [
            (indexer(3, src_shape=(10,)), 3),
            (indexer(om.slicer[1:5], src_shape=(10,)), '1:5:1'),
            (indexer([0, 2, 4], src_shape=(10,)), [0, 2, 4]),
            (indexer(om.slicer[0:2, 1:3], src_shape=(4, 4)), {'tuple': ['0:2:1', '1:3:1']}),
        ]
        for idxr, expected in cases:
            self.assertEqual(_serialize_index(idxr()), expected)


class TestDynSpecs(unittest.TestCase):
    """
    Dynamic shapes and units become om4 DynSpec objects where om4 can express them.
    """

    def test_shape_and_units_by_conn_on_input(self):
        class Src(om.ExplicitComponent):
            def setup(self):
                self.add_output('y', shape=(4,), units='m')

        class Snk(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape_by_conn=True, units_by_conn=True)
                self.add_output('z', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('src', Src())
        p.model.add_subsystem('snk', Snk())
        p.model.connect('src.y', 'snk.x')
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['subsystems']['snk']['inputs']['x'],
                         {'shape': {'by_conn': True}, 'units': {'by_conn': True}})

    def test_copy_shape_and_units_on_output(self):
        class CopyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape=(3,), units='m')
                self.add_output('y', copy_shape='x', copy_units='x')

        p = om.Problem()
        p.model.add_subsystem('c', CopyComp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        self.assertEqual(spec['subsystems']['c']['outputs']['y'],
                         {'shape': {'copy': 'x'}, 'units': {'copy': 'x'}})

    def test_copy_shape_on_input_falls_back(self):
        class CopyInComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('src', shape=(3,))
                self.add_input('x', copy_shape='src')
                self.add_output('y', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('c', CopyInComp())
        p.setup()
        p.final_setup()

        msg = ("'c.x' declares 'copy_shape' on an input, which has no om4 equivalent, so its "
               "resolved shape was used instead.")
        with assert_warning(UserWarning, msg, contains_msg=True):
            spec = generic_model_dict(p)

        self.assertEqual(spec['subsystems']['c']['inputs']['x']['shape'], [3])

    def test_compute_shape_falls_back(self):
        class ComputeComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape=(3,))
                self.add_output('y', compute_shape=lambda shapes: (2 * shapes['x'][0],))

        p = om.Problem()
        p.model.add_subsystem('c', ComputeComp())
        p.setup()
        p.final_setup()

        msg = ("'c.y' declares 'compute_shape', which has no om4 equivalent, so its resolved "
               "shape was used instead.")
        with assert_warning(UserWarning, msg, contains_msg=True):
            spec = generic_model_dict(p)

        self.assertEqual(spec['subsystems']['c']['outputs']['y']['shape'], [6])

    def test_shape_by_conn_on_output_falls_back(self):
        class DynOut(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', shape_by_conn=True)

        class Snk(om.ExplicitComponent):
            def setup(self):
                self.add_input('z', shape=(5,))
                self.add_output('w', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('c', DynOut())
        p.model.add_subsystem('snk', Snk())
        p.model.connect('c.y', 'snk.z')
        p.setup()
        p.final_setup()

        msg = ("'c.y' declares 'shape_by_conn' on an output, which has no om4 equivalent, so "
               "its resolved shape was used instead.")
        with assert_warning(UserWarning, msg, contains_msg=True):
            spec = generic_model_dict(p)

        self.assertEqual(spec['subsystems']['c']['outputs']['y']['shape'], [5])


class TestGenericModelFidelity(unittest.TestCase):
    """
    Interpret the emitted spec independently and check it against what OpenMDAO resolved.

    This catches promotion routing and connection bugs that reading the JSON would not.
    """

    def _check(self, p, system=None):
        model = p.model if system is None else p.model._get_subsystem(system)
        spec = generic_model_dict(p, system=system)

        proms, conns = _resolve_spec(spec)
        spec_abs2prom = {absname: prom
                         for prom, absnames in proms['input'].items()
                         for absname in absnames}

        om_abs2prom, om_conns = _om_truth(model)

        self.assertEqual(spec_abs2prom, om_abs2prom)
        self.assertEqual(conns, {k: v for k, v in om_conns.items() if v is not None})

    def test_promotions_and_connections(self):
        p = om.Problem()
        model = p.model
        g = model.add_subsystem('g', om.Group(), promotes=['*'])
        g.add_subsystem('c1', Comp(), promotes_inputs=['x'], promotes_outputs=[('y', 'yy')])
        g.add_subsystem('c2', Comp(), promotes_inputs=['x', 'a'])
        model.add_subsystem('d', Comp(), promotes_inputs=['a*'])
        model.connect('yy', 'd.x')
        p.setup()
        p.final_setup()

        self._check(p)

    def test_glob_promotions(self):
        p = om.Problem()
        g = p.model.add_subsystem('g', om.Group(), promotes_inputs=['*'])
        g.add_subsystem('c1', Comp(), promotes=['*'])
        g.add_subsystem('c2', Comp(), promotes_inputs=['a'])
        p.setup()
        p.final_setup()

        self._check(p)

    def test_nested_renames(self):
        p = om.Problem()
        outer = p.model.add_subsystem('outer', om.Group(), promotes_outputs=[('yy', 'zz')])
        outer.add_subsystem('c', Comp(), promotes_outputs=[('y', 'yy')])
        p.model.add_subsystem('snk', Comp())
        p.model.connect('zz', 'snk.x')
        p.setup()
        p.final_setup()

        self._check(p)

    def test_implicit_promotion_connection(self):
        p = om.Problem()
        p.model.add_subsystem('c1', Comp(), promotes_outputs=['y'])
        p.model.add_subsystem('c2', Comp(), promotes_inputs=[('x', 'y')])
        p.setup()
        p.final_setup()

        self._check(p)

    def test_sellar(self):
        from openmdao.test_suite.components.sellar import SellarDerivatives

        p = om.Problem(model=SellarDerivatives())
        p.setup()
        p.final_setup()

        self._check(p)

    def test_input_to_input_without_a_real_source(self):
        class Simple(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

        p = om.Problem()
        p.model.add_subsystem('c1', Simple())
        p.model.add_subsystem('c2', Simple())
        p.model.connect('c1.x', 'c2.x')
        p.setup()
        p.final_setup()

        # both inputs are auto_ivc sourced in OpenMDAO, so both are unconnected in the spec
        self._check(p)

    def test_subtree(self):
        p = om.Problem()
        g = p.model.add_subsystem('g', om.Group())
        g.add_subsystem('c1', Comp(), promotes_outputs=['y'])
        g.add_subsystem('c2', Comp(), promotes_inputs=[('x', 'y')])
        p.setup()
        p.final_setup()

        self._check(p, system='g')


@unittest.skipUnless(MPI and PETScVector, 'MPI and PETSc are required.')
class TestGenericModelMPI(unittest.TestCase):

    N_PROCS = 2

    def test_parallel_group_children_gathered(self):
        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('c1', Comp())
        par.add_subsystem('c2', Comp())
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        par_spec = spec['subsystems']['par']

        # both children must be present on both ranks, in declaration order
        self.assertEqual(list(par_spec['subsystems']), ['c1', 'c2'])
        for name in ('c1', 'c2'):
            self.assertEqual(par_spec['subsystems'][name]['inputs']['x']['shape'], [3])

    def test_distributed_var(self):
        class DistComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', shape=3, distributed=True)
                self.add_output('y', shape=3, distributed=True)

        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('src', shape=3, distributed=True)
        p.model.add_subsystem('c', DistComp())
        p.model.connect('ivc.src', 'c.x')
        p.setup()
        p.final_setup()

        spec = generic_model_dict(p)
        var = spec['subsystems']['c']['inputs']['x']
        self.assertIs(var['distributed'], True)
        # the local shape is recorded; how the variable is allocated is out of scope
        self.assertEqual(var['shape'], [3])


if __name__ == '__main__':
    unittest.main()
