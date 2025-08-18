"""
Utility functions for working with configuration of an OpenMDAO problem.
"""

import os
import importlib
import yaml
from io import StringIO

from openmdao.utils.om_warnings import issue_warning


# if a module path is not specified, use this map to find the default module path for a given key
_key2module_map = {
    'problem': 'openmdao.core.problem.Problem',
    'group': 'openmdao.core.group.Group',
}


def load_config(fname):
    """
    Load a configuration file and return the corresponding dict.

    Paramters
    ---------
    fname : str
        Name of the file containing the configuration.
    """
    with open(fname, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_config(cfg):
    """
    Return a dict configuration.

    Parameters
    ----------
    cfg : str or dict
        If a string, may be a yaml file name or a yaml string.  If a dict, just return it.

    Returns
    -------
    dict
        The configuration in dict form.
    """
    if isinstance(cfg, dict):
        return cfg

    if os.path.isfile(cfg):
        return load_config(cfg)

    # assume it's a YAML string
    if isinstance(cfg, str):
        if '\n' in cfg:
            newcfg = yaml.safe_load(StringIO(cfg))
            if isinstance(newcfg, dict) and newcfg:
                return newcfg
        raise RuntimeError("Given configuration is not a YAML filename or a YAML string.")

    raise RuntimeError(f"resolve_config expects a dict or string, but got a {type(cfg).__name__}.")


def process_config(cfg, topname='problem'):
    """
    Use this to create a top level object with the given name.
    """
    global _key2module_map

    cfg = resolve_config(cfg)

    scope = []  # name stack to keep track of pathname for error reporting

    if topname in cfg:
        default_type_path = _key2module_map.get(topname)
        top = configure_type(topname, cfg[topname], scope=scope,
                             default_type_path=default_type_path)
    else:
        raise RuntimeError(f"Key '{topname}' not found in configuration.")

    keyset = set(cfg) - {topname}
    if len(keyset) > 0:
        issue_warning(f"The following top level YAML keys were not recognized: {sorted(keyset)}.")

    return top


def set_config(instance, cfg, scope=None, verbose=False):
    """
    Set the configuration for an instance.
    """
    if scope is None:
        scope = instance._get_config_scope_stack()
    klass = type(instance)
    config_functs = klass.get_config_handlers()
    ignored = []
    for name, subcfg in resolve_config(cfg).items():
        if name in config_functs:
            config_functs[name](instance, name, subcfg, scope)
        elif verbose and name != 'type':
            ignored.append(name)
    if ignored:
        issue_warning(scope_msg(scope,
                                f"During loading of a configuration, the following items "
                                f"were ignored: {sorted(ignored)}."))


def scope_msg(scope, msg):
    return f"{'.'.join(scope)}: {msg}"


def import_type(module_path, scope):
    """
    Return the type referred to by the module_path.

    Parameters
    ----------
    module_path : str
        The module path of the type to import.
    scope : list
        Stack of names to determine current pathname.

    Returns
    -------
    type or None
        The type referred to by the module_path.
    """
    # TODO: add support for nested classes by changing split location and retrying until module
    #       is found
    parts = module_path.split('.')
    if len(parts) >= 2:
        mod_name = '.'.join(parts[:-1])
        type_name = parts[-1]
        try:
            module = importlib.import_module(mod_name)
        except ImportError:
            raise ImportError(scope_msg(scope, f"Failed to import module '{mod_name}'."))

        try:
            return getattr(module, type_name)
        except AttributeError:
            raise AttributeError(scope_msg(scope, f"No attribute '{type_name}' found in "
                                           f"module '{mod_name}'."))


def configure_type(name, cfg, scope, strict=True, default_type_path=None, verbose=False):
    """
    Instantiate an object based on type information and optional args and/kwargs.

    A type can use default initialization, in which case you can declare it as as:

      instance_name: module.path.to.class

    OR, for initialization with args and/or kwargs,

      instance_name:
        type: module.path.to.class
        args:   # optional positional args (list)
          - arg1
          - arg2
        kwargs:  # optional keyword args (dict)
          kw1: foo
          kw2: 2.3

    Parameters
    ----------
    name : str
        The name of the configuration.
    cfg : any
        Typically a configuration dict, but could be the module path of a type or if strict is
        False could be any object.
    scope : list
        Stack of names to determine current pathname.
    strict : bool
        If False, cfg can be any object and if it isn't a type config it will just be returned.
    """

    # the cfg passed into this function with either be a str (the module path for a default inst),
    # or a dict containing type (the module path) and optionally args and/or kwargs.

    if isinstance(cfg, dict):  # non-default init args
        module_path = cfg.get('type')
        if module_path is None:
            if default_type_path is None:
                raise RuntimeError(scope_msg(scope,
                                             f"'type' not specified for instance '{name}'."))
            else:
                module_path = default_type_path
        # instantiate a type specified by the given module path to the class, passing it
        # *args and **kwargs if they exist
        nkeys = len(cfg) - 1
        if 'args' in cfg:
            args = cfg['args']
            nkeys -= 1
        else:
            args = []
        if 'kwargs' in cfg:
            kwargs = cfg['kwargs']
            nkeys -= 1
        else:
            kwargs = {}
        klass = import_type(module_path, scope)
        instance = klass(*args, **kwargs)
        if nkeys > 0:
            scope.append(name)
            set_config(instance, cfg, scope, verbose=verbose)
            scope.pop()

    elif isinstance(cfg, str) and '.' in cfg:
        typ = import_type(cfg, scope)
        if typ is None and not strict:
            return cfg
        instance = typ()

    elif not strict:
        return cfg

    else:
        raise RuntimeError(scope_msg(scope,
                                     "Expected a type configuration but got a "
                                     f"{type(cfg).__name__}."))

    return instance


def configure_type_list(name, lst, scope):
    """
    Process a list of type configs.
    """
    scope.append(name)
    instances = []
    for cfg in lst:
        if isinstance(cfg, dict):
            assert len(cfg) == 1, "Entry of type list should only have one entry if it's a dict."
            for name, subcfg in cfg.items():
                instances.append((name, configure_type(name, subcfg, scope)))
        else:
            raise RuntimeError(scope_msg(scope,
                                         "Expected all members of type list to be dicts, but got "
                                         f"{type(cfg).__name__}."))

    scope.pop()
    return instances


# Helper functions to update an instance for common configuration items

def attr_config(instance, key, cfg, scope):
    """
    Process a config and set attribute in instance corresponding to key.

    The config may be a type definition or just a simple value.
    """
    if hasattr(instance, key):
        obj = configure_type(key, cfg, scope, strict=False)
        setattr(instance, key, obj)
    else:
        raise RuntimeError(f"Attribute '{key}' not found in instance of type "
                           f"'{type(instance).__name__}.")


def dict_like_config(instance, dict_name, dct, scope):
    """
    Set multiple items in a dict-like attribute in the instance.

    This is also used for OptionsDictionary objects.
    """
    scope.append(dict_name)
    dictattr = getattr(instance, dict_name)

    for name, val in dct.items():
        dictattr[name] = configure_type(name, val, scope, strict=False)

    scope.pop()


_voi_func_map = {
    'design_variables': 'add_design_var',
    'constraints': 'add_constraint',
    'responses': 'add_response',
    'objectives': 'add_objective'
}


def voi_list_config(instance, lstname, lst, scope):
    """
    Process a group of design vars, constraints, responses, or objectives.
    """
    global _voi_func_map

    try:
        voifunc = getattr(instance, _voi_func_map[lstname])
    except KeyError:
        raise RuntimeError(scope_msg(scope, f"Unrecognized voi group name '{lstname}'."))

    scope.append(lstname)

    for entry in lst:
        if isinstance(entry, dict):
            for name, kwargs in entry.items():
                voifunc(name, **kwargs)
        else:
            voifunc(entry)

    scope.pop()


def connection_list_config(instance, connsname, lst, scope):
    scope.append(connsname)
    for conndct in lst:
        if isinstance(conndct, dict):
            skip = ('src', 'tgt')
            kwargs = {k: v for k, v in conndct.items() if k not in skip}
            missing = [n for n in ('src', 'tgt') if n not in conndct]
            if missing:
                raise RuntimeError(scope_msg(scope,
                                             f"Key(s) {missing} were not found when declaring "
                                             "a connection."))
            else:
                instance.connect(conndct['src'], conndct['tgt'], **kwargs)
        else:
            raise TypeError(scope_msg(scope,
                                      "Entries in connections list should be dicts, but got "
                                      f"{type(conndct).__name__} instead."))
    scope.pop()


def subsystem_list_config(instance, lstname, lst, scope):
    scope.append(lstname)
    kwargnames = {'promotes_inputs', 'promotes_outputs', 'promotes', 'min_procs', 'max_procs',
                  'proc_weight', 'proc_group'}
    for subsysdct in lst:
        if isinstance(subsysdct, dict):
            kwargs = {}
            sub = None
            subname = None
            for name, val in subsysdct.items():
                if name in kwargnames:
                    kwargs[name] = val
                elif sub is None:
                    sub = configure_type(name, val, scope)
                    subname = name
                else:
                    raise RuntimeError(scope_msg(scope,
                                                 "Only one subsystem per list entry is allowed."))

            instance.add_subsystem(subname, sub, **kwargs)
        else:
            raise TypeError(scope_msg(scope,
                                      "Entries in subsystems list should be dicts, but got "
                                      f"{type(subsysdct).__name__} instead."))
    scope.pop()


def input_defaults_list_config(instance, lstname, lst, scope):
    scope.append(lstname)
    for dct in lst:
        if isinstance(dct, dict):
            if len(dct) == 1:
                # single key, so assume key is variable name
                for vname, data in dct.items():
                    break
                if isinstance(data, dict):
                    kwargs = data
                else:
                    kwargs = {'val': data}
            else:
                kwargnames = {'val', 'units', 'src_shape'}
                kwargs = {}
                vname = None
                for key, data in dct.items():
                    if key in kwargnames:
                        kwargs[key] = data
                    elif vname is None:
                        vname = key
                    else:
                        raise RuntimeError(scope_msg(scope, f"Unrecognized key: '{key}'."))

                if vname is None:
                    raise RuntimeError(scope_msg(scope, "No variable name specified when "
                                                 "specifying input defaults."))

            instance.set_input_defaults(vname, **kwargs)
        else:
            raise TypeError(scope_msg(scope,
                                      "Entries in input_defaults list should be dicts, but got "
                                      f"{type(dct).__name__} instead."))
    scope.pop()
