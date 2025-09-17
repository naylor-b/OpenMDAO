"""
Validation utilities for OpenMDAO.
"""
from pydantic import BaseModel, Field, ConfigDict
from pydantic_core import core_schema
from typing import List, Dict, Optional, Type, Any, Iterator, Tuple
import importlib
import inspect

from openmdao.core.constants import _UNDEFINED
from openmdao.visualization.tables.table_builder import generate_table


def _class_to_type(cls):
    prefix = '' if cls.__module__ == '__main__' else f"{cls.__module__}."
    return f"{prefix}{cls.__qualname__}"


class _TypeBaseModel(BaseModel):
    """
    Base class for 'polymorphic' data models.
    """

    # This will catch typos. Otherwise, by default extra fields are silently ignored.
    # This behavior can be overridden in subclasses.
    model_config = ConfigDict(extra="forbid")

    type: str = Field(default=None, desc='The class path of the type to be instantiated.')
    args: Optional[List[Any]] = Field(default_factory=list,
                                      desc="Positional arguments to pass to __init__.")
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                             desc="Keyword arguments to pass to __init__.")


# -------------------
# Polymorphic dispatcher
# -------------------
def _poly_validate(value: Any) -> BaseModel:
    if isinstance(value, dict) and "type" in value:
        t = value["type"]
        model_cls = DataModelManager.type_to_data_model(t)
        if not model_cls:
            raise ValueError(f"Unknown type: {t}")
        return model_cls.model_validate(value)
    if isinstance(value, BaseModel):
        return value
    raise TypeError(f"Cannot coerce {value!r} into a polymorphic model")


# -------------------
# Custom Pydantic type
# -------------------
class PolymorphicModel:
    """
    A pydantic-compatible type that auto-dispatches to registered _TypeBaseModels.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: Any) -> core_schema.CoreSchema:
        """
        Get the pydantic core schema for the PolymorphicModel type.

        Parameters
        ----------
        _source_type : Any
            The source type.
        _handler : Any
            The handler.

        Returns
        -------
        core_schema.CoreSchema
            The pydantic core schema.
        """
        return core_schema.no_info_after_validator_function(
            _poly_validate,
            core_schema.any_schema(),
        )


class _ValidateOnAssignModel(BaseModel):
    """
    BaseModel that validates on assignment.

    This is used to ensure that options are validated on assignment, not just on initialization.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class _VOIModel(_ValidateOnAssignModel):
    """
    BaseModel for Variables of Interest (Design Variables, Constraints, and Objectives).
    """

    name: str
    lower: float = Field(default=None, desc="Lower bound.")
    upper: float = Field(default=None, desc="Upper bound.")
    ref: float = Field(default=None)
    ref0: float = Field(default=None)
    indices: List[int] = Field(default=None)
    adder: float = Field(default=None)
    scaler: float = Field(default=None)
    units: str = Field(default=None)
    parallel_deriv_color: str = Field(default=None,
                                      desc="Parallel derivative color.")
    cache_linear_solution: bool = Field(default=None,
                                        desc="Cache linear solution.")
    flat_indices: bool = Field(default=None, desc="Assume indices into a flat source array.")


class _DesignVariableModel(_VOIModel):
    """
    BaseModel for design variables.
    """

    pass


class _ResponseModel(_VOIModel):
    """
    BaseModel for responses.
    """

    equals: float = Field(default=None, desc="Equality constraint value for the response.")
    index: int = Field(default=None, desc="Index for the response.")
    linear: bool = Field(default=None, desc="Linear for the response.")
    alias: str = Field(default=None, desc="Alias for the response.")


class _ConstraintModel(_ResponseModel):
    """
    BaseModel for constraints.
    """

    pass


class _ObjectiveModel(_ResponseModel):
    """
    BaseModel for objectives.
    """

    pass


class _OptionsBaseModel(_ValidateOnAssignModel):
    """
    BaseModel for options that attempts to mimic OptionsDictionary behavior.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def __getitem__(self, name: str) -> Any:
        """
        Get an option from the model.
        """
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        """
        Set an option in the model.
        """
        setattr(self, name, value)

    def __contains__(self, name: str) -> bool:
        """
        Check if the option is in the model.
        """
        return name in self.__class__.model_fields

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the options in the model.
        """
        return iter(self.model_fields)

    def items(self, recordable_only=False) -> Iterator[Tuple[str, Any]]:
        """
        Iterate over the options in the model.

        Parameters
        ----------
        recordable_only : bool
            If True, only return recordable options.

        Returns
        -------
        Iterator[Tuple[str, Any]]
            An iterator over the options in the model.
        """
        for key, meta in self.model_fields.items():
            if not recordable_only or not meta.exclude:
                yield key, getattr(self, key)

    def values(self) -> Iterator[Any]:
        """
        Iterate over the values of the options in the model.
        """
        return [getattr(self, key) for key in self.model_fields]

    def keys(self) -> Iterator[str]:
        """
        Iterate over the keys of the options in the model.
        """
        return iter(self.model_fields)

    def __len__(self) -> int:
        """
        Return the number of options in the model.
        """
        return len(self.model_fields)

    def update(self, dct: Dict[str, Any]):
        for name, value in dct.items():
            setattr(self, name, value)

    def set(self, **kwargs):
        self.update(kwargs)

    def get_meta(self, key):
        """
        Get the metadata for an option.

        Parameters
        ----------
        key : str
            The name of the option.

        Returns
        -------
        dict
            A dictionary of the option's value and recordability.
        """
        pydantic_meta = self.model_fields[key]
        meta = {
            'val': getattr(self, key),
            'recordable': not pydantic_meta.exclude,
        }
        return meta

    def is_serializable(self, key):
        """
        Check if the option is serializable.

        Parameters
        ----------
        key : str
            The name of the option.

        Returns
        -------
        bool
            Whether the option is serializable.
        """
        return not self.model_fields[key].exclude

    def raw_items(self):
        """
        Yield a dict wrapped around a value for compatibility with the old OptionsDictionary.

        Yields
        ------
        key : str
            The name of the option.
        wrapper : dict
            A dictionary of the option's value and recordability.
        """
        for key, info in self.model_fields.items():
            wrapper = {}
            wrapper['val'] = getattr(self, key)
            wrapper['recordable'] = not info.exclude
            yield key, wrapper

    def is_read_only(self, key: str) -> bool:
        """
        Check if the option is read-only.
        """
        fieldinfo = self.model_fields[key]
        try:
            return fieldinfo.json_schema()['readOnly']
        except (KeyError, AttributeError):
            return False

    def to_table(self, fmt='github', missingval='N/A', max_width=None, display=True):
        """
        Get a table representation of this OptionsDictionary as a table in the requested format.

        Parameters
        ----------
        fmt : str
            The formatting of the requested table.  Options are
            ['github', 'rst', 'text', 'html', 'tabulator'] and several 'grid' and 'outline'
            formats that mimic those found in the python 'tabulate' library.
            Default value of 'github' produces a table in GitHub-flavored markdown.
            'html' and 'tabulator' produce output viewable in a browser.
        missingval : str
            The value to be displayed in place of None.
        max_width : int or None
            If not None, try to limit the total width of the table to this value.
        display : bool
            If True, display the table, typically by writing it to stdout or opening a
            browser.

        Returns
        -------
        str
            A string representation of the table in the requested format.
        """
        hdrs = ['Option', 'Default', 'Acceptable Values', 'Acceptable Types', 'Description']
        rows = []

        # deprecations = False
        # for meta in self._dict.values():
        #     if meta['deprecation'] is not None:
        #         deprecations = True
        #         hdrs.append('Deprecation')
        #         break

        for key in sorted(self.model_fields.keys()):
            option = getattr(self, key)
            default = option if option is not _UNDEFINED else '**Required**'
            default_str = str(default)

            # if the default is an object instance, replace with the (unqualified) object type
            idx = default_str.find(' object at ')
            if idx >= 0 and default_str[0] == '<':
                parts = default_str[:idx].split('.')
                default = parts[-1]

            # acceptable_values = option['values']
            # if acceptable_values is not None:
            #     if not isinstance(acceptable_values, (set, tuple, list)):
            #         acceptable_values = (acceptable_values,)
            #     acceptable_values = [value for value in acceptable_values]

            # acceptable_types = option['types']
            # if acceptable_types is not None:
            #     if not isinstance(acceptable_types, (set, tuple, list)):
            #         acceptable_types = (acceptable_types,)
            #     acceptable_types = [type_.__name__ for type_ in acceptable_types]

            desc = option['desc']

            # deprecation = option['deprecation']
            # if deprecation is not None:
            #     deprecation = deprecation[0]

            # if deprecations:
            #     rows.append([key, default, acceptable_values, acceptable_types, desc,
            #                  deprecation])
            # else:
            rows.append([key, default, desc])  # acceptable_values, acceptable_types, desc])

        kwargs = {
            'tablefmt': fmt,
            'headers': hdrs,
            'missing_val': missingval,
            'max_width': max_width,
        }
        if fmt == 'tabulator':
            kwargs['filter'] = False
            kwargs['sort'] = False

        tab = generate_table(rows, **kwargs)

        if display:
            tab.display()

        return str(tab)


class DataModelManager:
    """
    Manager of the mapping between pydantic models and their corresponding classes.
    """

    # type path --> (class, Pydantic model)
    MODELS: Dict[str, Tuple[Type[Any], Type[BaseModel]]] = {}

    @classmethod
    def register(cls, class_: Type[BaseModel]):
        """
        Class decorator to bind a Pydantic model to an OpenMDAO class.

        Both the class and the Pydantic model will then be retrievable using the type path.
        This should wrap the *data model* class, not the corresponding OpenMDAO class.

        Parameters
        ----------
        class_ : Type[BaseModel]
            The class to bind the Pydantic model to.

        Returns
        -------
        Type[BaseModel]
            The Pydantic model.
        """
        def decorator(pydantic_model: Type):
            type_path = _class_to_type(class_)
            pydantic_model.type = type_path
            cls.MODELS[type_path] = (class_, pydantic_model)
            return pydantic_model
        return decorator

    @classmethod
    def type_to_info(cls, model_type: str) -> Tuple[Type[Any], Type[BaseModel]]:
        """
        Retrieve a class and Pydantic model from the registry using the type path.

        Parameters
        ----------
        model_type : str
            The type path to retrieve the class and Pydantic model for.

        Returns
        -------
        Tuple[Type[Any], Type[BaseModel]]
            The class and Pydantic model.
        """
        if model_type not in cls.MODELS:
            try:
                # Dynamically import the module to trigger the decorator
                module_path, _, class_name = model_type.rpartition('.')
                if module_path:
                    mod = importlib.import_module(module_path)
                elif model_type not in globals():
                    raise RuntimeError(f"Can't find type '{model_type}'.")
            except (ImportError, AttributeError) as e:
                raise RuntimeError(f"Failed to import module for type '{model_type}': {e}")

            # some classes may not register themselves and just use the data model that they
            # inherit, so we need to look through the base classes to find the right data model.
            if model_type not in cls.MODELS:
                try:
                    klass = getattr(mod, class_name)
                except AttributeError:
                    raise RuntimeError(f"Class '{class_name}' not found in module '{module_path}'.")

                return cls.get_from_base(klass, model_type)

        return cls.MODELS[model_type]

    @classmethod
    def type_to_class(cls, model_type: str) -> Type[Any]:
        """
        Retrieve a class from the registry using the type path.

        Parameters
        ----------
        model_type : str
            The type path to retrieve the class for.

        Returns
        -------
        Type[Any]
            The class.
        """
        return cls.type_to_info(model_type)[0]

    @classmethod
    def get_from_base(cls, klass: Type[Any], model_type: str):
        """
        Retrieve a base class' Pydantic model when this class is not registered.

        Parameters
        ----------
        klass : Type[Any]
            The class to retrieve the Pydantic model for.
        model_type : str
            The type path to retrieve the Pydantic model for.

        Returns
        -------
        Tuple[Type[Any], Type[BaseModel]]
            The class and Pydantic model.
        """
        MODELS = cls.MODELS
        for base in klass.__mro__[1:]:
            class_path = _class_to_type(base)
            if class_path in MODELS:
                _, parent_model = MODELS[class_path]
                MODELS[model_type] = (klass, parent_model)
                return MODELS[model_type]

        raise RuntimeError(f"No data model registered for type '{model_type}'.")

    @classmethod
    def type_to_data_model(cls, model_type: str) -> Type[BaseModel]:
        """
        Retrieve a Pydantic model classfrom the registry using the type path.

        Parameters
        ----------
        model_type : str
            The type path to retrieve the Pydantic model for.

        Returns
        -------
        Type[BaseModel]
            The Pydantic model class.
        """
        return cls.type_to_info(model_type)[1]

    @classmethod
    def type_to_class_instance(cls, type_path: str) -> str:
        """
        Retrieve the type path from the registry using the type path.

        Parameters
        ----------
        type_path : str
            The type path to retrieve the type path for.

        Returns
        -------
        str
            The type path.
        """
        class_, data_model = cls.type_to_info(type_path)
        return class_.from_data_model(data_model)

    @classmethod
    def class_to_data_model(cls, klass: Type[Any]) -> Type[BaseModel]:
        """
        Given a class, return the associated Pydantic model.

        Parameters
        ----------
        klass : Type[Any]
            The class to retrieve the Pydantic model for.

        Returns
        -------
        Type[BaseModel]
            The Pydantic model class.
        """
        type_path = _class_to_type(klass)
        if type_path not in cls.MODELS:
            return cls.get_from_base(klass, type_path)[1]
        return cls.MODELS[type_path][1]

    @classmethod
    def class_to_data_model_instance(cls, klass: Type[Any], **kwargs) -> BaseModel:
        """
        Given a class, return an instance of the associated Pydantic model instance.

        Parameters
        ----------
        klass : Type[Any]
            The class to retrieve the Pydantic model for.
        **kwargs : dict
            The keyword arguments to pass to the Pydantic model.

        Returns
        -------
        Type[BaseModel]
            The Pydantic model instance.
        """
        type_path = _class_to_type(klass)
        dm = cls.class_to_data_model(klass)
        kwargs = kwargs.copy()
        kwargs['type'] = type_path
        return dm(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to create an instance from.

        Returns
        -------
        Any
            An instance of the given class.
        """
        try:
            type_path = data['type']
        except KeyError:
            raise ValueError("Missing 'type' field in data.")

        _, data_model_class = cls.type_to_info(type_path)
        dm_instance = data_model_class.model_validate(data)
        return cls.from_data_model(dm_instance)

    @classmethod
    def from_data_model(cls, data_model: _TypeBaseModel, orig: Any = None) -> Any:
        """
        Create an instance from a data model.

        Parameters
        ----------
        data_model : _TypeBaseModel
            The data model to create an instance from.
        orig : Any, optional
            The original instance to update from the data model.

        Returns
        -------
        Any
            An instance of the given class.
        """
        class_, _ = cls.type_to_info(data_model.type)
        if orig is not None and isinstance(orig, class_):
            orig.update_from_data_model(data_model)
            return orig
        return cls.inst_from_type_model(class_, data_model)

    @staticmethod
    def inst_from_type_model(klass: Type[Any], data_model: _TypeBaseModel):
        """
        Create an instance of the given class, passing args and kwargs from the data model.

        Parameters
        ----------
        klass : Type[Any]
            The class to instantiate.
        data_model : _TypeBaseModel
            The data model to use for the instance.

        Returns
        -------
        Any
            An instance of the given class.
        """
        args = data_model.args
        kwargs = data_model.kwargs.copy()
        kwargs['data_model'] = data_model
        inst = klass(*args, **kwargs)
        return inst

    @staticmethod
    def setup_data_model(inst: Any, kwargs: Dict[str, Any]):
        """
        Ensure the data model for an instance is fully initialized.

        Parameters
        ----------
        inst : Any
            The instance to setup the data model for.
        kwargs : Dict[str, Any]
            The keyword arguments to use for the instance.
        """
        data_model = kwargs.pop('data_model', None)
        if data_model is None:
            data_model = inst.init_data_model()
        else:
            inst.data_model = data_model
            inst.update_from_data_model(data_model)

        if 'options' in data_model.__class__.model_fields:
            inst.data_model.options.update(kwargs)

    @staticmethod
    def field(annotation: Any, **field_kwargs):
        """
        Create an annotated field tuple.

        Used when creating a data model class dynamically.

        Parameters
        ----------
        annotation : Any
            The annotation for the field.
        **field_kwargs : dict
            The keyword arguments to pass to the Field factory function.

        Returns
        -------
        Tuple[Any, Field]
            The annotated field tuple.
        """
        return (annotation, Field(**field_kwargs))

    @staticmethod
    def create_class(class_name: str, model_config: ConfigDict = None, base=None,
                    **kwargs) -> Type[BaseModel]:
        """
        Create a Pydantic model class from fields and annotations.

        Parameters
        ----------
        class_name : str
            The name of the class to create.
        model_config : ConfigDict, optional
            The model configuration to use for the class.
        base : Type[BaseModel], optional
            The base class to inherit from.
        **kwargs : Dict[str, Any]
            Each named argument should be either a type or the return value of a call to
            field(annotation_type, **field_kwargs).  **field_kwargs are passed to the
            Field factory function.

        Returns
        -------
        Type[BaseModel]
            The created Pydantic model class.
        """
        # pydantic needs an attribute dict with a specific format
        # fieldinfo is the return value of the Field factory function
        attrs = {
            '__annotations__': {},
            'model_config': model_config,
            '__module__': inspect.currentframe().f_back.f_globals['__name__']
        }
        for field_name, info in kwargs.items():
            if isinstance(info, tuple):
                annotation, fieldinfo = info
                attrs[field_name] = fieldinfo
                attrs['__annotations__'][field_name] = annotation
            elif isinstance(info, type):
                attrs[field_name] = None
                attrs['__annotations__'][field_name] = info
            else:
                raise ValueError(f"Invalid argument: {field_name} = {info}")

        if base is None:
            base = BaseModel

        return type(class_name, (base,), attrs)
