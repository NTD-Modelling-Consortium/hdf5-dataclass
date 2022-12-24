__all__ = ["serialisable"]

import dataclasses

# from pydantic import BaseModel
from pathlib import Path
from typing import IO, Any, Type, TypeVar
import typing
import types

import numpy as np
import h5py

FileType = str | Path | IO[bytes]


# TODO:
# * pydantic -- I think just adapt _fields & _is_class_serialisable for pydantic models
# * list (list of primitives only?)


class AbstractSerialisable:
    def serialise(self, output: FileType | h5py.File | h5py.Group) -> None:
        raise NotImplementedError()

    @classmethod
    def deserialise(cls, input: FileType | h5py.File | h5py.Group):
        raise NotImplementedError()


def _is_primitive(obj_or_class: Any) -> bool:
    primitives = (int, float, str)
    if type(obj_or_class) == type:
        return obj_or_class in primitives
    else:
        return isinstance(obj_or_class, primitives)


def _is_union(T: type) -> bool:
    # One is for A | B, another is for Union[A, B]. Shrug.
    return typing.get_origin(T) in (types.UnionType, typing.Union)


def _is_optional(T: type) -> bool:
    if not _is_union(T):
        return False
    args = typing.get_args(T)
    if len(args) != 2:
        return False
    T1, T2 = args
    return T1 != T2 and types.NoneType in (T1, T2)


def _extract_type_from_optional(T: type) -> type:
    assert _is_optional(T)
    T1, T2 = typing.get_args(T)
    return T1 if T2 == types.NoneType else T2


def _is_optional_primitive(T: type) -> bool:
    return _is_optional(T) and _is_primitive(_extract_type_from_optional(T))


def _is_numpy_array(T: type) -> bool:
    return typing.get_origin(T) == np.ndarray


def _is_class_serialisable(T: type) -> bool:
    # TODO: or a serialisable pydantic model
    return dataclasses.is_dataclass(T) and issubclass(T, AbstractSerialisable)


def _is_supported_dict(T: type) -> bool:
    if not typing.get_origin(T) == dict:
        return False
    K, V = typing.get_args(T)
    return _is_primitive(K) and _is_type_supported(V)


def _is_type_supported(T: type) -> bool:
    return (
        _is_primitive(T)
        or (_is_optional(T) and _is_type_supported(_extract_type_from_optional(T)))
        or _is_numpy_array(T)
        or _is_class_serialisable(T)
        or _is_supported_dict(T)
    )


def _fields(T: type) -> dict[str, type]:
    # TODO: two separate implementations - one for dataclass, one for pydantic model
    ret: dict[str, type] = {}
    for field in dataclasses.fields(T):
        ret[field.name] = field.type
    return ret


def _serialise(
    obj, output: FileType | h5py.File | h5py.Group, serialisable_attrs: dict[str, type]
):
    def serialise_single(
        attr: str, val: Any, T: type, h5: h5py.File | h5py.Group
    ) -> None:
        if val is None:
            return

        if _is_primitive(T):
            assert val is not None
            h5.attrs[attr] = val
        elif _is_optional_primitive(T) and val is not None:
            h5.attrs[attr] = val
        # TODO: elif dict/list -- json? check size?
        elif _is_numpy_array(T):
            h5.create_dataset(attr, data=val)
        elif _is_class_serialisable(T):
            grp = h5.create_group(attr)
            val.serialise(output=grp)
        elif _is_supported_dict(T):
            _, V = typing.get_args(T)
            grp = h5.create_group(attr)
            for k, v in val.items():
                serialise_single(k, v, V, grp)
        else:
            raise Exception(f"Unsupported type of attribute '{attr}'")

    h5 = (
        output
        if isinstance(output, (h5py.File, h5py.Group))
        else h5py.File(output, "w")
    )

    for attr, T in serialisable_attrs.items():
        val = getattr(obj, attr)
        serialise_single(attr, val, T, h5)


Serialise = TypeVar("Serialise", bound=AbstractSerialisable)


def _deserialise(
    class_type: Type[Serialise],
    input: FileType | h5py.File | h5py.Group,
    serialisable_attrs: dict[str, type],
) -> Serialise:
    def deserialise_single(attr: str, T: type, h5: h5py.File | h5py.Group) -> Any:
        val = None
        if _is_primitive(T):
            val = h5.attrs.get(attr)
            assert (
                val is not None
            ), f"Attribute '{attr}' marked as non-optional, but value is not present!"
        elif _is_optional_primitive(T):
            val = h5.attrs.get(attr)
        else:
            serialised = h5[attr]
            if isinstance(serialised, h5py.Dataset):
                val = np.array(serialised)
            elif isinstance(serialised, h5py.Group):
                assert _is_class_serialisable(T) or _is_supported_dict(T)
                if _is_class_serialisable(T):
                    val = T.deserialise(serialised)
                else:
                    # dict case
                    _, V = typing.get_args(T)
                    val = {}
                    keys = (
                        serialised.attrs.keys()
                        if _is_primitive(V) or _is_optional_primitive(V)
                        else serialised.keys()
                    )
                    for key in keys:
                        val[key] = deserialise_single(key, V, serialised)
            else:
                raise Exception("Unknown type of data in hdf5")
        return val

    h5 = input if isinstance(input, (h5py.File, h5py.Group)) else h5py.File(input, "r")

    attrs = {}
    for attr, T in serialisable_attrs.items():
        attrs[attr] = deserialise_single(attr, T, h5)

    return class_type(**attrs)


def serialisable(input_class):
    serialisable_attrs = _fields(input_class)
    unsupported_attrs = [
        attr for attr, T in serialisable_attrs.items() if not _is_type_supported(T)
    ]
    assert (
        not unsupported_attrs
    ), f"Types of attributes {', '.join(unsupported_attrs)} are not supported!"

    class NewCls(input_class, AbstractSerialisable):
        def serialise(self, output: FileType | h5py.File | h5py.Group) -> None:
            return _serialise(self, output, serialisable_attrs)

        @classmethod
        def deserialise(cls, input: FileType | h5py.File | h5py.Group):
            return _deserialise(cls, input, serialisable_attrs)

    return NewCls
