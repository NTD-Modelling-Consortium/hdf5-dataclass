import dataclasses
from typing import Any, Union, get_args, get_origin, TypeGuard, Type
import types

from pydantic import BaseModel
import numpy as np


def fields(T: type) -> dict[str, type]:
    assert dataclasses.is_dataclass(T)
    ret: dict[str, type] = {}
    for field in dataclasses.fields(T):
        # Only serialise init fields
        if field.init:
            ret[field.name] = field.type
    return ret


def is_primitive(obj_or_class: Any) -> bool:
    primitives = (int, float, str)
    if type(obj_or_class) == type:
        return obj_or_class in primitives
    else:
        return isinstance(obj_or_class, primitives)


def is_union(T: type) -> bool:
    # One is for A | B, another is for Union[A, B]. Shrug.
    return get_origin(T) in (types.UnionType, Union)


def is_optional(T: type) -> bool:
    if not is_union(T):
        return False
    args = get_args(T)
    if len(args) != 2:
        return False
    T1, T2 = args
    return T1 != T2 and types.NoneType in (T1, T2)


def extract_type_from_optional(T: type) -> type:
    assert is_optional(T)
    T1, T2 = get_args(T)
    return T1 if T2 == types.NoneType else T2


def is_optional_primitive(T: type) -> bool:
    return is_optional(T) and is_primitive(extract_type_from_optional(T))


def is_numpy_array(T: type) -> bool:
    return get_origin(T) == np.ndarray


def is_pydantic_model(T: type) -> TypeGuard[Type[BaseModel]]:
    # Annoying error without try...catch: "issubclass() arg 1 must be a class"
    try:
        return issubclass(T, BaseModel)
    except TypeError:
        return False
