from pydantic import BaseModel
from io import BytesIO

import numpy as np
from numpy.typing import NDArray

from hdf5_serialiser import HDF5Dataclass


class Base(HDF5Dataclass):
    name: str


class DataClass(HDF5Dataclass, eq=False):
    data: NDArray[np.float_]


class Pyd(BaseModel):
    z: int


class Small(HDF5Dataclass):
    x: int | None
    y: str


class Big(HDF5Dataclass):
    a: dict[str, dict[str, int]]
    s: Small
    d: DataClass
    p: Pyd


s = Small(x=None, y="three")
d = DataClass(data=np.random.rand(2, 3))
b = Big(a={"k1": {"k11": 123}, "k2": {"k21": 456}}, s=s, d=d, p=Pyd(z=123))

buff = BytesIO()
b.to_hdf5(buff)
buff.seek(0)
x: Big = Big.from_hdf5(buff)
