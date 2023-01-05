# HDF5 Dataclass

This module provides an implementation of a base class for hdf5-(de)serialisable dataclasses. It supports the following members of dataclasses:

- primitives: `int`, `float`, `str`
- **Pydantic** models -- which are simply converted into json
- **numpy** arrays
- dictionaries of primitive types to keys of any of the above
- optionals of the above

## Dependencies

The project relies on **h5py** package.

# Usage

Suppose we have an existing dataclass definition:

```python
@dataclass(eq=False)
class A:
    a: int
    b: str
    c: np.array
```

In order to add serialisation mechanism to it, you must change the definition above
above to the following:

```python
from hdf5_dataclass import HDF5Dataclass

class A(HDF5Dataclass, eq=False):
    a: int
    b: str
    c: np.array
```

class A _is_ and _behaves_ like a regular dataclass, but it adds two methods:

- instance method: `to_hdf5(output: FileType)`
- class method: `from_hdf5(input: FileType)`

Now in order to use the serialisation, you can do the following:

```python

a = A(a=123, b="abc", c=np.random.rand(10))

with file("f.hdf5", "wb") as output:
    a.to_hdf5(output)

...

with file("f.hdf5", "rb") as input:
    aa = A.from_hdf5(input)
```
