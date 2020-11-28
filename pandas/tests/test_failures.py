import numpy as np

from pandas._libs.testing import assert_almost_equal
from pandas.core.dtypes.common import is_array_like, is_scalar


class MockNumpyLikeArray:
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy
    The key is that it is not actually a numpy array so
    ``util.is_array(mock_numpy_like_array_instance)`` returns ``False``. Other
    important properties are that the class defines a :meth:`__iter__` method
    (so that ``isinstance(abc.Iterable)`` returns ``True``) and has a
    :meth:`ndim` property which can be used as a check for whether it is a
    scalar or not.
    """

    def __init__(self, values):
        self._values = values

    def __iter__(self):
        iter_values = iter(self._values)

        def it_outer():
            for element in iter_values:
                yield element

        return it_outer()

    @property
    def ndim(self):
        return self._values.ndim


def test_assert_almost_equal():
    eg = MockNumpyLikeArray(np.array(2))
    assert_almost_equal(eg, eg)


def test_correctly_identify_numpy_like_array_as_array():
    # happy
    assert is_array_like(np.array([2, 3]))
    # fails
    assert is_array_like(MockNumpyLikeArray(np.array([2, 3])))
