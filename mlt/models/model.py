"""class Model"""

from abc import abstractmethod, ABCMeta

from mlt.utils import weight


class Model(metaclass=ABCMeta):
    """the base of models"""

    _w = None

    @abstractmethod
    def fit(self, X, y, w=None, w_init_func=weight.init_zeros):
        """Fit model"""

    def _init_weight(self, w, shape, func=weight.init_zeros):
        """initial or set weight"""
        if w is None:
            self._w = func(shape)
        else:
            self._w = w

    @abstractmethod
    def _decision_function(self, X):
        """Decision function"""
