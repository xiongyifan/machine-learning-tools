"""class Model"""

from abc import abstractmethod, ABCMeta


class Model(metaclass=ABCMeta):
    """the base of models"""

    _w = None

    @abstractmethod
    def fit(self, X, y, w):
        """Fit model"""

    def _init_weight(self, w, func, shape):
        """initial or set weight"""
        if w is None:
            self._w = func(shape)
        else:
            self._w = w

    @abstractmethod
    def _decision_function(self, X):
        """Decision function"""
