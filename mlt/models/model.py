"""class Model"""

from abc import abstractmethod, ABCMeta

from mlt.utils import weight


class Model(metaclass=ABCMeta):
    """the base of models"""

    _w = None

    def __init__(self, w):
        self._w = w

    @abstractmethod
    def fit(self, X, y):
        """Fit model"""

    @abstractmethod
    def _decision_function(self, X):
        """Decision function"""

    def _init_weight(self, n, func='zeros'):  # todo: change func to func_name
        """initial weight"""
        if self._w is None:
            self._w = weight.init(func, n)

    def predict(self, X):
        """predict from input data"""
        return self._decision_function(X)
