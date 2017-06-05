"""class Model"""

from abc import abstractmethod, ABCMeta


class Model(metaclass=ABCMeta):
    """the base of models"""

    w = None

    @abstractmethod
    def fit(self, X, y, w):
        """Fit model"""

    def _init_weight(self, w, func, shape):
        """initial or set weight"""
        if w is None:
            self.w = func(shape)
        else:
            self.w = w

    @abstractmethod
    def _decision_function(self, X):
        """Decision function"""
