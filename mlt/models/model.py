"""class Model"""

from abc import abstractmethod, ABCMeta


class Model(metaclass=ABCMeta):
    """the base of models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model"""
