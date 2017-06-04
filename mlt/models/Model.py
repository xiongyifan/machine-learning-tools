from abc import abstractmethod


class Model:

    @abstractmethod
    def fit(self, X, Y):
        pass