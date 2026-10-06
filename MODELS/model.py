from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def valid(self):
        pass
