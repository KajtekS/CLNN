from abc import ABC, abstractmethod

class model(ABC):
    @abstractmethod
    def train():
        pass
    @abstractmethod
    def test():
        pass
    @abstractmethod
    def valid():
        pass