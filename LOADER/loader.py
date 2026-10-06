from abc import ABC, abstractmethod


class loader(ABC):
    @abstractmethod
    @staticmethod
    def load_data(path):
        pass
