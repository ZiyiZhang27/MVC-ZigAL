import inflect
from importlib import resources
from random import choices

IE = inflect.engine()
DATA_PATH = resources.files("mvczigal.data")


class SimpleAnimals:
    def __init__(self, batch_size):
        self.data = []
        self.batch_size = batch_size
        with open(DATA_PATH.joinpath("simple_animals.txt"), "r") as f:
            for line in f:
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def sample(self):
        return choices(self.data, k=self.batch_size)


class MATE3D:
    def __init__(self, batch_size):
        self.data = []
        self.batch_size = batch_size
        with open(DATA_PATH.joinpath("MATE_3D.txt"), "r") as f:
            for line in f:
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def sample(self):
        return choices(self.data, k=self.batch_size)
