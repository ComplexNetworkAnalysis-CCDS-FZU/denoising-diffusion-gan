import scipy.io as sio
from sympy import N
import torch.utils.data as data
import numpy as np


class ACMMatrixDataset(data.Dataset):
    def __init__(
        self, matfile: str, name="", require_number: int = 100, transform=None
    ) -> None:
        super().__init__()

        self.root = matfile
        self.name = name
        self.require_number = require_number
        self.transform = transform

        self.mat = sio.loadmat(self.root)["sorted_M"]

    def __len__(self):
        return self.require_number

    def __getitem__(self, index):
        if index > self.require_number:
            raise IndexError(f"index out of range {index} >= {self.require_number}")
        mat = np.copy(self.mat)
        if self.transform is not None:
            mat = self.transform(mat)

        return mat


if __name__ == "__main__":
    path = "./datasets/ACM.mat/matrix.mat"

    dataset = ACMMatrixDataset(path, "ACM", 100)

    print(dataset[1])
