import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset


class LoadISL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        # change from 10000 to 5000
        self.train_size = 5000
        self.val_size = val_size
        self.pool_size = 28750 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/sampled_sign_ISL_train.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_ISL_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(26):
            #if i == 9:
            #    continue
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

class LoadGSL_ASL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        # change from 10000 to 5000
        self.train_size = 5000
        self.val_size = val_size
        self.pool_size = 48728 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch does not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/sampled_ASL_GSL.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_GSL_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(25):
            if i == 9:
                continue
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

class LoadGSL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100,less_data=False):
        # change from 10000 to 5000
        self.train_size = 1000
        if less_data == True:
            self.train_size =16000
        self.val_size = val_size
        self.pool_size = 18985 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/sampled_sign_GSL_train.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_GSL_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(26):
            #if i == 9:
            #    continue
            #some characters have low frequency
            if len(np.where(self.y_train_All == i)[0])>2:
                idx = np.random.choice(
                    np.where(self.y_train_All == i)[0], size=2, replace=False
                )
                initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

class LoadGSL_imbal_Data_less:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        # change from 10000 to 5000
        self.train_size = 500
        self.val_size = val_size
        self.pool_size = 3797 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/sampled_sign_GSL_train_twe_per.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_GSL_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(26):
            #if i == 9:
            #    continue
            #some characters have low frequency
            if len(np.where(self.y_train_All == i)[0])>2:
                idx = np.random.choice(
                    np.where(self.y_train_All == i)[0], size=2, replace=False
                )
                initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )
        
class LoadData:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        self.train_size = 10000
        self.val_size = val_size
        self.pool_size = 60000 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data

    def check_MNIST_folder(self) -> bool:
        """Check whether MNIST folder exists, skip download if existed"""
        if os.path.exists("MNIST/"):
            return False
        return True

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        download = self.check_MNIST_folder()
        mnist_train = MNIST(".", train=True, download=download, transform=transform)
        mnist_test = MNIST(".", train=False, download=download, transform=transform)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=10000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(10):
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        y_init = self.y_train_All[initial_idx]
        print(f"Initial training data points: {X_init.shape[0]}")
        print(f"Data distribution for each class: {np.bincount(y_init)}")
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

class LoadASLData:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        # change from 10000 to 5000
        self.train_size = 5000
        self.val_size = val_size
        self.pool_size = 27455 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=[ "./sign_dataset/unwei_sampled_sign_mnist_train_full.csv"
        #"./sign_dataset/sign_mnist_train.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_mnist_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(25):
            if i == 9:
                continue
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )
        
class LoadASL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        # change from 10000 to 5000
        self.train_size = 5000
        self.val_size = val_size
        self.pool_size = 27455 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/sampled_sign_mnist_train_full.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_mnist_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(25):
            if i == 9:
                continue
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )
        
class LoadASL_sample_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        # change from 10000 to 5000
        self.train_size = 5000
        self.val_size = val_size
        self.pool_size = 27455 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/unwei_sampled_sign_mnist_train_full.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_mnist_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=5000, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(25):
            if i == 9:
                continue
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

class LoadBSLData:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        self.train_size = 5000
        self.val_size = val_size
        self.pool_size = 11021 - self.train_size - self.val_size
        self.mnist_train, self.mnist_test = self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data


    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["./sign_dataset/sign_BSL_train.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./sign_dataset/sign_BSL_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [self.train_size, self.val_size, self.pool_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        #test size 7172
        test_loader = DataLoader(
            dataset=self.mnist_test, batch_size=1500, shuffle=True
        )
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        X_test = np.reshape(X_test, (-1,1,28,28))
        X_val = np.reshape(X_val, (-1,1,28,28))
        X_pool = np.reshape(X_pool, (-1,1,28,28))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(38):
            print("i : ",i)
            if i ==9:
                continue
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=2, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        X_init = np.reshape(X_init, (-1,1,28,28))
        y_init = self.y_train_All[initial_idx]
        print(f"Initial training data points: {X_init.shape[0]}")
        print(f"Data distribution for each class: {np.bincount(y_init)}")
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )

class MyDataset(Dataset):
 
    def __init__(self, file_name, binary=False): 
      ## Read data from csv file and make it to Dataset
        
        self.transform =transforms.Compose([
                                       transforms.ToTensor()])

        df = pd.concat([pd.read_csv(i) for i in file_name])
        length = [len(pd.read_csv(i)) for i in file_name]

        #make label binary for different SL
        if binary:
          for index, lenn in enumerate(length):
            if index ==0:
              a = 0
            else:
              a = length[index-1]
            df.iloc[a:a+lenn].label =index 

        x=df.iloc[:,1:785].values
        y=df.iloc[:,0].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y)
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        
        return x/255.0,self.y_train[idx]