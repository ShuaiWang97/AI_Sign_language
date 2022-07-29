import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Dataset


class LoadASL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100):
        self.mnist_train, self.mnist_test = self.download_dataset()
    
    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        #train_csv= ["D:/AI_thesis/sign_dataset/sign_mnist_train.csv",
        #         ]
        #test_csv = ["D:/AI_thesis/sign_dataset/sign_mnist_test.csv",
        #         ]
                 
        train_csv= ["D:/AI_thesis/sign_dataset/full_size_dataset/sign_ASL_train_96.csv",
                 ]
        test_csv = ["D:/AI_thesis/sign_dataset/full_size_dataset/sign_ASL_test_96.csv",
                 ]
        
        mnist_train = MyDataset(train_csv)
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test



class LoadGSL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100,less_data=False):
        # change from 10000 to 5000
        self.mnist_train, self.mnist_test = self.download_dataset()

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["D:/AI_thesis/sign_dataset/full_size_dataset/sign_GSL_train_96.csv",
                 ]
        #train_csv=["D:/AI_thesis/sign_dataset/full_size_dataset/sampled_sign_GSL_train.csv",
        #         ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["D:/AI_thesis/sign_dataset/full_size_dataset/sign_GSL_test_96.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test
        
        
class LoadChinese_SL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100,less_data=False):
        # change from 10000 to 5000
        self.mnist_train, self.mnist_test = self.download_dataset()

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv=["D:/AI_thesis/Sign_dataset/full_size_dataset/sign_CSL_train_96.csv"]
        #train_csv=["D:/AI_thesis/sign_dataset/sign_Chinese_SL_train.csv"]
        mnist_train = MyDataset(train_csv)

        test_csv = ["D:/AI_thesis/Sign_dataset/full_size_dataset/sign_CSL_test_96.csv"]
        #test_csv = ["D:/AI_thesis/Sign_dataset/sign_Chinese_SL_test.csv"]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

class LoadIrish_SL_imbal_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100,less_data=False):
        # change from 10000 to 5000
        self.mnist_train, self.mnist_test = self.download_dataset()

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv = ["D:/AI_thesis/Sign_dataset/full_size_dataset/sign_ISL_train_96.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["D:/AI_thesis/Sign_dataset/full_size_dataset/sign_ISL_test_96.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test
        

class LoadFashion_MNIST_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100,less_data=False):
        # change from 10000 to 5000
        self.mnist_train, self.mnist_test = self.download_dataset()

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv = ["./fashion_data/fashion-mnist_train.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["./fashion_data/fashion-mnist_test.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test


class LoadIndian_MNIST_Data:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 100,less_data=False):
        self.mnist_train, self.mnist_test = self.download_dataset()

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""

        train_csv = ["D:/AI_thesis/Sign_dataset/full_size_dataset/sign_Indian_SL_train_96.csv",
                 ]
        mnist_train = MyDataset(train_csv)

        test_csv = ["D:/AI_thesis/Sign_dataset/full_size_dataset/sign_Indian_SL_test_96.csv",
                  ]
        mnist_test = MyDataset(test_csv)
        return mnist_train, mnist_test

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

        x=df.iloc[:,1:].values
        y=df.iloc[:,0].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y)
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        
        return x/255.0,self.y_train[idx]