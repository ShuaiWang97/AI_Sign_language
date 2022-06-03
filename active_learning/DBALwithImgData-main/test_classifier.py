import torch
import numpy as np
import argparse
import os
from modAL.models import ActiveLearner
from sklearn.metrics import confusion_matrix
from torchvision import models
import torch.nn as nn
from skorch import NeuralNetClassifier
from acquisition_functions import uniform,var_ratios

from load_data import LoadChinese_SL_imbal_Data, LoadGSL_ASL_imbal_Data, LoadGSL_imbal_Data, LoadIrish_SL_imbal_Data, LoadData, LoadASLData, \
    LoadBSLData, LoadASL_imbal_Data 
    
def load_CNN_model(args, device):
    """Load CNN model"""
    print("args.dataset is: ", args.dataset)
    label_num = 25
    
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, label_num))
    ######
    cnn_classifier = NeuralNetClassifier(
        module = model,
        lr = 1e-3,
        batch_size=args.batch_size,
        max_epochs=50,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device,
    )
    return cnn_classifier
    
def test_active_learning(args, device, datasets: dict):
    """ test the accuracy for loaded cnn model"""

    dataset=args.dataset,
    X_test=datasets["X_test"]
    y_test=datasets["y_test"]
    X_init=datasets["X_init"]
    y_init=datasets["y_init"]
    cnn_classifier=load_CNN_model(args, device)
    pretrain=args.pretrain
    query_strategy=var_ratios

    cnn_classifier.initialize()  # This is important!
    cnn_classifier.load_params(f_params=args.model_path)
    perf_hist = [cnn_classifier.score(X_test, y_test)]
    print("test score is: ",perf_hist)
    y_pred = cnn_classifier.predict(X_test)
    con_mat = confusion_matrix(y_test, y_pred)
    
    #fname="con_mat/"+dataset+"_pre_"+pretrain+"_con_mat"+str(query_strategy)[10:14]+".csv"
    #print(dataset.type)
    np.savetxt("con_mat/con_mat"+str(dataset)+pretrain+str(query_strategy)[10:14]+".csv", con_mat, delimiter=',')

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ISL_MNIST_imbal",
        help="Training on MNIST dataset or ASL MNIST dataset",
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="path to the model to test", 
        default='') 
    
    parser.add_argument(
        "--pretrain", 
        type=str, 
        help="if pretrain", 
        default='No') 
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )


    args = parser.parse_args()
    torch.manual_seed(369)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()
    # change LoadData of MNIST to LoadSignData SignMNIST
    if args.dataset == "MNIST":
        DataLoader = LoadData(val_size=10)
    elif args.dataset == "Chinese_SL_MNIST_imbal": 
        DataLoader = LoadChinese_SL_imbal_Data(val_size=10)
    elif args.dataset == "ASL_MNIST_imbal":
        DataLoader = LoadASL_imbal_Data(val_size=10)
    elif args.dataset == "Irish_SL_MNIST_imbal":
        DataLoader = LoadIrish_SL_imbal_Data(val_size=10)
    elif args.dataset == "GSL_MNIST_imbal":
        DataLoader = LoadGSL_imbal_Data(val_size = 10,less_data=False)
    else:
        print("!!! This dataset is not included !!!")
    (
        datasets["X_init"],
        datasets["y_init"],
        datasets["X_val"],
        datasets["y_val"],
        datasets["X_pool"],
        datasets["y_pool"],
        datasets["X_test"],
        datasets["y_test"],
    ) = DataLoader.load_all()


    results = test_active_learning(args, device, datasets)


if __name__ == "__main__":
    main()