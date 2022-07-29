###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from load_data import LoadASL_imbal_Data, LoadGSL_imbal_Data, LoadIrish_SL_imbal_Data,\
                      LoadChinese_SL_imbal_Data, LoadFashion_MNIST_Data,LoadIndian_MNIST_Data
from cnn_model import ConvNN
import copy
import time
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
plt.style.use("seaborn-darkgrid")

from gradcam import run_grad_cam
from datetime import datetime
import pathlib



def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def load_backbone(model, pretrained_dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_dir)
    # 1. filter out unnecessary keys, only load backbone
    back_bone = {k: v for k, v in pretrained_dict.items() if "fc" not in k}
    #back_bone = {k: v for k, v in pretrained_dict.items() }
    # 2. overwrite entries in the existing state dict
    model_dict.update(back_bone) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    
    return model


def get_all_preds(model, loader, device):
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    for batch in loader:
        x_data, labels = batch
        img_shape = int(x_data[-1].shape[0]**0.5)
        x_data= x_data.view(-1, 1, img_shape, img_shape)
        
        x_data = x_data.to(device)
        labels = labels.to(device)

        preds = model.forward(x_data).argmax(dim=1)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels)
            ,dim=0
        )
    return all_preds.cpu().detach().numpy(),all_labels.cpu().detach().numpy()


def get_model(model_name, num_classes=25):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'cnn', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'cnn':  # Use this model for debugging
        cnn_model = ConvNN(out_size=num_classes,img_rows = 96,img_cols=96 )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
        cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'res18':
        cnn_model = models.resnet18(pretrained=False)
        cnn_model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        #origin (fc) is: Linear(in_features=512, out_features=26, bias=True)
        fc_inputs = cnn_model.fc.in_features
        cnn_model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes))
            
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    print("cnn_model is: ",cnn_model)
    return cnn_model

def get_freq():
    ASL_freq={"E":0.112,"A":0.085,"R":0.076,"I":0.075,"O":0.072,"T":0.070,"N":0.066,"S":0.057,"L":0.055,
          "C":0.045,"U":0.036,"D":0.034,"P":0.032,"M":0.030,"H":0.030,"G":0.025,"B":0.021,"F":0.018,
          "Y":0.018,"W":0.013,"K":0.011,"V":0.010,"X":0.003,"Z":0,"J":0,"Q":0.020}
    GSL_freq={"A":0.056,"B":0.020,"C":0.032,"D":0.050,"E":0.169,"F":0.015,"G":0.030,"H":0.050,"I":0.080,
          "J":0,"K":0.013,"L":0.036,"M":0.026,"N":0.105,"O":0.022,"P":0.007,"Q":0.0002,"R":0.069,
         "S":0.064,"T":0.058,"U":0.038,"V":0.008,"W":0.018,"X":0.001,"Y":0.0001,"Z":0}
    
    Alphabet_list=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    ## change ASL_freq or GSL_freq here
    Freq = GSL_freq

    #new_freq = defaultdict(float)
    for i in Alphabet_list:
        index = Alphabet_list.index(i)
        Freq[str(index)] = Freq[i]
        del Freq[i]  
    alpha_freq=[i for i in Freq.values() if i !=0]
    
    return alpha_freq

def train_model(pretrained_model, dataset, save_path, train_set, test_set, model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # Load the datasets
    #train_dataset, val_dataset = get_train_validation_set(data_dir, validation_size=5000)
    #test_set = get_test_set(data_dir)

    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    loss_function = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    x_train, y_train = next(iter(train_loader))
    
    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    loss_train_list=[]
    accuracy_test_list = []
    accuracy_train_list = []
    start = time.time()
    
    wandb.init(
    # Set the project where this run will be logged
    project=dataset,
    # Track hyperparameters and run metadata
    config={
    "architecture": model_name,
    "dataset": dataset,
    "pretrained_model":pretrained_model,
    "epochs": epochs,
    })
    
    for epoch in range(epochs):
        data_train_iter = iter(train_loader)
        step =0
        end = time.time()
        print("epoch :", epoch)
        print("Program has running: ", (end-start)/60,"min")
        epoch_loss = 0
        model.train()
        while step < len(train_loader):
            step +=1
            ## Step 1: prepare data
            x_train, y_train = next(data_train_iter)
            #print("x_train.shape: ",x_train.shape)
            x_train = x_train.to(device)
            img_shape = int(x_train[-1].shape[0]**0.5)
            x_train= x_train.view(-1, 1, img_shape, img_shape)
            y_train = y_train.to(device)
            
            ## Step 2: Run the model on the input data
            preds = model(x_train)
            ## Step 3: Calculate the loss
            loss = loss_function(preds, y_train)
            epoch_loss += loss
            ## Step 4: Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            ## Step 5: Update the parameters
            optimizer.step()
        avg_loss = epoch_loss/len(train_loader)
        loss_train_list.append(avg_loss)
        scheduler.step()
        
        accuracy_train = evaluate_model(model, train_loader, device)
        accuracy_test = evaluate_model(model, test_loader, device)
        accuracy_test_list.append(accuracy_test)
        accuracy_train_list.append(accuracy_train)
        all_preds,all_labels = get_all_preds(model, test_loader, device)
        
        wandb.log({"loss": loss,"accuracy_test": accuracy_test,"accuracy_train": accuracy_train,}
                    )
        #calculate the GradCAM
        run_grad_cam(dataset = dataset, model = model, save_path = save_path, iterr=epoch)
        
        if epoch%10==0:
            figure, ax1 = plt.subplots(1,1)
            ax2 = ax1.twinx()
            con_mat = confusion_matrix(all_labels, all_preds)
            accuracy_con_mat =con_mat.diagonal()/con_mat.sum(axis=1)
            miss_Alphabet_list=["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
 
        if epoch==0:
            best_model =copy.deepcopy(model)
        elif accuracy_test > accuracy_test_list[-2]:
            best_acc = accuracy_test
            best_model = copy.deepcopy(model)
            print("best model's epoch is: ", epoch)
        print("train_accuracy: ",accuracy_train)
        print("test_accuracy: ",accuracy_test)
    
    test_accuracy  = evaluate_model(best_model, test_loader, device)
    print("best_acc:", best_acc)
    
    # save the final model
    #torch.save(best_model.state_dict(), checkpoint_name+str(model_name)+dataset+"_"+str(pretrained_model)+'_96.pkl')
    plt.plot(accuracy_train,label="training")
    plt.plot(accuracy_test_list,label="test")
    plt.legend()
    plt.show()
    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    data_train_iter = iter(data_loader)
    step = 0
    epoch_accuracy=0
    while step < len(data_train_iter):
        step+=1
        x_data, y_data = next(data_train_iter)
        img_shape = int(x_data[-1].shape[0]**0.5)
        x_data= x_data.view(-1, 1, img_shape, img_shape)
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        
        #y_data = dense_to_one_hot(y_data, 10)
        
        out = model.forward(x_data)
        step_accuracy = (out.argmax(axis=1) == y_data).type(torch.FloatTensor).mean()

        epoch_accuracy += step_accuracy

    accuracy = epoch_accuracy/len(data_train_iter)
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy




def main(pretrained_model, model_name, lr, batch_size, epochs, data_dir, seed, dataset):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    print("device:",device)
    set_seed(seed)
    
    if dataset == "MNIST":
        DataLoader = LoadData(args.val_size)
    elif args.dataset == "ASL_MNIST_imbal":
        DataLoader = LoadASL_imbal_Data()
    elif args.dataset == "GSL_MNIST_imbal":
        DataLoader = LoadGSL_imbal_Data()
    elif args.dataset == "Irish_SL_MNIST_imbal":
        DataLoader = LoadIrish_SL_imbal_Data()
    elif args.dataset == "Chinese_SL_MNIST_imbal":
        DataLoader = LoadChinese_SL_imbal_Data()
    elif args.dataset == "Fashion_MNIST":
        DataLoader = LoadFashion_MNIST_Data()
    elif args.dataset == "Indian_SL_MNIST_imbal":
        DataLoader = LoadIndian_MNIST_Data()
    train_dataset, test_dataset = DataLoader.download_dataset()
    print("Dataset is:","----------",args.dataset,"---------")
    
    # Magic
    #wandb.watch(model, log_freq=100)
    if args.dataset == "Fashion_MNIST":
        num_class = 10
    else:
        num_class = 25
    model = get_model(model_name, num_classes=num_class).to(device)
    checkpoint_name ="./best_models_96/"
    if pretrained_model is not False:
        if not os.path.isfile(checkpoint_name+str(model_name)+pretrained_model+"_False_96.pkl"):
            print("Do not found pretrained model....")
        pretrained_dir=checkpoint_name+str(model_name)+pretrained_model+"_False_96.pkl"
        model =load_backbone(model=model, pretrained_dir=pretrained_dir)
        ## Freeze backbone(all conv layers)
        """
        num=0
        for child in model.children():
            num += 1
            if num<=8:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("The following layer is trainable is: ",child)
        """
        print("******** loading pretrained model *********")
        print("pretrained_dir: ",pretrained_dir)
    
    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    trunc_str = f"{args.dataset}_trunc_{args.pretrained_model}"
    save_path = os.path.join('GradCam', time_str + trunc_str)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    #if not os.path.isfile(checkpoint_name+str(model_name)+dataset+"False.pkl"):
    #    print("Do not found model, begin training....")
    best_model = train_model(pretrained_model, dataset,save_path, train_dataset, test_dataset, model, lr, batch_size, epochs, data_dir, checkpoint_name, device)
    #else:
    #    print("found model, begin evaluation....")
    #test_results = test_model(model, batch_size, data_dir, device, 42)
    #######################
    # END OF YOUR CODE    #
    #######################





if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')
    parser.add_argument("--dataset",default="ASL_MNIST_imbal",type=str, 
                        help="Training on MNIST dataset or ASL MNIST dataset",)
    parser.add_argument("--pretrained_model",default=False, metavar="SD",
                        help="Choose if to load pretrained model (default: False)",)
    # Log in to your W&B account
    wandb.login()
    
    args = parser.parse_args()
    kwargs = vars(args)
    model_name = kwargs["model_name"]
    print("model_name",model_name)
    main(**kwargs)
    
    #exmaple running code:
    #python main_cnn.py --model_name res18 --epochs 50 --dataset Chinese_SL_MNIST_imbal