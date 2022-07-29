import os
import time
import argparse
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from scipy.ndimage.filters import gaussian_filter1d
from torchvision import models

from load_data import LoadChinese_SL_imbal_Data, LoadGSL_imbal_Data, LoadIrish_SL_imbal_Data, LoadData, LoadASL_imbal_Data 
from cnn_model import ConvNN
from active_learning import select_acq_function, active_learning_procedure
import pickle
from datetime import datetime
import pathlib


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def load_pre_model(model, pretrained_dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_dir)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items()  if "fc" not in k}
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    
    return model

def load_CNN_model(args, device):
    """Load new model each time for different acqusition function
    each experiments"""
    print("args.dataset is: ", args.dataset)
    if args.dataset == "MNIST":
        label_num = 10
    elif args.dataset == "ASL_MNIST" or args.dataset == "ASL_MNIST_imbal" or args.dataset == "unwei_ASL_MNIST" or args.dataset == "ASL_GSL_MNIST_imbal":
        label_num = 25
    elif args.dataset == "Irish_SL_MNIST_imbal" or args.dataset == "GSL_MNIST_imbal" or args.dataset == "GSL_MNIST_imbal_less" or args.dataset=="Chinese_SL_MNIST_imbal":
        label_num = 25
    elif args.dataset == "BSL_MNIST":
        label_num = 38
    print("label_num is: ", label_num)
    
    if args.backbone=="cnn":
      model = ConvNN(out_size=label_num,img_rows=96,img_cols=96).to(device)
    elif args.backbone=="res18":
      model = models.resnet18(pretrained=False)
      model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
      fc_inputs = model.fc.in_features
      model.fc = nn.Sequential(
      nn.Linear(fc_inputs, 128),
      nn.ReLU(),
      nn.Dropout(0.4),
      nn.Linear(128, label_num))
    print("args.backbone is: ",args.backbone)

    if args.pretrained_model != "False":
        if args.pretrained_model == "GSL_MNIST_imbal":
            pretrained_dir="cnn_models/pretrained_96/res18GSL_MNIST_imbal_False_96.pkl"
        if args.pretrained_model == "Chinese_SL_MNIST_imbal":
            pretrained_dir = "cnn_models/pretrained_96/res18Chinese_SL_MNIST_imbal_False_96.pkl"
        if args.pretrained_model == "Irish_SL_MNIST_imbal":
            pretrained_dir = "cnn_models/pretrained_96/res18Irish_SL_MNIST_imbal_False_96.pkl"
        if args.pretrained_model == "ASL_MNIST_imbal":
            pretrained_dir = "cnn_models/pretrained_96/res18ASL_MNIST_imbal_False_96.pkl"
        if args.pretrained_model == "Fashion_MNIST":
            pretrained_dir = "cnn_models/pretrained/resnet18Fashion_MNIST_False_96.pkl"
        model =load_pre_model(model=model, pretrained_dir=pretrained_dir)
        print("******** loading pretrained model *********")
        print("pretrained_dir: ",pretrained_dir)
    ######
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device,
    )
    return cnn_classifier


def save_as_npy(data: np.ndarray, folder: str, name: str, sub_name: str):
    """Save result as npy file
    
    Attributes:
        data: np array to be saved as npy file,
        folder: result folder name,
        name: npy filename
    """
    file_name = os.path.join(folder, name + sub_name + ".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")


def plot_results(args, data: dict, name: str):
    """Plot results histogram using matplotlib"""
    sns.set()
    for key in data.keys():
        # data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.legend()
    plt.show()
    plt.savefig( name +"_"+ args.backbone+".png")


def print_elapsed_time(start_time: float, exp: int, acq_func: str):
    """Print elapsed time for each experiment of acquiring

    Attributes:
        start_time: Starting time (in time.time()),
        exp: Experiment iteration
        acq_func: Name of acquisition function
    """
    elp = time.time() - start_time
    print(
        f"********** Experiment {exp} ({acq_func}): {int(elp // 3600)}:{int(elp % 3600 // 60)}:{int(elp % 60)} **********"
    )

#def load_pre_train_cnn(pre_dir= pre_dir ):
#    pretrained_dict 


def train_active_learning(args, device, datasets: dict) -> dict:
    """Start training process

    Attributes:
        args: Argparse input,
        estimator: Loaded model, e.g. CNN classifier,
        device: Cpu or gpu,
        datasets: Dataset dict that consists of all datasets,
    """
    acq_functions, acq_functions_str = select_acq_function(args.acq_func)
    results = dict()
    if args.determ:
        state_loop = [True, False]  # dropout VS non-dropout
    else:
        state_loop = [True]  # run dropout only

    print("state_loop: ", state_loop)
    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    trunc_str = f"{args.dataset}_trunc_{args.pretrained_model}"
    save_path = os.path.join('GradCam', time_str + trunc_str)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for state in state_loop:
        for i, acq_func in enumerate(acq_functions):
            
            all_hist = []
            test_scores = []
            all_con_mat=[]
            acq_func_name = str(acq_func).split(" ")[1] + "-MC_dropout=" + str(state)
            print(f"\n---------- Start {acq_func_name} training! ----------")
            for e in range(args.experiments):
                set_seed(e)
                print("random seed is:",e)
                start_time = time.time()
                estimator = load_CNN_model(args, device)
                print(
                    f"********** Experiment Iterations: {e + 1}/{args.experiments} **********"
                )
                con_max_hist, training_hist, test_score = active_learning_procedure(pretrain=args.pretrained_model,
                                                                      dataset=args.dataset,
                                                                      save_path = save_path,
                                                                      backbone = args.backbone,
                                                                      query_strategy=acq_func,
                                                                      X_val=datasets["X_val"],
                                                                      y_val=datasets["y_val"],
                                                                      X_test=datasets["X_test"],
                                                                      y_test=datasets["y_test"],
                                                                      X_pool=datasets["X_pool"],
                                                                      y_pool=datasets["y_pool"],
                                                                      X_init=datasets["X_init"],
                                                                      y_init=datasets["y_init"],
                                                                      estimator=estimator,
                                                                      T=args.dropout_iter,
                                                                      n_query=args.query,
                                                                      training=state
                                                                      )
                all_hist.append(training_hist)
                print("training_hist is: ",training_hist)
                test_scores.append(test_score)
                all_con_mat.append(con_max_hist)
                print_elapsed_time(start_time, e + 1, acq_func_name)
            avg_hist = np.average(np.array(all_hist), axis=0)
            avg_test = sum(test_scores) / len(test_scores)
            print(f"Average Test score for {acq_func_name}: {avg_test}")
            results[acq_func_name] = avg_hist
            save_as_npy(data=all_con_mat, folder="con_mat" ,name="con_mat_"+args.dataset+str(acq_func)[10:14]+args.backbone, sub_name="pre_train"+args.pretrained_model )
            save_as_npy(data=all_hist, folder=args.result_dir ,name=args.dataset+str(acq_func)[10:14]+args.backbone, sub_name="pre_train"+args.pretrained_model )
            
    print("--------------- Done Training! ---------------")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ISL_MNIST_imbal",
        help="Training on MNIST dataset or ASL MNIST dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="EP",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=369, metavar="S", help="random seed (default: 369)"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=3,
        metavar="E",
        help="number of experiments (default: 3)",
    )
    parser.add_argument(
        "--dropout_iter",
        type=int,
        default=150,
        metavar="T",
        help="dropout iterations,T (default: 100)",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=10,
        metavar="Q",
        help="number of query (default: 10)",
    )
    parser.add_argument(
        "--acq_func",
        type=int,
        default=0,
        metavar="AF",
        help="acqusition functions: 0-all, 1-uniform, 2-max_entropy, \
                            3-bald, 4-var_ratios, 5-mean_std (default: 0)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        metavar="V",
        help="validation set size (default: 100)",
    )
    parser.add_argument(
        "--determ",
        action="store_true",
        help="Compare with deterministic models (default: False)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_npy",
        metavar="SD",
        help="Save npy file in this folder (default: result_npy)",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="False",
        metavar="SD",
        help="Choose if to load pretrained model (default: False)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="cnn",
        metavar="SD",
        help="Choose what backbone to use (default: cnn)",
    )

    args = parser.parse_args()
    print(args)
    #torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()
    # change LoadData of MNIST to LoadSignData SignMNIST
    if args.dataset == "MNIST":
        DataLoader = LoadData(args.val_size)
    elif args.dataset == "Chinese_SL_MNIST_imbal": 
        DataLoader = LoadChinese_SL_imbal_Data(val_size=10)
    elif args.dataset == "ASL_MNIST_imbal":
        DataLoader = LoadASL_imbal_Data(args.val_size)
    elif args.dataset == "Irish_SL_MNIST_imbal":
        DataLoader = LoadIrish_SL_imbal_Data(args.val_size)
    elif args.dataset == "GSL_MNIST_imbal":
        DataLoader = LoadGSL_imbal_Data(val_size = args.val_size,less_data=False)
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

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    results = train_active_learning(args, device, datasets)
    plot_results(args=args, data=results, name=args.dataset)


if __name__ == "__main__":
    main()
