import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
from torchvision import datasets
import logging

from flcore.servers.serverALA import FedALA
from flcore.trainmodel.models import *


warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635
max_len=200

hidden_dim=32

def dict_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        logger.handlers.clear()
        
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.terminator = ""
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.terminator = ""
        logger.addHandler(console_handler)
    logger.info(filepath)
    
    # with open(filepath, "r") as f:
        # logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def run(args):

    time_list = []
    model_str = args.model
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    args.logger = logger

    # for i in range(args.prev, args.times):
    for i in range(1,10):
        print(f"\n============= Running hetero-model: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        dataset_train = datasets.MNIST('.data/mnist', train=True, download=True)
        dict_users = dict_iid(dataset_train, int(1/args.partial_data*args.num_clients))
        args.dict_users = dict_users

        # Generate args.model
        if model_str == "cnn":
            if args.dataset[:5] == "mnist":
                from flcore.trainmodel.mnist_models import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c
                # args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1600).to(args.device) # 1024 for 28x28
            elif args.dataset == "fmnist":
                from flcore.trainmodel.fmnist_models import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c
            elif args.dataset == "svhn":
                from flcore.trainmodel.svhn_models import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c
            elif args.dataset == "cifar10":
                from flcore.trainmodel.cifar10_models import CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5, CNN5b, CNN5c                

            if i == 0:
                args.model = CNN2().to(args.device)
            elif i == 1:
                args.model = CNN3().to(args.device)
            elif i == 2:
                args.model = CNN3b().to(args.device)
            elif i == 3:
                args.model = CNN3c().to(args.device)
            elif i == 4:
                args.model = CNN4().to(args.device)
            elif i == 5:
                args.model = CNN4b().to(args.device)
            elif i == 6:
                args.model = CNN4c().to(args.device)
            elif i == 7:
                args.model = CNN5().to(args.device)
            elif i == 8:
                args.model = CNN5b().to(args.device)
            elif i == 9:
                args.model = CNN5c().to(args.device)

            # elif args.dataset[:5] == "Cifar":
            #     args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            # else:
            #     args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)
        
        else:
            raise NotImplementedError
                            
        print(args.model)

        if args.algorithm == "FedALA":
            server = FedALA(args, i)
        else:
            raise NotImplementedError
            
        server.train()
        
        # torch.cuda.empty_cache()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="svhn")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedALA")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=10,
                        help="Rounds gap for evaluation")
   
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=100)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")
    parser.add_argument('-pd', "--partial_data", type=float, default=0.01,
                        help="amount of partial data")
    parser.add_argument("--save", type=str, default='logs')    
    
    # MNIST 60000 samples
    # 100 clients in gefl: 10 clients have an equivalent model ( 60 samples/UE) => partial_data = 0.01, client = 10 
    #  50 clients in gefl:  5 clients have an equivalent model (120 samples/UE) => partial_data = 0.01, client = 5
    #  10 clients in gefl:  1 client  has a model              (600 samples/UE) => partial_data = 0.01, client = 1 (FedAvg)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    # torch.cuda.set_device(int(args.device_id))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)