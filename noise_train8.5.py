# %%
import torch
from torch import nn
# import nresnet
import nvgg
from utils import NEval, NTrain
from time import ctime
import tqdm.auto as tqdm

import pickle

# %%
# import tracemalloc

# %%
from fitzpatrick17k_data import fitzpatric17k_dataloader_score_v2
from celeba_data import celeba_dataloader_score_v2
from isic2019_data import isic2019_dataloader_score_v2

# %%
import argparse

# from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train')

    parser.add_argument('-n', '--num_classes', type=int, default=114,
                        help="number of classes; used for fitzpatrick17k")
    parser.add_argument('-f', '--fair_attr', type=str, default="Young",
                        help="fairness attribute; now support: Male, Young; used for celeba")
    parser.add_argument('-y', '--y_attr', type=str, default="Attractive",
                        help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")
    parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k",
                        help="the dataset to use; now support: fitzpatrick17k, celeba, isic2019")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs')

    parser.add_argument('--backbone', type=str, default="resnet18",
                        help="backbone model; now support: resnet18, vgg11, vgg11_bn")
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 0.001)',
                        dest='weight_decay')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to log the checkpoint and training log to')
    parser.add_argument('--csv_file_name', type=str, default="fitzpatrick17k/fitzpatrick17k.csv",
                        help="CSV file position")
    parser.add_argument('--image_dir', type=str, default="fitzpatrick17k/dataset_images",
                        help="Image files directory")
    parser.add_argument('--ckpt_limit', default=0, type=int, metavar='N',
                        help='max number of checkpoints to save; default now is 0; must be non-negative, or it would be considered as 0')
    parser.add_argument('--step_size', default=100, type=int, metavar='N',
                        help='after epochs of step size, learning rate decay')
    parser.add_argument('--gamma', default=0.57, type=float, metavar='N', # 0.57^4 is about 0.1
                        help='learning rate decay by gamma*')

    # add noise_std to parser
    parser.add_argument('--dev_var', type=float, default=0.0,
                        help="noise std for fair arg")
    parser.add_argument('--ctype', type=str, default="low",
                        help="noise type for fair arg; now support: gaussian, laplace")

    parser.add_argument('--start_iter', type=int, default=1,
                    help="if you want to continue from a checkpoint, change this to the iteration number of the checkpoint")
    parser.add_argument('--type', type=str, default="train")
    # %%
    args = parser.parse_args(['--fair_attr', 'Young',
                                '--y_attr', 'Attractive',
                                '--dataset', 'fitzpatrick17k',
                                '--epochs', '200',
                                '--batch_size', '128',
                                '--csv_file_name', './fitzpatrick17k/fitzpatrick17k.csv',
                                '--image_dir', './fitzpatrick17k/dataset_images',
                                '--weight_decay', '1e-5',
                                '--lr', '1e-4',
                                '--start_iter', '1',
                                '--dev_var', '0.00',
                                '--type', 'test'
                            ])

    # %%
    model = nvgg.vgg11(num_classes=114)
    model.load_state_dict(torch.load("dewen_train.pt"))
    
    criteria = nn.CrossEntropyLoss()  

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
        
    # model.clear_noise()
    print("Using device: ", device)
    trainloader, _, testloader, _ = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, args.image_dir, args.csv_file_name, args.ctype)


    if args.type == "train":
        model_group = model, criteria, optimizer, scheduler, device, trainloader, testloader

        NTrain(model_group, epochs=args.epochs, dev_var=args.dev_var, verbose=True)  # You may modify this function in utils.py to fit Yuanbo's setting
    
    
    print("Start testing")

    NUMBER_OF_MC_RUNS = 1000
    acc_list = []
    fair_list = []
    print("Start MC runs")
    for _ in range(NUMBER_OF_MC_RUNS):
        acc, fair = NEval(model, device, testloader, args.dev_var)
        acc_list.append(acc)
        fair_list.append(fair)
        print(acc_list)

    file_path = f"acc_{args.dev_var}_{NUMBER_OF_MC_RUNS}.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(acc_list, file)
        
    file_path = f"fair_{args.dev_var}_{NUMBER_OF_MC_RUNS}.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(fair_list, file)

# %%


# %%



