import numpy as np
import argparse
import pdb
import os
from timeit import default_timer as timer
import sys
import numpy as np
import pandas as pd
import json

# Internal Imports
from utils.datasets  import Generic_Muti_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex = []
    folds = np.arange(start, end)

    # Start 5-Fold CV Evaluation.
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(
            args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path) and (not args.overwrite):
            print("Skipping Split %d" % i)
            continue

        # Gets the Train + Val Dataset Loader.
        if args.bag_loss == 'contrast':
            train_dataset,train_val_split, val_dataset = dataset.return_splits(from_id=False,
                                                           csv_path='{}/splits_{}.csv'.format(dataset.split_path, i),contrast=True)
        else:
            train_dataset,train_val_split, val_dataset = dataset.return_splits(from_id=False,
                                                           csv_path='{}/splits_{}.csv'.format(dataset.split_path, i))
        train_dataset.set_split_id(split_id=i)
        val_dataset.set_split_id(split_id=i)

        # pdb.set_trace()
        print('training: {}, validation: {}'.format(
            len(train_dataset), len(val_dataset)))
        datasets = (train_dataset,train_val_split, val_dataset)

        # Specify the input dimension size if using genomic features.
        if 'sigsets' in args.data_mode:
            args.omic_sizes = train_dataset.omic_sizes
            print('Genomic Dimensions', args.omic_sizes)
        elif 'omic' in args.data_mode or args.data_mode == 'cluster' or args.data_mode == 'graph' or args.data_mode == 'pyramid':
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print("Genomic Dimension", args.omic_input_dim)
        else:
            args.omic_input_dim = 0

        # Run Train-Val on Survival Task.
        val_latest, cindex_latest = train(datasets, i, args)
        latest_val_cindex.append(cindex_latest)

        # Write Results for Each Split to PKL
        save_pkl(results_pkl_path, val_latest)
        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    # Finish 5-Fold CV Evaluation.
    results_latest_df = pd.DataFrame(
        {'folds': folds, 'val_cindex': latest_val_cindex})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'

    results_latest_df.to_csv(os.path.join(
        args.results_dir, save_name))


# Training settings
parser = argparse.ArgumentParser(
    description='Configurations for Survival Analysis on TCGA Data.')

# Dataset path
parser.add_argument('--study',     		 type=str,
                    default='LUAD', help='study type')
parser.add_argument('--dataset_dir',     type=str,
                    default='/mnt/sdc-1/yzk/lung/dataset', help='path to genome dataset')
parser.add_argument('--target_gene',     default=None)
parser.add_argument('--data_dir',   	 type=str, default='path/to/data_root_dir',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--alpha_surv',      type=float, default=0.0,
                    help='How much to weigh uncensored patients')
# Checkpoint + Misc. Pathing Parameters
parser.add_argument('--seed', 			 type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1,
                    help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1,
                    help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results_new',
                    help='Results directory (Default: ./results)')
parser.add_argument('--log_data',        action='store_true',
                    default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')

# Model Parameters.
parser.add_argument('--model_type',      type=str,
                    default='snn', help='Type of model (Default: snn)')
parser.add_argument('--omic_embedding_size',	type=int,
                    default=256, help='dimension of omic embedding')
parser.add_argument('--data_mode',       type=str, default=None, help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=[
                    'None', 'concat', 'bilinear','add'], default=None, help='Type of fusion. (Default: None).')
parser.add_argument('--apply_sig',		 action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--drop_out',        action='store_true',
                    default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str,
                    default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str,
                    default='small', help='Network size of SNN model')
parser.add_argument('--drop_instance', type=float,default=0)
parser.add_argument('--n_classes', type=int, default=4)


# PORPOISE
# parser.add_argument('--apply_mutsig', action='store_true', default=False)
parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--use_mlp', action='store_true', default=False)


# Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str,
                    choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1,
                    help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int,
                    default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20,
                    help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4,
                    help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['ce_surv','nll_surv','contrast'],
                    default='nll_surv', help='sloss function (default: nll)')
parser.add_argument('--label_frac',      type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', 			 type=float, default=1e-5,
                    help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                    default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-5,
                    help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true',
                    default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true',
                    default=False, help='Enable early stopping')

# CLAM-Specific Parameters
parser.add_argument('--bag_weight',      type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--testing', 	 	 action='store_true',
                    default=False, help='debugging tool')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
print("Experiment Name:", args.exp_code)

# Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'studty': args.study,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'data_mode': args.data_mode,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

dataset = Generic_Muti_Survival_Dataset(dataset_dir=args.dataset_dir,
                                        study=args.study,
                                        target_gene=args.target_gene,
                                        data_mode=args.data_mode,
                                        data_dir=args.data_dir,
                                        shuffle=True,
                                        seed=args.seed,
                                        print_info=True,
                                        patient_strat=False,
                                        n_bins=4,
                                        label_col='survival_months',
                                        ignore=[])

# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.param_code)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()


with open(args.results_dir + '/experiment_{}.json'.format(args.exp_code), 'w') as f:
    json.dump(settings, f, indent=4)
with open(args.results_dir + '/config.json'.format(args.exp_code), 'w') as f:
    json.dump(vars(args), f, indent=4)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    start = timer()
    print(args.dataset_dir)
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
