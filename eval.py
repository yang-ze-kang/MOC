import torch
from models import create_model
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.datasets import Generic_Muti_Survival_Dataset
from sksurv.metrics import concordance_index_censored
import argparse
import numpy as np
import pandas as pd
import pdb
import os
from utils.file_utils import save_pkl, load_pkl
from utils.utils import get_split_loader

# Training settings
parser = argparse.ArgumentParser(
    description='Configurations for Survival Analysis on TCGA Data.')

# Dataset path
parser.add_argument('--study',     		 type=str,
                    default='LGGGBM', help='study type')
parser.add_argument('--dataset_dir',     type=str,
                    default='/mnt/sdc-1/yzk/lung/dataset', help='path to genome dataset')
parser.add_argument('--fold',     type=int, default=0)
parser.add_argument('--target_gene',     default=None)
parser.add_argument('--data_dir',   	 type=str, default='path/to/data_root_dir',
                    help='Data directory to WSI features (extracted via CLAM')

parser.add_argument('--seed', 			 type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--save_dir',     type=str, default='./results',
                    help='Results save directory (Default: ./results)')

# Model Parameters.
parser.add_argument('--weights',      type=str,
                    default='', help='path to pretrained weights')
parser.add_argument('--model_type',      type=str,
                    default='mcat', help='Type of model (Default: mcat)')
parser.add_argument('--omic_embedding_size',	type=int,
                    default=256, help='dimension of omic embedding')
parser.add_argument('--data_mode',       type=str, default=None,
                    help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=[
                    'None', 'concat', 'bilinear'], default=None, help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--drop_out',        action='store_true',
                    default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str,
                    default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str,
                    default='small', help='Network size of SNN model')

parser.add_argument('--n_classes', type=int, default=4)


# PORPOISE
parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--use_mlp', action='store_true', default=False)



def summary_survival(model, loader):
    model.eval()

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        # To device
        if isinstance(data_WSI, dict):
            for key in data_WSI:
                data_WSI[key] = data_WSI[key].to(device)
        else:
            data_WSI = data_WSI.to(device)
        for key in data_omic:
            data_omic[key] = data_omic[key].to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            outs = model(x_path=data_WSI, **data_omic)
        if isinstance(outs, torch.Tensor):
            risk = outs.detach().cpu().numpy()
        elif isinstance(outs, dict):
            risk = outs['hazards'].detach().cpu().numpy()
        else:
            assert NotImplemented

        # event_time = np.asscalar(event_time)
        event_time = event_time.item()
        # censor = np.asscalar(censor)
        censor = censor.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {
            'slide_id': np.array(slide_id),
            'risk': risk,
            'disc_label': y_disc.item(),
            'survival': event_time,
            'censorship': censor
        }})

    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index

def seed_torch(seed=7, device='cuda'):
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

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed, device)
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
    _, _, val_dataset = dataset.return_splits(from_id=False,csv_path='{}/splits_{}.csv'.format(dataset.split_path, args.fold))
    if 'sigsets' in args.data_mode:
        args.omic_sizes = dataset.omic_sizes
        print('Genomic Dimensions', args.omic_sizes)
    elif 'omic' in args.data_mode:
        args.omic_input_dim = val_dataset.genomic_features.shape[1]
        print("Genomic Dimension", args.omic_input_dim)
    else:
        args.omic_input_dim = 0
    val_loader = get_split_loader(val_dataset,mode=args.data_mode)
    model = create_model(args)
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
    model.load_state_dict(torch.load(args.weights))
    model = model.to(device)
    val_latest, cindex_latest = summary_survival(model, val_loader)
    print(f'C-Index:{cindex_latest}')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_pkl(os.path.join(args.save_dir,'val_results.pkl'), val_latest)