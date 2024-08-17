import os
import torch
import pickle
from sksurv.metrics import concordance_index_censored
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Late Fusion Configurations Survival Analysis on TCGA Data.')
parser.add_argument('--wsi_dir', type=str, default='', help='input the histological results dir')
parser.add_argument('--rna_dir', type=str, default='', help='input the genomic results dir')
parser.add_argument('--muti_dir', type=str, default='', help='output the mutimodal results')



def read_pkl(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    ids,risks,survivals,censorships = [],[],[],[]
    for key in data:
        ids.append(key)
        censorships.append(data[key]['censorship'])
        if isinstance(data[key]['risk'],torch.Tensor) or isinstance(data[key]['risk'],np.ndarray):
            risks.append(np.squeeze(data[key]['risk']).item())
        else:
            risks.append(data[key]['risk'])
        survivals.append(data[key]['survival'])
    survivals = np.array(survivals)
    censorships = np.array(censorships)
    risks = np.array(risks)
    return ids,risks,survivals,censorships

def write_pkl(ids, risks,survivals,censorships,filepath):
    data = {}
    for id,r,s,c in zip(ids,risks,survivals,censorships):
        data.update({id: {
                'slide_id': np.array(id),
                'risk': r,
                'disc_label': -1,
                'survival': s,
                'censorship': c
            }})
    writer = open(filepath,'wb')
    pickle.dump(data, writer)
    writer.close()

def summary_mean_results(rs):
    mean = np.mean(rs)
    maxi = max(rs)
    mini = min(rs)
    print(f"{round(mean,3)}({round(mini,3)}-{round(maxi,3)}),std:{round(np.std(rs),3)}")
    
if __name__=='__main__':
    args = parser.parse_args()
    if not os.path.exists(args.muti_dir):
        os.mkdir(args.muti_dir)
    c_indexs_wsi = []
    c_indexs_rna = []
    c_indexs_merge = []
    for epoch in range(5):
        wsi_path = os.path.join(args.wsi_dir,f'split_latest_val_{epoch}_results.pkl')
        rna_path = os.path.join(args.rna_dir,f'split_latest_val_{epoch}_results.pkl')

        ids1,r1,s1,c1 = read_pkl(wsi_path)
        ids2,r2,s2,c2 = read_pkl(rna_path)
        merge_r = np.maximum(r1,r2)
        assert ids1==ids2 and (s1==s2).all()
        write_pkl(ids1,merge_r,s1,c1,os.path.join(args.muti_dir,f'split_latest_val_{epoch}_results.pkl'))
        c_index1 = concordance_index_censored((1-c1).astype(bool),s1,r1,tied_tol=1e-8)[0]
        c_index2 = concordance_index_censored((1-c1).astype(bool),s1,r2,tied_tol=1e-8)[0]
        c_index_merge = concordance_index_censored((1-c1).astype(bool),s1,merge_r,tied_tol=1e-8)[0]
        c_indexs_wsi.append(c_index1)
        c_indexs_rna.append(c_index2)
        c_indexs_merge.append(c_index_merge)
    print('wsi:',end='')
    summary_mean_results(c_indexs_wsi)
    print('rna:',end='')
    summary_mean_results(c_indexs_rna)
    print('muti:',end='')
    summary_mean_results(c_indexs_merge)
