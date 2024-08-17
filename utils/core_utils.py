from argparse import Namespace
from collections import OrderedDict
import os
import pdb
import pickle

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from datasets.dataset_generic import save_splits
from models.genomic import Genomic_SNN_Transformer_MIL, GenomicTransformer
from models.model_genomic import SNN
from models.model_set_mil import MIL_Sum_FC_surv, MIL_Attention_FC_surv, MIL_Cluster_FC_surv, TwoAMIL
from models.model_coattn import MCAT_Surv
from models.model_porpoise import PorpoiseMMF, PorpoiseAMIL
from models.idea1 import Idea1
from models.idea2 import WSIClusterSurvival, Idea2_2, Idea2_2_SnnMIL
from models.muti_amil_snn import MutiAmilSnn
from models.TransMIL import TransMIL
from utils.utils import *
from utils.loss_func import NLLSurvLoss,CoxPHLoss,ELBLoss,SDLoss

from utils.coattn_train_utils import *
from utils.cluster_train_utils import *
from pyinstrument import Profiler

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, train_val_split, val_split = datasets
    # save_splits(datasets, ['train', 'val'], os.path.join(
    #     args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'coxph':
        loss_fn = CoxPHLoss()
    elif args.bag_loss == 'elb':
        loss_fn = ELBLoss()
    elif args.bag_loss in ['sd','sd-weight']:
        loss_fn = SDLoss()
    else:
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)

    if args.reg_type == 'omic':
        reg_fn = l1_reg_omic
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')

    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type == 'porpoise_mmf':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes,
                      'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2,
                      'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,
                      }
        model = PorpoiseMMF(**model_dict)
    elif args.model_type == 'porpoise_amil':
        model_dict = {'n_classes': args.n_classes}
        model = PorpoiseAMIL(**model_dict)
    elif args.model_type == 'snn':
        model_dict = {'omic_input_dim': args.omic_input_dim,'dropout':args.dropout,
                      'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == 'snn_mil':
        model_dict = {'omic_embedding_size': args.omic_embedding_size,
                      'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = Genomic_SNN_Transformer_MIL(**model_dict)
    elif args.model_type == 'two-amil':
        model_dict = {'omic_embedding_size': args.omic_embedding_size,
                      'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = TwoAMIL(**model_dict)
    elif args.model_type == 'deepset':
        model_dict = {'omic_input_dim': args.omic_input_dim,
                      'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Sum_FC_surv(**model_dict)
    elif args.model_type == 'amil':
        model_dict = {'omic_input_dim': args.omic_input_dim,'drop_instance':args.drop_instance,
                      'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv(**model_dict)
    elif args.model_type == 'mi_fcn':
        model_dict = {'omic_input_dim': args.omic_input_dim,
                      'fusion': args.fusion, 'num_clusters': 10, 'n_classes': args.n_classes}
        model = MIL_Cluster_FC_surv(**model_dict)
    elif args.model_type == 'mcat':
        model_dict = {'fusion': args.fusion,
                      'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'idea1':
        model_dict = {'path_input_dim': args.path_input_dim, 'omic_input_dim': args.omic_input_dim, 'fuse': args.fuse,
                      'n_classes': args.n_classes, 'dropinput': args.dropinput, 'mil_method': args.mil_method, 'use_transformer': args.use_transformer
                      }
        model = Idea1(**model_dict)
    elif args.model_type == 'gt':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fuse': args.fuse, 'num_layer': args.num_layer,
                      'n_classes': args.n_classes
                      }
        model = GenomicTransformer(**model_dict)
    elif args.model_type == 'wsi-cluster':
        model_dict = {'cluster_agg_method': args.cluster_agg_method,
                      'cluster_pool_method': args.cluster_pool_method,
                      'phenotype_pool_method': args.phenotype_pool_method,
                      'num_clusters': args.num_cluster, 'n_classes': args.n_classes}
        model = WSIClusterSurvival(**model_dict)
    elif args.model_type == 'idea2-2':
        model_dict = {'cluster_agg_method': args.cluster_agg_method,
                      'cluster_pool_method': args.cluster_pool_method,
                      'phenotype_pool_method': args.phenotype_pool_method,
                      'num_clusters': args.num_cluster, 'n_classes': args.n_classes,
                      'sim_fun': args.cluster_sim_fun, 'max_iter': args.cluster_max_iter, 'cluster_lr': args.cluster_lr,
                      'filter_prob': args.filter_prob, 'global_centers_update_method': args.global_centers_update_method}
        model = Idea2_2(**model_dict)
    elif args.model_type == 'idea2-2SnnMIL':
        model_dict = {'cluster_agg_method': args.cluster_agg_method,
                      'cluster_pool_method': args.cluster_pool_method,
                      'phenotype_pool_method': args.phenotype_pool_method,
                      'num_clusters': args.num_cluster, 'n_classes': args.n_classes, 'omic_embedding_size': args.omic_embedding_size,
                      'omic_sizes': args.omic_sizes, 'sim_fun': args.cluster_sim_fun, 'max_iter': args.cluster_max_iter, 'cluster_lr': args.cluster_lr,
                      'filter_prob': args.filter_prob, 'global_centers_update_method': args.global_centers_update_method}
        model = Idea2_2_SnnMIL(**model_dict)
    elif args.model_type == 'transmil':
        model_dict = {'n_classes': args.n_classes}
        model = TransMIL(**model_dict)
    elif args.model_type == 'mwg':
        model_dict = {'omic_input_dim': args.omic_input_dim,
                      'model_size_omic': args.model_size_omic,
                      'n_classes': args.n_classes}
        model = MutiAmilSnn(**model_dict)
    else:
        raise NotImplementedError

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    if args.bag_loss in ['contrast','elb','sd','sd-weight']:
        patient_num = train_split.get_num()
        train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                        weighted=False, mode=args.data_mode, batch_size=args.batch_size, collate=None, sample=args.sample)
        train_val_loader = get_split_loader(train_val_split, training=False, testing=args.testing,
                                           weighted=False, mode=args.data_mode, batch_size=args.batch_size)
    else:
        train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                        weighted=args.weighted_sample, mode=args.data_mode, batch_size=args.batch_size)
    val_loader = get_split_loader(
        val_split,  testing=args.testing, mode=args.data_mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(
            warmup=0, patience=10, stop_epoch=20, verbose=True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type == 'idea2-2' and (epoch+1) >= args.filter_start_epoch:
            model.start_filtering()
        if args.bag_loss in ['contrast','elb','sd','sd-weight']:
            train_loop_survival_contrast(args, epoch, model, train_loader, optimizer,
                                         args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc,patient_num)
            with torch.no_grad():
                validate_survival_contrast(cur, epoch, model, train_val_loader, args.n_classes, early_stopping,
                                           monitor_cindex,writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir,name='train')
        else:
            train_loop_survival(epoch, model, train_loader, optimizer,
                                args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
        with torch.no_grad():
            if args.bag_loss in ['contrast','elb','sd','sd-weight']:
                stop = validate_survival_contrast(cur, epoch, model, val_loader, args.n_classes, early_stopping,
                                     monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
            else:
                stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping,
                                     monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
        torch.save(model.state_dict(), os.path.join(
            args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    if args.model_type == 'idea2-2':
        torch.save(model.cluster.global_centers, os.path.join(
            args.results_dir, "s_{}_checkpoint_kmeans_global_centers.pt".format(cur)))
    torch.save(model.state_dict(), os.path.join(
        args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(
        args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    if args.model_type == 'idea2-2':
        model.cluster.global_centers = torch.load(os.path.join(
            args.results_dir, "s_{}_checkpoint_kmeans_global_centers.pt".format(cur)))
    with torch.no_grad():
        if args.bag_loss in ['contrast','elb','sd','sd-weight']:
            results_val_dict, val_cindex = summary_survival_contrast(
                model, val_loader, args.n_classes, loss_fn)
        else:
            results_val_dict, val_cindex = summary_survival(
                model, val_loader, args.n_classes)

    print('Val c-Index: {:.4f}'.format(val_cindex))
    writer.close()
    return results_val_dict, val_cindex


def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    # centers = []

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        # To device
        if isinstance(data_WSI, dict):
            for key in data_WSI:
                data_WSI[key] = data_WSI[key].to(device)
            bag_size = data_WSI['path_features'].size(0)
        else:
            data_WSI = data_WSI.to(device)
            bag_size = data_WSI.size(0)
        for key in data_omic:
            data_omic[key] = data_omic[key].to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)
        outs = model(x_path=data_WSI, **data_omic)
        # center = model.cluster.global_centers.cpu().numpy()
        # centers.append(center)
        if isinstance(outs, torch.Tensor): #MMF
            loss = loss_fn(h=outs, y=y_disc, t=event_time, c=censor)
            survival = torch.cumprod(1 - outs, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        elif isinstance(outs, dict):
            hazards = outs['hazards']
            # survival = outs['survival']
            # risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            # loss = loss_fn(h=hazards, y=y_disc, t=event_time, c=censor)
            
            # CoxPHLoss
            risk = hazards.squeeze().detach().cpu().numpy()
            loss = loss_fn(hazards,event_time)
        else:
            h_path, h_omic, h_mm = outs
            loss = 0.5*loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            h = h_mm
            survival = torch.cumprod(1 - h, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value +
                  loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), bag_size))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(
                batch_idx, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0]), float(risk[0]), bag_size))

        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
    # pdb.set_trace()
    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships)
    if np.any(np.isnan(all_risk_scores)):
        pdb.set_trace()
    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def train_loop_survival_contrast(args, epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, patient_num=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.
    loader.training = True
    print('\n')

    for batch_idx, data in enumerate(loader):
        # To device
        # pdb.set_trace()
        d1, d2, weight = data
        d1,o1 = d1
        d2,o2=d2
        o1 = o1.to(device)
        o2 = o2.to(device)
        weight = weight.to(device)
        img1, omics1, y_disc1, t1, c1 = d1
        img2, omics2, y_disc2, t2, c2 = d2
        y_disc1 = y_disc1.to(device)
        y_disc2 = y_disc2.to(device)
        c1 = c1.to(device)
        c2 = c2.to(device)
        if isinstance(img1, dict):
            for key in img1:
                img1[key] = img1[key].to(device)
            for key in img2:
                img2[key] = img2[key].to(device)
        else:
            img1 = img1.to(device)
            img2 = img2.to(device)
        for key in omics1:
            omics1[key] = omics1[key].to(device)
            omics2[key] = omics2[key].to(device)
        t1 = t1.to(device)
        t2 = t2.to(device)
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        #     risk1 = model(x_path=img1, **omics1)
        # print(prof.table())
        # pdb.set_trace()
        # if c1==1:
        #     with torch.no_grad():
        #         risk1 = model(x_path=img1, **omics1)
        # else:
        outs1 = model(x_path=img1, **omics1)
        outs2 = model(x_path=img2, **omics2)
        if isinstance(outs1,dict):
            logit1 = outs1['logits']
            logit2 = outs2['logits']
            hazard1 = outs1['hazards']
            hazard2 = outs2['hazards']
            # hazards1 = outs1['hazards']
            # hazards2 = outs2['hazards']
            # survival2 = outs1['survival']
            # survival2 = outs2['survival']
            # risk1 = -torch.sum(survival2, dim=1)
            # risk2 = -torch.sum(survival2, dim=1)
            # # pdb.set_trace()
            # loss1 = loss_fn(h=hazards1, y=y_disc1, t=t1, c=c1)
            # loss2 = loss_fn(h=hazards2, y=y_disc2, t=t2, c=c2)
            # loss3 = risk1/(risk2+1e-7)
            # loss = loss1 + loss2 +loss3
            # loss = risk1/(risk2+1e-7)
        elif isinstance(outs1,list):
            w1,g1,w2,g2 = outs1[0], outs1[1],outs2[0],outs2[1]
            w1 = torch.sigmoid(w1)
            g1 = torch.sigmoid(g1)
            w2 = torch.sigmoid(w2)
            g2 = torch.sigmoid(g2)
            risk1 = (w1+g1)/2
            risk2 = (w2+g2)/2
            loss = w1/(w2+1e-7)+w1/(g2+1e-7)+g1/(w2+1e-7)+g1/(g2+1e-7)+risk1/(risk2+1e-7)
        if isinstance(loss_fn, ELBLoss):
            loss = loss_fn(logit1, logit2)
            risk1 = logit1
            risk2 = logit2
        elif isinstance(loss_fn, SDLoss):
            loss = loss_fn(logit1, logit2, weight=weight)
            risk1 = hazard1
            risk2 = hazard2
        # else:
        #     raise NotImplementedError
        # loss = (risk1+1e-7)/(risk2+1e-7)
        # loss = torch.exp(risk1/(risk2+1e-7))
        # loss = risk1/(risk1+risk2+1e-7)
        # loss = torch.pow((risk1/(risk1+risk2+1e-7)-t1/(t2+1e-7)),2)
        
        # loss = risk1- risk2
        
        # sub symmetry
        # scale = risk1.detach()/(risk2.detach()+1e-7)
        # loss = (risk1-risk2)*scale
        # loss = torch.pow((c2/(c1+1e-7)-risk1/(risk2+1e-7)),2) # mse2
        # loss = risk1/(risk2+1e-7)*((o2-o1)/patient_num)
        # loss = torch.exp(risk1-risk2)
        # loss = torch.pow(risk1/(risk2+1e-7),2)
        # loss = (risk1+1)/(risk2+1)
        # loss = risk1-risk2
        # loss = torch.exp(risk1-risk2)
        # if risk1<0:
        #     loss =loss + torch.pow(risk1-0.1,2)
        # elif risk1>1:
        #     loss =loss + torch.pow(risk1-0.9,2)
        # if risk2<0:
        #     loss =loss + torch.pow(risk2-0.1,2)
        # elif risk2>1:
        #     loss =loss + torch.pow(risk2-0.9,2) 
        # loss = risk1*(1-risk2)
        # loss = (1-risk2)/(1-risk1)
        
        # mse3
        # if t1==1 and risk1*c1/((risk2+1e-7)*c2) <=1:
        #     loss = 0
        # else:
        #     loss = torch.pow((c2/(c1+1e-7)-risk1/(risk2+1e-7)),2)
            
        loss_value = loss.item()
        train_loss += loss_value

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if (batch_idx + 1) % 2048 == 0:
            print(
                f'batch:{batch_idx},risk1:{risk1.item()},risk2:{risk2.item()},t1:{t1.item()},t2:{t2.item()},loss:{loss_value}\n')
        # backward pass
        if args.loss_threshold is not None:
            if loss>args.loss_threshold:
                loss = loss / gc + loss_reg
                loss.backward()
        else:
            loss = loss / gc + loss_reg
            loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
    train_loss = train_loss/len(loader)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, name='val'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        # To device
        if isinstance(data_WSI, dict):
            for key in data_WSI:
                data_WSI[key] = data_WSI[key].to(device)
        else:
            data_WSI = data_WSI.to(device)
        for key in data_omic:
            data_omic[key] = data_omic[key].to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        with torch.no_grad():
            # return hazards, S, Y_hat, A_raw, results_dict
            outs = model(x_path=data_WSI, **data_omic)

        if isinstance(outs, torch.Tensor): # MMF
            loss = loss_fn(h=outs, y=y_disc, t=event_time, c=censor)
            survival = torch.cumprod(1 - outs, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        elif isinstance(outs, dict):
            hazards = outs['hazards']
            # survival = outs['survival']
            # risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            # loss = loss_fn(h=hazards, y=y_disc, t=event_time, c=censor)
            
            # CoxPHLoss
            risk = hazards.squeeze().detach().cpu().numpy()
            loss = loss_fn(hazards,event_time)
        else:
            h_path, h_omic, h_mm = outs
            loss = 0.5*loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            h = h_mm
            survival = torch.cumprod(1 - h, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print(f'epoch:{epoch},{name},C-Index:{c_index}\n')
    if writer:
        writer.add_scalar(f'{name}/loss_surv', val_loss_surv, epoch)
        writer.add_scalar(f'{name}/loss', val_loss, epoch)
        writer.add_scalar(f'{name}/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(
            results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_survival_contrast(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, name='val'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    loader.training = False
    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        # To device
        if isinstance(data_WSI, dict):
            for key in data_WSI:
                data_WSI[key] = data_WSI[key].to(device)
        else:
            data_WSI = data_WSI.to(device)
        for key in data_omic:
            data_omic[key] = data_omic[key].to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        with torch.no_grad():
            outs = model(x_path=data_WSI, **data_omic)

        if isinstance(outs, torch.Tensor):
            risk = outs.detach().cpu().numpy()
        elif isinstance(outs, dict):
            if isinstance(loss_fn, ELBLoss):
                risk = outs['logits'].detach().cpu().numpy()
            else:
                risk = outs['hazards'].detach().cpu().numpy()
        elif isinstance(outs,list):
            risk = (outs[0]+outs[1])/2
        else:
            h_path, h_omic, h_mm = outs
            h = h_mm
            survival = torch.cumprod(1 - h, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()



    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print(f'epoch:{epoch},{name},C-Index:{c_index}\n')
    if writer:
        writer.add_scalar(f'{name}/c-index', c_index, epoch)

    return False


def summary_survival(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

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
            survival = torch.cumprod(1 - outs, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        elif isinstance(outs, dict):
            hazards = outs['hazards']
            # survival = outs['survival']
            # risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            
            # CoxPHLoss
            risk = hazards.detach().cpu().numpy()
        else:
            h_path, h_omic, h_mm = outs
            h = h_mm
            survival = torch.cumprod(1 - h, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

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
    print(f'Val C-Index:{c_index}\n')
    return patient_results, c_index

def summary_survival_contrast(model, loader, n_classes, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            if isinstance(loss_fn,ELBLoss):
                risk = outs['logits'].detach().cpu().numpy()
            else:
                risk = outs['hazards'].detach().cpu().numpy()
        elif isinstance(outs,list):
            risk = (outs[0]+outs[1])/2
        else:
            h_path, h_omic, h_mm = outs
            h = h_mm
            survival = torch.cumprod(1 - h, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

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
