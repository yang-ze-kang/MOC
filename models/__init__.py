from .model_genomic import SNN
from .model_set_mil import MIL_Sum_FC_surv, MIL_Attention_FC_surv
from .model_coattn import MCAT_Surv
from .model_porpoise import PorpoiseMMF

def create_model(args):
    if args.model_type == 'porpoise_mmf':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes,
                      'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2,
                      'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,
                      }
        model = PorpoiseMMF(**model_dict)
    elif args.model_type == 'snn':
        model_dict = {'omic_input_dim': args.omic_input_dim,
                      'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == 'deepset':
        model_dict = {'omic_input_dim': args.omic_input_dim,
                      'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Sum_FC_surv(**model_dict)
    elif args.model_type == 'amil':
        model_dict = {'omic_input_dim': args.omic_input_dim,
                      'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv(**model_dict)
    elif args.model_type == 'mcat':
        model_dict = {'fusion': args.fusion,
                      'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)
    else:
        raise NotImplementedError
    return model