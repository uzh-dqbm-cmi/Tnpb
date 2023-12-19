import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from .dataset import ProtospacerDataset, ProtospacerExtendedDataset, PartitionDataTensor
from src.utils import RegModelScore, plot_y_distrib_acrossfolds, ReaderWriter, one_hot_encode
import numpy as np
from src.utils import get_char

class MaxNormalizer:
    def __init__(self):
        self.length_norm = ['polytvalues', 'polygvalues', 'polyavalues', 'polycvalues', 'Proto_GC_count']
        self.mfe_norm = ['MFE_scaffold_spacer_HDV', 'MFE_spacer_HDV', 'MFE_scaffold_spacer']
        self.mt_norm = ['spacermt']
        self.percent_norm = ['Proto_GC_content']
        
        # theoretical max
        self.normalizer_info_max = [(self.length_norm, 20.), 
                                    (self.mfe_norm, 120.), 
                                    (self.mt_norm, 150.),
                                    (self.percent_norm, 100.)]

        self.normalizer_info_minmax = [(self.length_norm, 0., 20.), 
                                       (self.mfe_norm, -120., 0.), 
                                       (self.mt_norm, 0., 150.),
                                      (self.percent_norm, 0., 100.)]
    def get_colnames(self):
        return ['polytvalues', 'polygvalues', 'polyavalues', 'polycvalues',
                'Proto_GC_content', 'Proto_GC_count', 'spacermt', 'MFE_scaffold_spacer_HDV', 'MFE_spacer_HDV',
                'MFE_scaffold_spacer']
    
    def normalize_cont_cols(self, df, normalize_opt = 'max', suffix=''):
        if normalize_opt == 'max':
            print('--- max normalization ---')
            return self.normalize_cont_cols_max_(df, suffix=suffix)
        elif normalize_opt == 'minmax':
            print('--- minmax normalization ---')
            return self.normalize_cont_cols_minmax_(df, suffix=suffix)

    def normalize_cont_cols_max_(self, df, suffix=''):
        """inplace max normalization of columns"""
        normalizer_info = self.normalizer_info_max
        for colgrp in normalizer_info:
            colnames, max_val = colgrp
            for colname in colnames:
                df[colname+suffix] = df[colname]/max_val
            
    def normalize_cont_cols_minmax_(self, df, suffix=''):
        """inplace min-max normalization of columns"""
        normalizer_info = self.normalizer_info_minmax
        for colgrp in normalizer_info:
            colnames, min_val, max_val = colgrp
            for colname in colnames:
                df[colname+suffix] = ((df[colname] - min_val)/(max_val - min_val)).clip(lower=0., upper=1.)

def prepare_nonproto_features(x_non_protos_f, normalize_opt):
    """get the normalized derived features
    
    Args:
        x_non_protos_f: np.array, (bsize, feat_dim), representing the derived features
        normalize_opt: str, {'max', 'minmax'}, normalization option that is passed to :class:`MaxNormalizer`
    
    """
    x_non_protos_f_df = pd.DataFrame(x_non_protos_f)

    non_protos_colnames = ['polytvalues', 'polygvalues', 'polyavalues', 'polycvalues', 
                           'Proto_GC_content', 'Proto_GC_count', 'spacermt', 
                           'MFE_scaffold_spacer_HDV', 'MFE_spacer_HDV','MFE_scaffold_spacer']
    x_non_protos_f_df.columns = non_protos_colnames
    
    MaxNormalizer().normalize_cont_cols(x_non_protos_f_df, normalize_opt = normalize_opt, suffix='_norm')
    
    norm_colnames = [f'{colname}_norm' for colname in MaxNormalizer().get_colnames()]
    
    x_non_protos_f_norm = x_non_protos_f_df[norm_colnames].values
    return x_non_protos_f_df, x_non_protos_f_norm


def generate_partition_datatensor(criscas_datatensor, data_partitions):
    datatensor_partitions = {}
    for run_num in data_partitions:
        datatensor_partitions[run_num] = {}
        for dsettype in data_partitions[run_num]:
            target_ids = data_partitions[run_num][dsettype]
            datatensor_partition = PartitionDataTensor(criscas_datatensor, target_ids, dsettype, run_num)
            datatensor_partitions[run_num][dsettype] = datatensor_partition
    
    return(datatensor_partitions)


## do train validation seperation
def get_datatensor_partitions(data_partitions, model_name, x_proto, y, x_feat=None, 
                              fdtype=torch.float32, train_size=0.9, random_state=42):
    train_val_test = {}
    for i in range(len(data_partitions)):
        if data_partitions[i]['train_index'] is not None:
            tr_index, val_index = train_test_split(data_partitions[i]['train_index'],
                                               train_size=train_size,
                                               random_state= random_state, 
                                               shuffle=True)
            
        else:
            
            tr_index =  data_partitions[i]['test_index']
            val_index =  data_partitions[i]['test_index']
            
        train_val_test[i] = {'train': tr_index.tolist(),
                             'validation': val_index.tolist(),
                             'test': data_partitions[i]['test_index'].tolist()}

    dpartitions = train_val_test
    num_runs = len(dpartitions)

    if model_name in {'FFN', 'CNN'}:
        #print('we are here')
        xproto_dtype = fdtype
        xfeat_dtype = fdtype
        ydtype = fdtype
    elif model_name in {'RNN', 'Transformer'}:
        xproto_dtype = torch.int64        
        xfeat_dtype = fdtype
        ydtype = fdtype
    else: # make sure to not pass incorrect types 
        xproto_dtype = None
        xfeat_dtype = None
        ydtype = None

    if x_feat is not None:
        criscas_dtensor = ProtospacerExtendedDataset(x_proto, x_feat, y, xproto_dtype, xfeat_dtype, ydtype=ydtype)
    else:
        # no extended features
        criscas_dtensor = ProtospacerDataset(x_proto, y, xproto_dtype, ydtype=ydtype)

    datatensor_partitions = generate_partition_datatensor(criscas_dtensor, dpartitions)
    return dpartitions, datatensor_partitions

def construct_load_dataloaders(dataset_fold, dsettypes, score_type, config, wrk_dir):
    """construct dataloaders for the dataset for one run or fold
       Args:
            dataset_fold: dictionary,
                          example: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                                    'class_weights': tensor([0.6957, 1.7778])
                                   }
            score_type:  str, either {'regression'}
            dsettype: list, ['train', 'validation', 'test']
            config: dict, {'batch_size': int, 'num_workers': int}
            wrk_dir: string, folder path
    """

    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    flog_out = {}
    score_dict = {}
   
    for dsettype in dsettypes:
        if(dsettype == 'train'):
            shuffle = True
            sampler = None
            batch_size = config['batch_size']
        else:
            print(dsettype)
            shuffle = False
            sampler = None
            batch_size = dataset_fold[dsettype].num_samples
        

        data_loaders[dsettype] = DataLoader(dataset_fold[dsettype],
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=config['num_workers'],
                                            sampler=sampler)
        
        epoch_loss_avgbatch[dsettype] = []
        
        if(score_type == 'regression'):
            score_dict[dsettype] = RegModelScore(0, 0.0, 0.0, 0.0)

        if(wrk_dir):
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
        else:
            flog_out[dsettype] = None

    return (data_loaders, epoch_loss_avgbatch, score_dict,  flog_out) 

# workflow to get the data ready depending on chosen model and experiment
def get_data_ready(args, normalize_opt = 'max', train_size=0.9, fdtype=torch.float32, plot_y_distrib=True):
    ## prepare the data
    data_dir = args.data_dir + args.target_dir
    data_partitions = ReaderWriter.read_data(data_dir + '/data_partitions.pkl')
    data = ReaderWriter.read_data(data_dir + '/list_of_x_f_y.pkl')
    x_protospacer, x_extended_f,x_non_protos_f, y = data
    

    if args.model_name in {'CNN', 'FFN'}:
        ## onehot-encode the protospacer features
        
        proc_x_protospacer = one_hot_encode(x_protospacer)
        proc_x_protospacer = proc_x_protospacer.reshape(proc_x_protospacer.shape[0], -1)
    elif args.model_name in {'Transformer', 'RNN'}:
        proc_x_protospacer = x_protospacer 
        
    x_non_protos_f_df, x_non_protos_f_norm = prepare_nonproto_features(x_non_protos_f, normalize_opt)
    
    if args.exp_name == 'protospacer_extended':
        x_non_protos_features = x_non_protos_f_norm
    elif args.exp_name == 'protospacer':
        x_non_protos_features = None
    dpartitions, datatensor_partitions = get_datatensor_partitions(data_partitions,
                                                                   args.model_name,
                                                                   proc_x_protospacer,
                                                                   y,
                                                                   x_non_protos_features,
                                                                   fdtype=fdtype,
                                                                   train_size=train_size,
                                                                   random_state=args.random_seed)
    if plot_y_distrib:
        plot_y_distrib_acrossfolds(dpartitions, y, opt='separate_dsettypes')
        plot_y_distrib_acrossfolds(dpartitions, y, opt='separate_folds')
    return dpartitions, datatensor_partitions

def get_data_ready_for_user_samples(data, args, num_runs=5, normalize_opt = 'max', train_size=0.0, fdtype=torch.float32):
    ## prepare the data for the user samples (only used for inference)
    x_protospacer, y, x_extended_f,x_non_protos_f = data
    data_partitions = {}
    for run_num in range(num_runs):
        data_partitions[run_num] = {'train_index': None, 'test_index':np.arange(y.shape[0]) }


    if args.model_name in {'CNN', 'FFN'}:
        ## onehot-encode the protospacer features
        proc_x_protospacer = one_hot_encode(x_protospacer)
        proc_x_protospacer = proc_x_protospacer.reshape(proc_x_protospacer.shape[0], -1)
    elif args.model_name in {'Transformer', 'RNN'}:
        proc_x_protospacer = x_protospacer 
    
    if x_extended_f is not None:
        x_non_protos_f_df, x_non_protos_f_norm = prepare_nonproto_features(x_non_protos_f, normalize_opt)
    
    if args.exp_name == 'protospacer_extended':
        x_non_protos_features = x_non_protos_f_norm
    elif args.exp_name == 'protospacer':
        x_non_protos_features = None
        
    dpartitions, datatensor_partitions = get_datatensor_partitions(data_partitions,
                                                                   args.model_name,
                                                                   proc_x_protospacer,
                                                                   y,
                                                                   x_non_protos_features,
                                                                   fdtype=fdtype,
                                                                   train_size=train_size,
                                                                   random_state=args.random_seed)
    return dpartitions, datatensor_partitions

## prepare the data for the user samples, (only used for inference)
def prepare_the_data_for_user_samples(args):
    df = pd.read_csv(os.path.join(args.data_dir,args.data_name ),
                 header=0, 
                 delimiter=',' )
    if 'efficiency' in df.columns:
        y = df['efficiency']
    else:
        df['efficiency'] = 0
        y = np.array(df['efficiency'])
        print('true target is not provided, the correlation and loss will be meaningless')
        
    args.feature_list = ['polytvalues', 'polygvalues', 'polyavalues', 'polycvalues',
       'Proto_GC_content', 'Proto_GC_count', 'spacermt', 'MFE_scaffold_spacer_HDV', 'MFE_spacer_HDV',
       'MFE_scaffold_spacer']
    
    if len(df.iloc[0])>3:
        extended_featurs = df.columns[5:12].tolist() + df.columns[15:].tolist()
        Non_protospacer_features = [args.feature_list]
        df_extended_features = df[extended_featurs]
        x_extended_f = np.array(df_extended_features)
        x_non_protos_f =  np.array(Non_protospacer_features)
    else:
        x_extended_f = None
        x_non_protos_f = None
    
    protospacer = df['sequence'].apply(get_char)
    num_nucl = len(protospacer.columns)
    protospacer.columns = [f'B_encoded{i}' for  i in range(1, num_nucl+1)]
    protospacer.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
    x_protospacer = np.array(protospacer)
    
    return [x_protospacer, y, x_extended_f, x_non_protos_f]
