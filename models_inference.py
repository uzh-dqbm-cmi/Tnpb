import os
import numpy as np
import pandas as pd
import torch
import argparse
from models.data_process import get_datatensor_partitions, prepare_nonproto_features, generate_partition_datatensor
from models.dataset import ProtospacerDataset, ProtospacerExtendedDataset
from src.utils import create_directory, one_hot_encode, get_device, ReaderWriter, print_eval_results
import matplotlib.pyplot as plt
from models.trainval_workflow import run_inference
from src.utils import compute_eval_results_df

cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-model_name',  type=str, help = 'name of the model')
cmd_opt.add_argument('-exp_name',  type=str, help = 'name of the experiment')

cmd_opt.add_argument('-data_dir',  type=str,default = './data/', help = 'directory of the data')
cmd_opt.add_argument('-target_dir',  type=str, default='processed',  help = 'folder name to save the processed data')
cmd_opt.add_argument('-working_dir',  type=str, default='./', help = 'the main working directory')
cmd_opt.add_argument('-output_path', type=str, help='path to save the trained model')
cmd_opt.add_argument('-model_path', type=str, help='path to trained model')
cmd_opt.add_argument('-random_seed', type=int,default=42)
cmd_opt.add_argument('-epoch_num', type=int, default =200, help='number of training epochs')
args, _ = cmd_opt.parse_known_args()


def get_data_ready(args, normalize_opt = 'max', train_size=0.9, fdtype=torch.float32):
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
    return dpartitions, datatensor_partitions


gpu_index = 0
res_desc = {}
version=2
for model_name in ['FFN', 'CNN', 'RNN', 'Transformer']:
    args.model_name =  model_name# {'RNN','CNN', 'Transformer'}
    res_desc[model_name] = {}
    for exp_name in ['protospacer', 'protospacer_extended']:
        args.exp_name = exp_name
        model_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}',
                                  exp_name)
        dpartitions, datatensor_partitions = get_data_ready(args, 
                                                            normalize_opt='max',
                                                            train_size=0.9, 
                                                            fdtype=torch.float32)

        train_val_path = os.path.join(model_path, 'train_val')
        test_path = os.path.join(model_path, 'test')
        
        print(f'Running model: {model_name}, exp_name: {exp_name}, saved at {train_val_path}')
        a, b = run_inference(datatensor_partitions, 
                             train_val_path, 
                             test_path, 
                             gpu_index, 
                             to_gpu=True)
        print('='*15)
        res_desc[model_name][exp_name] = compute_eval_results_df(test_path, len(dpartitions))        


for model_name in res_desc:
    for exp_name in res_desc[model_name]:
        print(f'model_name: {model_name}, exp_name: {exp_name}')
        display(res_desc[model_name][exp_name])
        print('='*15)

