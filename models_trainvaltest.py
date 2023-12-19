import os
import numpy as np
import pandas as pd
import torch
import argparse
from models.data_process import get_datatensor_partitions, prepare_nonproto_features, generate_partition_datatensor,get_data_ready
from models.dataset import ProtospacerDataset, ProtospacerExtendedDataset
from models.trainval_workflow import run_trainevaltest_workflow
from models.trainval_workflow import run_inference
from models.hyperparam import build_config_map
from src.utils import create_directory, one_hot_encode, get_device, ReaderWriter 
from src.utils import print_eval_results, plot_y_distrib_acrossfolds, compute_eval_results_df
import matplotlib.pyplot as plt


cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-model_name',  type=str, help = 'name of the model')
cmd_opt.add_argument('-exp_name',  type=str, help = 'name of the experiment')

cmd_opt.add_argument('-data_dir',  type=str,default = './data/', help = 'directory of the data')
cmd_opt.add_argument('-target_dir',  type=str, default='processed',  help = 'folder name to save the processed data')
cmd_opt.add_argument('-working_dir',  type=str, default='./', help = 'the main working directory')
cmd_opt.add_argument('-output_path', type=str, help='path to save the trained model')
cmd_opt.add_argument('-random_seed', type=int,default=42)
cmd_opt.add_argument('-epoch_num', type=int, default =200, help='number of training epochs')
args, _ = cmd_opt.parse_known_args()

# predefined hyperparameters depending on the chosen model and experiment
def get_hyperparam_config(args):
    "return predefined hyperparameters for each model"
    to_gpu = True
    gpu_index=0
    optim_tup = None

    if args.model_name == 'FFN':
        batch_size = 100
        num_epochs = 300
        h = [60,10]
        l2_reg =0.1
        model_config_tup = (h, l2_reg, batch_size, num_epochs)

        if args.exp_name == 'protospacer_extended':
            mlpembedder_tup = (10, 16, 2, torch.nn.ReLU, 0.1, 1)
            xproto_inputsize = 20 + 10
        else:
            mlpembedder_tup = None
            xproto_inputsize = 20

        loss_func_name = 'MSEloss'
        perfmetric_name = 'pearson'

    if args.model_name == 'CNN':
        k = 2
        l2_reg = 0.5
        batch_size = 100
        num_epochs = 300
        model_config_tup = (k, l2_reg, batch_size, num_epochs)


        # input_dim, embed_dim, mlp_embed_factor, nonlin_func, p_dropout, num_encoder_units
        if args.exp_name == 'protospacer_extended':
            mlpembedder_tup = (10, 16, 2, torch.nn.ReLU, 0.1, 1)
            xproto_inputsize = 20 + 10
        else:
            mlpembedder_tup = None
            xproto_inputsize = 20

        loss_func_name = 'MSEloss'
        # loss_func_name = 'SmoothL1loss'
        perfmetric_name = 'spearman'

    elif args.model_name == 'RNN':
        embed_dim = 64
        hidden_dim = 64
        z_dim = 32
        num_hidden_layers =2
        bidirection = True
        p_dropout = 0.1
        rnn_class = torch.nn.GRU
        nonlin_func = torch.nn.ReLU
        pooling_mode = 'none'
        l2_reg = 1e-5
        batch_size = 1500
        num_epochs = 500

        model_config_tup = (embed_dim, hidden_dim, z_dim, num_hidden_layers, bidirection,
                   p_dropout, rnn_class, nonlin_func, pooling_mode, l2_reg, batch_size, num_epochs)

        # input_dim, embed_dim, mlp_embed_factor, nonlin_func, p_dropout, num_encoder_units
        if args.exp_name == 'protospacer_extended':
            mlpembedder_tup = (10, 16, 2, torch.nn.ReLU, 0.1, 1)
            xproto_inputsize = 20 + 10
        else:
            mlpembedder_tup = None
            xproto_inputsize = 20

        loss_func_name = 'SmoothL1loss'
        perfmetric_name = 'pearson'

    elif args.model_name == 'Transformer':
        embed_dim = 128
        num_attn_heads = 4
        num_trf_units = 1
        pdropout = 0.1
        activ_func = torch.nn.GELU
        multp_factor = 2
        multihead_type = 'Wide'
        pos_embed_concat_opt = 'stack'
        pooling_opt = 'none'
        weight_decay = 1e-8
        batch_size = 1000
        num_epochs = 1000


        model_config_tup = (embed_dim, num_attn_heads, num_trf_units,
                            pdropout, activ_func, multp_factor, multihead_type,
                            pos_embed_concat_opt, pooling_opt, weight_decay, batch_size, num_epochs)

        # input_dim, embed_dim, mlp_embed_factor, nonlin_func, p_dropout, num_encoder_units
        if args.exp_name == 'protospacer_extended':
            mlpembedder_tup = (10, 16, 2, torch.nn.GELU, 0.1, 1)
            xproto_inputsize = 20 + 10
        else:
            mlpembedder_tup = None
            xproto_inputsize = 20

        loss_func_name = 'SmoothL1loss'
        perfmetric_name = 'pearson'


    mconfig, options = build_config_map(args.model_name,
                                        optim_tup,
                                        model_config_tup,
                                        mlpembedder_tup,
                                        loss_func = loss_func_name)



    options['input_size'] = xproto_inputsize
    options['loss_func'] = loss_func_name # to refactor
    options['model_name'] = args.model_name
    options['perfmetric_name'] = perfmetric_name
    return mconfig, options

dsettypes = ['train', 'validation','test']
gpu_index = 0
res_desc = {}
version=2
for model_name in [ 'RNN', 'FFN','CNN', 'RNN','Transformer']
    print(model_name)
    args.model_name =  model_name # {'RNN','CNN', 'Transformer'}
    res_desc[model_name] = {}
    for exp_name in ['protospacer','protospacer_extended']:
        args.exp_name = exp_name
        model_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}',
                                  exp_name)
        dpartitions, datatensor_partitions = get_data_ready(args, 
                                                            normalize_opt='max',
                                                            train_size=0.9, 
                                                            fdtype=torch.float32,
                                                            plot_y_distrib=False)
        mconfig, options = get_hyperparam_config(args)
        print(options)
        
#         options['num_epochs'] = 10 # use this if you want to test a whole workflow run for all models using 10 epochs
        
        perfmetric_name = options['perfmetric_name']
        train_val_path = os.path.join(model_path, 'train_val')
        test_path = os.path.join(model_path, 'test')
        
        print(f'Running model: {model_name}, exp_name: {exp_name}, saved at {train_val_path}')
        perfmetric_run_map, score_run_dict = run_trainevaltest_workflow(datatensor_partitions, 
                                                                        (mconfig, options), 
                                                                        train_val_path,
                                                                        dsettypes,
                                                                        perfmetric_name,
                                                                        gpu_index, 
                                                                        to_gpu=True)
        print('='*15)
        res_desc[model_name][exp_name] = compute_eval_results_df(train_val_path, len(dpartitions))

print(res_desc)
