## use this file to get the inference for your sequences
import argparse
import pandas as pd
import os
import torch
from src.utils import get_char, outer_cross_val
import numpy as np
from src.utils import one_hot_encode
from models.data_process import prepare_nonproto_features
from models.dataset import ProtospacerDataset
from torch.utils.data import Dataset, DataLoader
from models.data_process import get_datatensor_partitions, prepare_nonproto_features, generate_partition_datatensor
from models.trainval_workflow import run_inference
from src.utils import compute_eval_results_df
from models.data_process import prepare_the_data_for_user_samples,get_data_ready_for_user_samples 


cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-data_dir',  type=str, default = './data/', help = 'directory of the data')
cmd_opt.add_argument('-target_dir',  type=str, default='processed',  help = 'folder name to save the processed data')
cmd_opt.add_argument('-working_dir',  type=str, default = './', help = 'the main working directory')
cmd_opt.add_argument('-data_name',  type=str, default='./', help = '')
cmd_opt.add_argument('-feature_list',  type=str, help = 'list of feature names we are gonna consider')
cmd_opt.add_argument('-random_seed', type=int,default=42)
args, _ = cmd_opt.parse_known_args()


data = prepare_the_data_for_user_samples(args)

def compile_prediction(tdir, num_runs=5):
    l = []
    for i in range(num_runs):
        df = pd.read_csv(os.path.join(tdir, f'run_{i}','predictions_test.csv'))
        if 'seq_id' not in df:
            df['seq_id'] = list(range(0, df.shape[0]))
        if 'Unnamed: 0' in df:
            del df['Unnamed: 0']
        df['run_num'] = i
        l.append(df)
    df = pd.concat(l, axis=0, ignore_index=True)
    return df        
    
def compute_avg_predictions(df):
    agg_df = df.groupby(by=['seq_id']).mean()
    agg_df.reset_index(inplace=True)
    for colname in ('run_num', 'Unnamed: 0'):
        if colname in agg_df:
            del agg_df[colname]
    return agg_df


gpu_index = 0
res_desc = {}
num_runs = 5 # number of trained model folds to use for prediction
version=2
for model_name in ['RNN', 'CNN']#, 'RNN', 'Transformer']:
    args.model_name =  model_name# {'RNN','CNN', 'Transformer'}
    res_desc[model_name] = {}
    for exp_name in ['protospacer']:
        args.exp_name = exp_name
        model_path = os.path.join(args.working_dir, 
                                  'output', 
                                  f'{model_name}_v{version}',
                                  exp_name)
        dpartitions, datatensor_partitions = get_data_ready_for_user_samples(data,
                                                                             args,
                                                                             num_runs=num_runs, # define how many model runs to be used
                                                                             normalize_opt='max',
                                                                             train_size=0., 
                                                                             fdtype=torch.float32)

        train_val_path = os.path.join(model_path, 'train_val')
        test_path = os.path.join(model_path, 'sample_test')
        
        print(f'Running model: {model_name}, exp_name: {exp_name}, saved at {train_val_path}')
        a, b = run_inference(datatensor_partitions, 
                             train_val_path, 
                             test_path, 
                             gpu_index, 
                             to_gpu=True)
                             #num_runs=num_runs)
        print('='*15)
        
        # save all predictions in one dataframe with the corresponding model run
        tdf = compile_prediction(test_path, num_runs=num_runs)
        # compute average prediction across the different runs of the same model
        tdf_ensemble = compute_avg_predictions(tdf)
        tdf.to_csv(os.path.join(test_path, f'{num_runs}fold_predictions.csv'), index=False)
        tdf_ensemble.to_csv(os.path.join(test_path, f'avg_{num_runs}fold_predictions.csv'), index=False)
        print('model prediction is saved at',os.path.join(test_path, f'avg_{num_runs}fold_predictions.csv'))
        #res_desc[model_name][exp_name] = compute_eval_results_df(test_path, len(dpartitions))        
