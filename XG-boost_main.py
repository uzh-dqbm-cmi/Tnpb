## train xg boost##
import os
import numpy as np
import argparse
import pandas as pd
import csv
from src.utils import ReaderWriter
from models.XGboost import main
from src.utils import one_hot_encode


cmd_opt = argparse.ArgumentParser(description='Argparser for data')
cmd_opt.add_argument('-model_name',  type=str, help = 'name of the model')
cmd_opt.add_argument('-data_dir',  type=str,default = './data/', help = 'directory of the data')
cmd_opt.add_argument('-target_dir',  type=str, default='processed',  help = 'folder name to save the processed data')
cmd_opt.add_argument('-working_dir',  type=str, default='./', help = 'the main working directory')
cmd_opt.add_argument('-data_name',  type=str, help = '')
cmd_opt.add_argument('-feature_list',  type=str, help = 'list of feature names we are gonna consider')
cmd_opt.add_argument('-used_for',type=str, help = '') 
cmd_opt.add_argument('-output_path', type=str, help='path to save the trained model')
cmd_opt.add_argument('-inner_cv', type=int, default = 2, help='number of inner cross validation folders')
args, _ = cmd_opt.parse_known_args()



#args.data_dir = './data/'
data_dir = args.data_dir + args.target_dir
#args.working_dir = './'
args.data_name = 'TnpB_nuclease_screen_for_ML_with_features.csv'

##1.load the data
data_partitions = ReaderWriter.read_data(data_dir + '/data_partitions.pkl')
data = ReaderWriter.read_data(data_dir + '/list_of_x_f_y.pkl')
x_protospacer, x_extended_f,x_non_protos_f, y = data

#args.model_name = 'protospacer_20'


if args.model_name == 'protospacer_20':
    print('we are running' + args.model_name + '_model')
    args.output_path = './output/XG_boost/protospacer_20'
    onehot_x = x_protospacer
    print('input size is',onehot_x.shape )
    main(args, data_partitions, onehot_x, y)

elif args.model_name == 'protospacer_80':
    print('we are running' + args.model_name + '_model')
    args.output_path = './output/XG_boost/protospacer_80'
    onehot_x = x_protospacer
    onehot_x = one_hot_encode(x_protospacer)
    onehot_x = onehot_x.reshape(onehot_x.shape[0], -1)
    print('input size is',onehot_x.shape )
    main(args, data_partitions, onehot_x, y)
    
elif args.model_name == 'only_extended_features':
    print('we are running' + args.model_name + '_model')
    args.output_path = './output/XG_boost/only_extended_features'
    onehot_x = x_extended_f
    print('input size is',onehot_x.shape )
    main(args, data_partitions, onehot_x, y)

elif args.model_name == 'protospacer_extended_features':
    print('we are running' + args.model_name + '_model')
    args.output_path = './output/XG_boost/protospacer_extended_features'
    onehot_x = one_hot_encode(x_protospacer)
    onehot_x = onehot_x.reshape(onehot_x.shape[0], -1)
    onehot_x = np.concatenate((onehot_x, x_extended_f), axis=1)
    print('input size is',onehot_x.shape )
    main(args, data_partitions, onehot_x, y)

else:
    print('please specify the model name from the following set:')
    print('protospacer_20', 'protospacer_80','only_extended_features', 'protospacer_extended_features')
    
    
