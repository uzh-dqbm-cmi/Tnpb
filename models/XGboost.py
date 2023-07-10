## functions that defines the XGboost model

import xgboost as xgb
import argparse
from src.utils import performance_statistics
from src.utils import ReaderWriter, create_directory, perfmetric_report_cont
from src.utils import compute_spearman_corr, compute_pearson_corr
from src.utils import one_hot_encode
from sklearn.model_selection import KFold
import itertools
import numpy as np
import csv
import sys

def main(args,data_partitions, x, y):
    N_innerCV  = args.inner_cv
    inner_cv = KFold(n_splits=N_innerCV, shuffle=True, random_state=0)
    other_params = {'subsample': 1.0,'min_child_weight': 10, 'gamma': 2.5,'colsample_bytree': 0.4,'alpha': 0, 'learning_rate':0.1}
    param_vals = {'n_estimators': [100,300,600],'max_depth': [8,10,12], 'reg_lambda': [1,10,100]}
    hyper_param = param_vals.values()
    combinations = list(itertools.product(*hyper_param))


    #3.train
    test_error = []
    test_spearm_corr = []
    test_pearson_corr = []
    # outer cross validation
    for i in range(len(data_partitions)):
        print(f"Outer Fold {i}:")
        print(f"   number of Training instances:", len(data_partitions[i]['train_index']))
        print(f"   number of test instances:", len(data_partitions[i]['test_index']))
        error = np.empty([len(data_partitions),len(combinations)])
        ## inner cross validation for hyper param tuning
        for j, (train_index_new, val_index) in enumerate(inner_cv.split(data_partitions[i]['train_index'])):
            print(f"Inner Fold {j}:")
            print(f"   number of Training instances:", len(train_index_new))
            print(f"   number of validation instances:", len(val_index))
            for index, c in enumerate(combinations):
                model = xgb.XGBRegressor(n_estimators=c[0], max_depth=c[1], reg_lambda=c[2], **other_params)
                model.fit(x[train_index_new,:], y[train_index_new], eval_set = [(x[val_index,:], y[val_index])], early_stopping_rounds=10) 
                ## train error:
                yhat_train = model.predict(x[train_index_new,:])
                train_error = np.absolute(y[train_index_new]-yhat_train).mean()
                #print('training error:', train_error)
                yhat_val = model.predict(x[val_index,:])
                #error = MAE(y[val_index], yhat)
                val_error = np.absolute(y[val_index]-yhat_val).mean()
                #print('validation error:', val_error)
                error[j,index] =val_error

        #get the best paramerter values       
        best_param_indx = error.mean(axis = 0).argmin()
        best_param = combinations[best_param_indx]
        print('best parameters:', best_param)
        #train the model with the best parameters
        model = xgb.XGBRegressor(n_estimators=best_param[0], max_depth=best_param[1], reg_lambda=best_param[2], **other_params)
        #predict on the test set 
        model.fit(x[data_partitions[i]['train_index'],:], y[data_partitions[i]['train_index']])
        yhat_test = model.predict(x[data_partitions[i]['test_index'],:])
        t_error = np.absolute(y[data_partitions[i]['test_index']]-yhat_test).mean()
        ## creat directory to save the model
        path_to_save = create_directory('run_'+ str(i), args.output_path)
        ## save the best model for this run
        model.save_model(path_to_save +'/best.model')
        ReaderWriter.write_csv(path_to_save+'/prediction', y[data_partitions[i]['test_index']],yhat_test, header=['true','predicted'])
        #ReaderWriter.write_log()
        test_error.append(t_error)
        test_spearm_corr.append(compute_spearman_corr(yhat_test,y[data_partitions[i]['test_index']]))
        test_pearson_corr.append(compute_pearson_corr(yhat_test,y[data_partitions[i]['test_index']]))

        perfmetric_report_cont(yhat_test, y[data_partitions[i]['test_index']], path_to_save +'/test_performance.csv' )
        ReaderWriter.write_params(path_to_save,param_vals,  best_param, other_params)
        #print('test error:', test_error)

    performance_statistics(len(data_partitions), args.output_path)
