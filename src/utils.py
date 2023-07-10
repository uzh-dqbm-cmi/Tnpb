##helper functions are defined here
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import pickle
import csv
import scipy
import json
import torch
from sklearn.metrics import mean_absolute_error,mean_squared_error
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


def perfmetric_report_cont(pred_score, ref_score,  outlog):  
    lsep = "\n"
    #report = "best model performance report" + lsep
    spearman_corr, pvalue_spc = compute_spearman_corr(pred_score, ref_score)
    pearson_corr, pvalue_prc = compute_pearson_corr(pred_score, ref_score)
    MAE_score = np.absolute(ref_score-pred_score).mean()
    
    # Write correlation scores and p-values to CSV file
    with open(outlog, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Correlation Type', 'Correlation Score', 'P-Value'])
        writer.writerow(['Spearman', spearman_corr, pvalue_spc])
        writer.writerow(['Pearson', pearson_corr, pvalue_prc])
        writer.writerow(['MAE_score', MAE_score, 0])
        
def perfmetric_report_best_epoch(pred_score, ref_score, epoch,  outlog):  
    lsep = "\n"
    #report = "best model performance report" + lsep
    spearman_corr, pvalue_spc = compute_spearman_corr(pred_score, ref_score)
    pearson_corr, pvalue_prc = compute_pearson_corr(pred_score, ref_score)
    MAE_score = np.absolute(ref_score-pred_score).mean()
    
    # Write correlation scores and p-values to CSV file
    with open(outlog, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best epoch', epoch, ''])
        writer.writerow(['Correlation Type', 'Correlation Score', 'P-Value'])
        writer.writerow(['Spearman', spearman_corr, pvalue_spc])
        writer.writerow(['Pearson', pearson_corr, pvalue_prc])
        writer.writerow(['MAE_score', MAE_score, 0])

def perfmetric_report_per_epoch(pred_score, ref_score, epoch, outlog):  
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    report += "Regression report on all events:" + lsep
    #report = "best model performance report" + lsep
    
    spearman_corr, pvalue_spc = compute_spearman_corr(pred_score, ref_score)
    pearson_corr, pvalue_prc = compute_pearson_corr(pred_score, ref_score)
    MAE_score = np.absolute(ref_score-pred_score).mean()
    
    report += "MAE:" + lsep
    #MAE_score = mean_absolute_error(ref_score, pred_score)
    report += str(MAE_score) + lsep

    report += "MSE:" + lsep
    MSE = mean_squared_error(ref_score, pred_score)
    report += str(MSE) + lsep

    report += "Spearman coefficient:" + lsep
    report += str(spearman_corr) + lsep
    
    report += "Pearson coefficient:" + lsep
    report += str(pearson_corr) + lsep
    
    
    report += "-"*30 + lsep
    ReaderWriter.write_log(report, outlog)
    

def perfmetric_report_regression(pred_target, ref_target, epoch, outlog):

    # print(ref_target.shape)
    # print(pred_target.shape)
    #
    # print("ref_target \n", ref_target)
    # print("pred_target \n", pred_target)
    
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    report += "Regression report on all events:" + lsep
    #report += str(classification_report(ref_target, pred_target)) + lsep

    
    report += "MAE:" + lsep
    MAE = mean_absolute_error(ref_target, pred_target)
    #MAE = (np.absolute(pred_target-ref_target)).mean()
    report += str(MAE) + lsep

    report += "MSE:" + lsep
    MSE = mean_squared_error(ref_target, pred_target)
    report += str(MSE) + lsep

    report += "Correlation coefficient:" + lsep
    #print(ref_target.shape)
    #print(pred_target.shape)
    correlation = np.corrcoef(ref_target.squeeze(), pred_target)[0,1]
    report += str(correlation) + lsep

    spearman_corr, pvalue_spc = compute_spearman_corr(pred_target, ref_target)
    pearson_corr, pvalue_prc = compute_pearson_corr(pred_target, ref_target)

    report += "Spearman coefficient:" + lsep
    report += str(spearman_corr) + lsep
    
    report += "Pearson coefficient:" + lsep
    report += str(pearson_corr) + lsep
    
    report += "-"*30 + lsep
    # (best_epoch_indx, binary_f1, macro_f1, aupr, auc)
    modelscore = RegModelScore(epoch,  MAE, MSE, (spearman_corr, pearson_corr))
    ReaderWriter.write_log(report, outlog)
    return modelscore


class RegModelScore:
    def __init__(self, best_epoch_indx, MAE, MSE, correlation):
        self.best_epoch_indx = best_epoch_indx
        if MAE == 0.0 and MSE == 0.0:
            
            self.MAE = np.inf
            self.MSE = np.inf
            self.correlation = correlation
        else:
            self.MAE = MAE
            self.MSE = MSE
            self.correlation = correlation
        

    def __repr__(self):
        desc = " best_epoch_indx:{}\n MAE:{}\n MSE:{}\n correlation:{}\n" \
               "".format(self.best_epoch_indx, self.MAE, self.MSE, self.correlation)
        return desc

    
def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)
    

def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Normalize X to have feature values follow N(0,1)
    X_normalized = (X - mean) / std
    return X_normalized

def performance_statistics(num_runs, output_path):
    results = []
    for i in range(num_runs):
        df = pd.read_csv(output_path + '/run_'+ str(i) + '/test_performance.csv')
        results.append(np.array(df))
        
    results = np.array(results)
    spearman_corr = results[:, 0, 1].astype(float)
    spearman_p = results[:, 0, 2].astype(float)
    preason_corr = results[:, 1, 1].astype(float)
    preason_p = results[:, 1, 2].astype(float)
    MAE_all = results[:, 2, 1].astype(float)
    
    with open(output_path + '/output_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Correlation Type', 'Correlation Score mean','Correlation Score std', 'P-Value mean', 'P-Value std'])
        writer.writerow(['Spearman', spearman_corr.mean(), spearman_corr.std(), spearman_p.mean(),spearman_p.std()])
        writer.writerow(['pearson', preason_corr.mean(), preason_corr.std(), preason_p.mean(),preason_p.std()])
        writer.writerow(['MAE_score', MAE_all.mean(), MAE_all.std(), 0,0])

def compute_spearman_corr(pred_score, ref_score):
    # return scipy.stats.kendalltau(pred_score, ref_score)
    return scipy.stats.spearmanr(pred_score, ref_score)

def compute_pearson_corr(pred_score, ref_score):
    # return scipy.stats.kendalltau(pred_score, ref_score)
    return scipy.stats.pearsonr(pred_score, ref_score)

def get_char(seq):
    """split string int sequence of chars returned in pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)


def one_hot_encode(x):
    one_hot_x = np.zeros((x.shape[0], x.shape[1], 4), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot_x[i,j,x[i,j]] = 1.0
    
    return one_hot_x


def outer_cross_val(N, k_fold ):
    index = np.arange(N)
    np.random.seed(seed=42)
    np.random.shuffle(index)
    outer_cv = KFold(n_splits= k_fold, shuffle=False)
    data_partitions = {}
    for i, (train_index, test_index) in enumerate(outer_cv.split(index)):
        data_partitions[i] = {'train_index': train_index,
                                    'test_index': test_index}
        print(test_index)
        
        print("run_num:", i)
        print('train data:{}/{} = {}'.format(len(train_index), N, len(train_index)/N))
        print('test data:{}/{} = {}'.format(len(test_index), N, len(test_index)/N))

    return data_partitions


class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
 
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            print(line)
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line
   
    @staticmethod
    def write_csv(file_name, y_true,y_pred, header=None):
        with open(file_name+'.csv', mode='w', newline='') as file:

            # Create a CSV writer object
            writer = csv.writer(file)
            if header is not None:
                writer.writerow(header)
            for i in range(len(y_true)):
                writer.writerow([y_true[i], y_pred[i]])
            
            
    @staticmethod
    def write_params(file, params, best_param, other_params):
        keys_list = list(params.keys())
        best_params_dic = dict(zip(keys_list, best_param))
        merged_dict = {'best_params': best_params_dic, 'params': params, 'fixed_params': other_params}
        with open(file +'/params.json', 'w') as f:
            json.dump(merged_dict, f)
    

def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)



def plot_loss(epoch_loss_avgbatch, wrk_dir):
    dsettypes = epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9, 6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r')
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()
 

def build_regression_df( true_class, pred_class):
    df_dict = {
            #'id': ids,
            'true_class': true_class,
            'pred_class': pred_class
        }
   
    predictions_df = pd.DataFrame(df_dict)
    #predictions_df.set_index('id', inplace=True)
    return predictions_df  

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)
        

### print out final results
def print_eval_results(tdir, num_runs=5):
    pearson_corr = {}
    spearman_corr = {}
    MAE_score = {}
    for i in range(num_runs):
        print('run_num:', i)

        file = tdir + '/run_'+ str(i)
        df = pd.read_csv(file + '/predictions_test.csv')

        spearman_corr[i], pvalue_spc = scipy.stats.spearmanr(df['pred_class'], df['true_class'])
        pearson_corr[i], pvalue_prc = scipy.stats.pearsonr(df['pred_class'], df['true_class'])
        MAE_score[i] = np.absolute(df['pred_class'] - df['true_class']).mean()
        print('spearman:', spearman_corr[i])
        print('pearson:', pearson_corr[i])

    print('over all pearson correlation mean', np.array(list(pearson_corr.values())).mean(),"standard deviation:", np.array(list(pearson_corr.values())).std())
    print('over all spearman correlation mean', np.array(list(spearman_corr.values())).mean(),"standard deviation:", np.array(list(spearman_corr.values())).std())
    print('over all mean absolute error', np.array(list(MAE_score.values())).mean(),"standard deviation:", np.array(list(MAE_score.values())).std())

def compute_eval_results_df(tdir, num_runs=5):

    num_metrics = 3
    metric_names = ('spearman', 'pearson', 'MAE')
    perf_dict = [{} for i in range(num_metrics)]

    for i in range(num_runs):
        run_name = f'run_{i}'
        print('run_name:', run_name)

        file = tdir + '/run_'+ str(i)
        df = pd.read_csv(file + '/predictions_test.csv')

        spearman_corr, pvalue_spc = scipy.stats.spearmanr(df['pred_class'], df['true_class'])
        pearson_corr, pvalue_prc = scipy.stats.pearsonr(df['pred_class'], df['true_class'])
        MAE_score = np.absolute(df['pred_class'] - df['true_class']).mean()
        
        perf_dict[0][run_name] = spearman_corr
        perf_dict[1][run_name] = pearson_corr
        perf_dict[2][run_name] = MAE_score


    perf_df_lst = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf, index=[f'{metric_names[i]}'])
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df_lst.append(all_perf_df.sort_values('mean', ascending=False))
    return pd.concat(perf_df_lst, axis=0)

def plot_y_distrib_acrossfolds(dpartitions, y, opt='separate_folds'):

    if opt == 'separate_dsettypes':
        fig, axs = plt.subplots(figsize=(9,11), 
                                nrows=3, 
                                constrained_layout=True)
        axs = axs.ravel()
        for run_num in range(len(dpartitions)):
            counter = 0
            for dsettype in ['train', 'validation', 'test']:
                curr_ax = axs[counter]
                ids = dpartitions[run_num][dsettype]
                curr_ax.hist(y[ids], alpha=0.4, label=f"{dsettype}_run{run_num}")
                counter+=1
                curr_ax.legend()
    elif opt == 'separate_folds':
        fig, axs = plt.subplots(figsize=(9,11), 
                                nrows=5, 
                                constrained_layout=True)
        axs = axs.ravel()
        for run_num in range(len(dpartitions)):
            curr_ax = axs[run_num]
            for dsettype in ['train', 'validation', 'test']:
                ids = dpartitions[run_num][dsettype]
                curr_ax.hist(y[ids], alpha=0.4, label=f"{dsettype}_run{run_num}")
                curr_ax.legend()


def concat_strings(row):
    return ' '.join(row)

def remove_spaces(s):
    return s.replace(' ', '')
