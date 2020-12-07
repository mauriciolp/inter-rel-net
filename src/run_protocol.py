import numpy as np
import argparse, sys, os, time
import pandas as pd
from shutil import copyfile

from datasets import UT, SBU, NTU, NTU_V2

from train_rn import train_rn, train_fused_rn
from train_temp_rn import train_temp_rn, train_fused_temp_rn

from predict_rn import predict_rn_seq, predict_fused_rn_seq
from misc.print_train_stats import pretty_print_stats
from misc.utils import read_config, find_best_weights

#%% Functions
def load_args():
    ap = argparse.ArgumentParser(
        description='Run protocol for the specified dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('experiment_name',
        help='Experiment name.',
        type=str)
    ap.add_argument('config_filepath',
        help='Filepath to hyper-parameters configuration .',
        type=str)
    ap.add_argument('dataset_name',
        help='Name of the dataset to run the protocol.',
        choices=['UT-1','UT-2','SBU','NTU','NTU-V2'],
        type=str)
    
    # Optional arguments
    ap.add_argument('-n','--num-reruns',
        help="number of reruns per fold",
        type=int,
        default=4)
    ap.add_argument('-f','--folds',
        help="list of folds to run",
        nargs='*')
    ap.add_argument('-F', '--fusion-mode',
        help='Mode to perform the fusion.',
        choices=['middle','late'],
        type=str)
    ap.add_argument('-s','--seqs-eval',
        help="evaluation over sequences average (sample 'all')",
        action='store_true')
    ap.add_argument('-t','--temp-rn',
        help="use temporal_rn",
        action='store_true')
    ap.add_argument('-v','--verbose',
        help="verbose level",
        type=int,
        default=0)
    ap.add_argument('-b','--batch-size',
        help="batch size for train or predict",
        type=int,
        default=64)
        
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def parse_fit_history(fit_history):
    hist_df = pd.DataFrame.from_dict(fit_history.history)
    hist_df['epoch'] = hist_df.index + 1
    hist_df['epoch'] = hist_df['epoch'].astype(int)
    
    return hist_df

def run_protocol(experiment_name, config_filepath, dataset_name, num_reruns=2,
        folds=None, verbose=0, batch_size=64, seqs_eval=False, fusion_mode=None,
        temp_rn=False):
    if fusion_mode is None:
        data_kwargs, model_kwargs, train_kwargs = read_config(config_filepath)
    else:
        fusion_kwargs, train_kwargs = read_config(config_filepath, fusion=True)
        weights_base_paths = ['/'.join(weights_filepath.split('/')[:-3])
            for weights_filepath in fusion_kwargs['weights_filepaths'] ]
        fusion_kwargs.pop('weights_filepaths', None)
        criteria = fusion_kwargs.pop('criteria', 
            'val_loss' if fusion_mode == 'middle' else 'val_acc')
        data_kwargs, model_kwargs, _ = read_config(
            fusion_kwargs['config_filepaths'][0])
    
    if dataset_name == 'UT-1':
        dataset_folds = UT.get_folds(setid=1)
        dataset_codename = 'UT'
    elif dataset_name == 'UT-2':
        dataset_folds = UT.get_folds(setid=2)
        dataset_codename = 'UT'
    elif dataset_name == 'SBU':
        dataset_folds = SBU.get_folds()
        dataset_codename = 'SBU'
    elif dataset_name == 'NTU':
        dataset_folds = NTU.get_folds()
        dataset_codename = 'NTU'
    elif dataset_name == 'NTU-V2':
        dataset_folds = NTU_V2.get_folds()
        dataset_codename = 'NTU-V2'
    
    if folds is not None:
        if dataset_codename.startswith('NTU'):
            dataset_folds = folds
        else:
            dataset_folds = list(map(int, folds))
    
    base_path = 'models/{}/{}'.format(dataset_name, experiment_name)
    print("Running for dataset:", dataset_name)
    print("Running for folds:", dataset_folds)
    print("Experiment name:", experiment_name)
    print("Experiment folder:", base_path)
    print("Number of reruns per dataset fold:", num_reruns)
    
    if os.path.exists(base_path):
        print("Resuming run protocol...")
    else:
        os.makedirs(base_path)
    
    if config_filepath != os.path.join(base_path, 'parameters.cfg'):
        copyfile(config_filepath, os.path.join(base_path, 'parameters.cfg'))
    
    print("***** Experiment parameters *****")
    print("Configuration file:", config_filepath)
    print("\t Skeleton info")
    for key, value in data_kwargs.items():
        print("\t > {}: {}".format(key, value))
    if fusion_mode is None:
        print("\t Model info")
        for key, value in model_kwargs.items():
            print("\t > {}: {}".format(key, value))
    else:
        print("\t Fusion info")
        print("\t > fusion mode:", fusion_mode)
        for key, value in fusion_kwargs.items():
            print("\t > {}: {}".format(key, value))
        print("\t > Criteria for weights:", criteria)
    print("\t > Using Temporal RN:", temp_rn)
    print("\t Training options")
    for key, value in train_kwargs.items():
        print("\t > {}: {}".format(key, value))
    print("\t > Batch Size:", batch_size)
    print("\t Evaluation options")
    print("\t > Pool average from videos sequences:", seqs_eval)
    
    
    
    if seqs_eval:
        fold_results_seqs = []
        data_kwargs_seqs = data_kwargs.copy()
        
        data_kwargs_seqs.setdefault('seq_step', data_kwargs['timesteps']//2)
        data_kwargs_seqs['flat_seqs'] = False
        print("\t > Step between sequences:", data_kwargs_seqs['seq_step'])
    
    fold_results = []
    for dataset_fold in dataset_folds:
        print("Running for fold:", dataset_fold)
        fold_path = base_path+'/fold_{}/'.format(dataset_fold)
        
        if fusion_mode == 'middle':  
            fold_weights_filepaths = [ 
                find_best_weights(weights_base_path+'/fold_{}/'.format(dataset_fold), criteria=criteria) 
                for weights_base_path in weights_base_paths ]
            print('> fold weights filepaths:', fold_weights_filepaths)
        
        if seqs_eval: reruns_val_acc_seqs = []
        best_epochs = []
        for rerun_idx in range(num_reruns):
            output_path = fold_path+'/rerun_{}'.format(rerun_idx)
            
            if os.path.exists(output_path+'/fit_history.csv'):
                hist_df = pd.read_csv(output_path+'/fit_history.csv')
            else:
                if fusion_mode is None:
                    if not temp_rn:
                        fit_history = train_rn(output_path, dataset_codename, 
                            dataset_fold=dataset_fold, **train_kwargs,
                            data_kwargs=data_kwargs, model_kwargs=model_kwargs, 
                            batch_size=batch_size,  gpus=1, verbose=verbose)
                    else:
                        fit_history = train_temp_rn(output_path, dataset_codename, 
                            dataset_fold=dataset_fold, **train_kwargs,
                            data_kwargs=data_kwargs, model_kwargs=model_kwargs, 
                            batch_size=batch_size,  gpus=1, verbose=verbose)
                elif fusion_mode == 'middle':
                    if not temp_rn:
                        fit_history = train_fused_rn(output_path, dataset_codename, 
                            dataset_fold=dataset_fold, 
                            weights_filepaths=fold_weights_filepaths,
                            **fusion_kwargs, **train_kwargs,
                            batch_size=batch_size,  gpus=1, verbose=verbose)
                    else:
                        fit_history = train_fused_temp_rn(output_path, 
                            dataset_codename, dataset_fold=dataset_fold,
                            weights_filepaths=fold_weights_filepaths,
                            **fusion_kwargs, **train_kwargs,
                            batch_size=batch_size,  gpus=1, verbose=verbose)
                
                hist_df = parse_fit_history(fit_history)
                hist_df.to_csv(output_path+'/fit_history.csv', index=False)
            
            sorted_hist_df = hist_df.sort_values(['val_acc', 'val_loss'], 
                ascending=[False, True])
            best_epoch = sorted_hist_df.iloc[0]
            print("> {}: ACC: {:.2%} Loss: {:.4f} - MAX val_ACC: {:.2%} val_Loss: {:.4f}".format(
                rerun_idx, best_epoch.acc, best_epoch.loss, 
                best_epoch.val_acc, best_epoch.val_loss))
            best_epochs.append(best_epoch)
            
            if seqs_eval:
                rerun_path = output_path
                pooled_val_acc_filepath = rerun_path+'/pooled_val_acc.csv'
                if os.path.exists(pooled_val_acc_filepath):
                    val_acc_df = pd.read_csv(pooled_val_acc_filepath)
                else:
                    weights_path = rerun_path+'/relnet_weights-val_acc.hdf5'
                    
                    if fusion_mode is None:
                        val_acc = predict_rn_seq(weights_path, dataset_codename, 
                            dataset_fold=dataset_fold, batch_size=batch_size, verbose=verbose,
                            data_kwargs_seqs=data_kwargs_seqs, model_kwargs=model_kwargs, 
                            return_acc=True)
                    elif fusion_mode == 'middle':
                        val_acc = predict_fused_rn_seq(weights_path, dataset_codename, 
                            dataset_fold=dataset_fold, batch_size=batch_size, verbose=verbose,
                            **fusion_kwargs, return_acc=True)
                    else:
                        print("Seqs eval not implemented for fusion_mode:", fusion_mode)
                        val_acc = -1
                    
                    val_acc_df = pd.DataFrame.from_dict({'val_acc': [val_acc]})
                    val_acc_df.to_csv(pooled_val_acc_filepath, index=False)
            
                val_acc_series = val_acc_df.iloc[0]
                print("> Pooled Seqs val_ACC: {:.2%}".format(val_acc_series.val_acc))
                reruns_val_acc_seqs.append(val_acc_series)
        
        summary_fold = pd.concat(best_epochs, axis=1).transpose().reset_index(drop=True)
        summary_fold.index.name = 'rerun'
        summary_fold.to_csv(fold_path+'/summary.csv')
        
        sorted_summary_fold = summary_fold.sort_values(['val_acc', 'val_loss'], 
            ascending=[False, True])
        best_fold = sorted_summary_fold.iloc[0]
        fold_results.append(best_fold)
        
        if seqs_eval:
            summary_fold_seqs = pd.concat(reruns_val_acc_seqs, axis=1).transpose().reset_index(drop=True)
            summary_fold_seqs.index.name = 'rerun'
            summary_fold_seqs.to_csv(fold_path+'/summary-pooled_val_acc.csv')
            
            best_fold_seqs = summary_fold_seqs.loc[summary_fold_seqs.val_acc.idxmax()]
            fold_results_seqs.append(best_fold_seqs)
    
    summary_df = pd.concat(fold_results, axis=1).transpose().reset_index(drop=True)
    summary_df.index.name = 'fold'
    summary_df.to_csv(base_path+'/summary.csv')
    
    mean = summary_df.mean().rename('mean')
    std = summary_df.std().rename('std')
    stats = pd.concat([mean, std], axis=1).T
    summary_df = pd.concat([summary_df, stats])
    
    print("+++++ Summarized results:")
    print("\nSummary for Dataset, Folds, Experiment:")
    print("> {} | {} | {}".format(dataset_codename, dataset_folds, experiment_name))
    pretty_print_stats(summary_df)
    print("Val ACC Mean: {:.2%} Std: {:.4f}".format(mean.val_acc, std.val_acc))
    
    if seqs_eval:
        summary_seqs = pd.concat(fold_results_seqs, axis=1).transpose().reset_index(drop=True)
        summary_seqs.index.name = 'fold'
        summary_seqs.to_csv(base_path+'/summary-pooled_val_acc.csv')
        
        print("+++++ Seqs eval results:")
        mean = summary_seqs.mean().rename('mean')
        std = summary_seqs.std().rename('std')
        stats = pd.concat([mean, std], axis=1).T
        summary_seqs = pd.concat([summary_seqs, stats])
        print(summary_seqs.to_string(formatters=["{:.2%}".format]))
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting Run Protocol - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    run_protocol(**args)

    print('\n> Finished Run Protocol -', time.asctime( time.localtime(time.time()) ))
