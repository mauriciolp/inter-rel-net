import numpy as np
import argparse, sys, os, time, glob

import pandas as pd

#%% Functions
def load_args():
    ap = argparse.ArgumentParser(
        description='Print train statistics stored on the csv log files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('summary_dirs',
        help='Directory with the train summary information.',
        nargs='*',
        type=str)
    
    # Optional arguments
    ap.add_argument('-c','--criteria',
        help="criteria for picking the best epoch",
        choices=['val_acc', 'val_loss'])
    ap.add_argument('-u','--update',
        help="update summary file (overwrite)",
        action='store_true')
    ap.add_argument('-s','--seqs-eval',
        help="evaluation over sequences average (sample 'all')",
        action='store_true')
        
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_false')
        # action='store_true')
    
    args = ap.parse_args()
    
    return args

def read_runs(summary_dir, criteria=None, seqs_eval=False):
    if not seqs_eval: # Central frames only
        summary_filename = '/summary.csv'
        run_filename = '/fit_history.csv'
        log_filename = '/training.log'
    else:
        summary_filename = '/summary-pooled_val_acc.csv'
        run_filename = '/pooled_val_acc.csv'
        log_filename = run_filename
    
    hist_paths = glob.glob(summary_dir+'/fold_*'+summary_filename)
    hist_paths.sort()
    
    folds_with_summary = [ hist_path.split('/')[-2] for hist_path in hist_paths ]
    
    reruns_dirs = glob.glob(summary_dir+'/rerun_*/') + glob.glob(summary_dir+'/fold_*/')
    reruns_dirs.sort()
    
    reruns_dirs = [ r for r in reruns_dirs 
        if not any( fold in r for fold in folds_with_summary) ]
    
    for rerun_dir in reruns_dirs:
        if 'rerun' in rerun_dir:
            rerun_hist = glob.glob(rerun_dir + run_filename)
            if rerun_hist == []:
                rerun_hist = glob.glob(rerun_dir + log_filename)
                if rerun_hist != []:
                    if os.stat(rerun_hist[0]).st_size != 0: # log is not empty
                        print("WARNING: Training not finished for '{}', using '{}'".format(
                            rerun_dir.split('/')[-2], log_filename))
                    else:
                        rerun_hist = []
        else:
            print("WARNING: Training not finished for '{}', using '{}'".format(
                            rerun_dir.split('/')[-2], log_filename))
            fold_reruns_dirs = glob.glob(rerun_dir+'/rerun_*/')
            fold_reruns_dirs.sort()
            
            fold_best_epochs = read_runs(rerun_dir, criteria=criteria, seqs_eval=seqs_eval)
            if fold_best_epochs != []:
                fold_df = pd.concat(fold_best_epochs, axis=1, sort=False).T.reset_index(drop=True)
                best_rerun = fold_reruns_dirs[fold_df.val_acc.idxmax()]
                best_hist = glob.glob(best_rerun + run_filename) + glob.glob(best_rerun + log_filename)
                
                rerun_hist = [ best_hist[0] ]
            else:
                rerun_hist = []
        hist_paths += rerun_hist
        
    hist_paths.sort()
    
    best_epochs = []
    for rerun_idx, hist_path in enumerate(hist_paths):
        hist_df = pd.read_csv(hist_path)
        
        if criteria.endswith('loss'):
            best_epoch = hist_df.loc[hist_df[criteria].idxmin()]
        elif criteria.endswith('acc'):
            if not seqs_eval: # Central frames only
                sorted_hist_df = hist_df.sort_values([criteria, 'val_loss'], 
                    ascending=[False, True])
            else:
                sorted_hist_df = hist_df.sort_values([criteria], ascending=[False])
            best_epoch = sorted_hist_df.iloc[0]
            
        best_epochs.append(best_epoch)
    
    return best_epochs

def pretty_print_stats(stats_df, short_version=False, seqs_eval=False):
    acc_tpl = "{:.2%}".format
    loss_tpl = "{:.4f}".format
    epoc_tpl = "{:.0f}".format
    print_order = ['acc','loss','val_acc','val_loss','epoch']
    
    if short_version:
        print_order = ['val_acc','val_loss']
        stats_df = stats_df[stats_df.index == 'mean']
    
    if seqs_eval:
        print_order = ['val_acc']
    
    print(stats_df[print_order].to_string(
        formatters=[acc_tpl,loss_tpl,acc_tpl,loss_tpl,epoc_tpl]))

def print_train_stats(summary_dir, criteria=None, update=False, seqs_eval=False):
    if not seqs_eval: # Central frames only
        summary_filename = '/summary.csv'
    else:
        summary_filename = '/summary-pooled_val_acc.csv'
    
    if not os.path.exists(summary_dir+summary_filename):
        criteria = 'val_acc'

    if criteria is None:
        summary_df = pd.read_csv(summary_dir+summary_filename, index_col=False)
    else:
        best_epochs = read_runs(summary_dir, criteria=criteria, seqs_eval=seqs_eval)
        if best_epochs == []:
            print("ERROR: Unable to read summary.csv or fit_history.csv")
            return
        summary_df = pd.concat(best_epochs, axis=1, sort=False).T.reset_index(drop=True)
        
    summary_df.drop('mean', errors='ignore', inplace=True)
    summary_df.drop('std', errors='ignore', inplace=True)
    
    if update:
        summary_df.to_csv(summary_dir+summary_filename)
    
    max = summary_df.loc[summary_df['val_acc'].idxmax()].rename('max')
    mean = summary_df.mean().rename('mean')
    std = summary_df.std().rename('std')
    stats = pd.concat([max,mean,std], axis=1).T

    pretty_print_stats(summary_df, seqs_eval=seqs_eval)
    print(stats[['val_acc']].T.to_string(float_format="{:.1%}".format))

def print_protocol_stats(summary_dir, criteria=None):
    pass

def print_train_stats_all(summary_dirs, **kwargs):
    for summary_dir in summary_dirs:
        if summary_dir.endswith('/hp_search/'):
            continue
        print('********** '+summary_dir+' **********')
        print_train_stats(summary_dir, **kwargs)
    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting HyperParameter Search - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    print_train_stats_all(**args)

    print('\n> Finished HyperParameter Search -', time.asctime( time.localtime(time.time()) ))
