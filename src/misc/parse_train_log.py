import numpy as np
import argparse, sys, os, time, glob

import pandas as pd

from print_train_stats import pretty_print_stats

import matplotlib.pyplot as plt

os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-mauricio"

#%% Functions
def load_args():
    ap = argparse.ArgumentParser(
        description='Print train statistics stored on the csv files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('train_log_filepath',
        help='Filepath to train log.',
        type=str)
    
    # Optional arguments
    ap.add_argument('-c','--criteria',
        help="criteria for picking the best epoch",
        default='val_acc',
        choices=['val_acc', 'val_loss'])
    ap.add_argument('-p','--plot',
        help="plot accuracies and losses curves",
        action='store_true')
    ap.add_argument('-s','--skip-epochs',
        help="number of epochs to skip when plotting",
        type=int,
        default=50)
    ap.add_argument('-t','--trunc-epochs',
        help="number of epochs to trunc at the end when ranking and plotting",
        type=int)
    
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_false')
    
    args = ap.parse_args()
    
    return args

def parse_train_log(train_log_filepath, criteria='val_acc', top_k=10, plot=True,
        skip_epochs=50, trunc_epochs=None):
    if not train_log_filepath.endswith(('.log','.csv')):
        train_log_filepath = os.path.join(train_log_filepath, 'training.log')
    train_log_df = pd.read_csv(train_log_filepath)
    
    if trunc_epochs is not None:
        train_log_df = train_log_df[:trunc_epochs]
    
    if criteria.endswith('loss'):
        sorted_log_df = train_log_df.sort_values([criteria, 'epoch'], 
            ascending=[True, False])
    elif criteria.endswith('acc'):
        sorted_log_df = train_log_df.sort_values([criteria, 'epoch'], 
            ascending=[False, False])
    
    pretty_print_stats(sorted_log_df.head(top_k))
    
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        train_log_df[skip_epochs:].plot.line(x='epoch', y=['acc','val_acc'], ax=axes[0])
        train_log_df[skip_epochs:].plot.line(x='epoch', y=['loss','val_loss'], ax=axes[1])
        
        plt.show()

#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting HyperParameter Search - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    parse_train_log(**args)

    print('\n> Finished HyperParameter Search -', time.asctime( time.localtime(time.time()) ))
