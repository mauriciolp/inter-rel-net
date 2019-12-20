import configparser, os, glob
from ast import literal_eval
import pandas as pd

def find_best_weights(base_path, criteria='val_loss', verbose=0):
    from misc.print_train_stats import pretty_print_stats
    rerun_paths = glob.glob(base_path+'/rerun_*/')
    rerun_paths += glob.glob(base_path+'/fold_*/')
    rerun_paths.sort()
    
    best_epochs = []
    for rerun_path in rerun_paths:
        rerun_df = pd.read_csv(rerun_path + 'fit_history.csv')
        rerun_df['path'] = rerun_path
        
        if criteria.endswith('loss'):
            best_epoch = rerun_df.loc[rerun_df[criteria].idxmin()]
        elif criteria.endswith('acc'):
            sorted_rerun_df = rerun_df.sort_values([criteria, 'val_loss'], 
                ascending=[False, True])
            best_epoch = sorted_rerun_df.iloc[0]
        
        best_epochs.append(best_epoch)
    
    summary_df = pd.concat(best_epochs, axis=1).T.reset_index(drop=True)
    summary_df = summary_df.astype({criteria: 'float'})
    
    if criteria.endswith('loss'):
        best_rerun = summary_df.loc[summary_df[criteria].idxmin()]
        weights_path = best_rerun.path + 'relnet_weights.hdf5'
    elif criteria.endswith('acc'):
        sorted_summary_df = summary_df.sort_values([criteria, 'val_loss'], 
            ascending=[False, True])
        best_rerun = sorted_summary_df.iloc[0]
        weights_path = best_rerun.path + 'relnet_weights-val_acc.hdf5'
    
    if verbose > 0:
        print("Best weights stats:")
        best_rerun_df = best_rerun.to_frame().transpose()
        best_rerun_df = best_rerun_df.astype(
            {'acc': 'float', 'loss': 'float', 'val_acc': 'float', 'val_loss': 'float'})
        pretty_print_stats(best_rerun_df)

    return weights_path

def read_config(config_filepath, fusion=False):
    def unstringify_dict(d):
        return dict((k,literal_eval(v)) for k,v in d.items())
    
    config = configparser.ConfigParser()
    with open(config_filepath) as config_file:
        config.read_file(config_file)
    
    if not fusion:
        data_kwargs = unstringify_dict(config['data'])
        model_kwargs = unstringify_dict(config['model'])
        kwargs = [data_kwargs, model_kwargs]
    else:
        fusion_kwargs = unstringify_dict(config['fusion'])
        kwargs = [fusion_kwargs]
    train_kwargs = unstringify_dict(config['train'])
    
    kwargs.append(train_kwargs)
    
    return tuple(kwargs)
