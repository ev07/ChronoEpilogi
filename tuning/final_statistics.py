import pandas as pd
import sys
sys.path.append("../")
from tuning.first_wave_main import setup_dataset
from tuning.routines import full_experiment
import numpy as np
    
    
def val_all_methods_best_configuration(filename, metric, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    compute the best configuration according to some metric.
    
    params:
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        
    """
    
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    
    results_dir = "./results/optuna/test_stats/"
    params_dir = "./results/optuna/params/"
    
    df_results = pd.read_csv(results_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    
    df_params = df_params[[cls_col, fs_col, key1, key2]]
    df_results = df_results[[key1, key2, metric]]
    
    df_select = df_params.merge(df_results, on=[key1, key2])
    
    
    #select best configs for each model type
    if direction=="max":
        df_best_config = df_select.loc[df_select.groupby(by=[cls_col, fs_col])[metric].idxmax()]
    else:
        df_best_config = df_select.loc[df_select.groupby(by=[cls_col, fs_col])[metric].idxmin()]
    
    
    return df_best_config

def add_value_to_config(config, param_name, value):
    splited = param_name.split(".")
    field  = splited[0]
    if field not in config:
        config[field]=dict()
    if len(splited) == 1:
        config[field] = value
        if isinstance(value, float) and int(value)==value:
            config[field] = int(value)
        return config
    tail_name = ".".join(splited[1:])
    config[field] = add_value_to_config(config[field], tail_name , value)
    return config

def get_best_hyperparameters_from_keys(filename, df_keys):
    key1 = "config_name"
    key2 = "start_time"
    params_dir = "./results/optuna/params/"
    df_params = pd.read_csv(params_dir+filename)
    df_keys = df_keys[[key1, key2]]
    
    rows = df_params.merge(df_keys, on=[key1, key2])
    configs = []
    for i,row in rows.iterrows():
        config = dict()
        for param_name in row.index:
            if row[param_name] is not None and not pd.isna(row[param_name]):
                config = add_value_to_config(config, param_name, row[param_name])
        configs.append(config)
    return configs
    
