import pandas as pd
import sys
sys.path.append("../")
from tuning.first_wave_main import setup_dataset
from tuning.routines import full_experiment
import numpy as np

#####
#
#   Bootstrap specific part
#
#####

def boot_extract_mean_best_configuration(df_select: pd.DataFrame, metric: str, direction="max"):
    """
    For a given pair (classifier, fs), gives back the best in score and corresponding out_score
    direction states whether the metric should be maximized or minimized
    
    params:
        df_select: dataframe, contains the rows associated to the chosen cls, fs algorithms.
            It has to contain at least three columns: 'fold', 'in_'+metric, 'out_'+metric
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    folds = "fold"
    
    if direction=="max":
        df_best_rows = df_select.loc[df_select.groupby(by=[folds])["in_"+metric].idxmax()]
    else:
        df_best_rows = df_select.loc[df_select.groupby(by=[folds])["in_"+metric].idxmin()]
        
    final_perf = df_best_rows["out_"+metric].mean()
    
    return final_perf 

def boot_extract_mean_best_configuration_from_file(cls_name: str, fs_name:str , filename: str, metric: str, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    extract rows corresponding to a given CLS and FS algorithm, 
    and compute the out-of-sample metric of the best configuration for each fold, to finally return the mean value.
    
    params:
        cls_name: name of the classifier whose hyperparameters are optimized
        fs_name: name of the feature selection algorithm whose hyperparameters are optimized
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    folds = "fold"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    df_params = df_params[df_params[cls_col]==cls_name & df_params[fs_col]==fs_name][[key1,key2]]
    df_bootstrap = df_bootstrap[[key1, key2, "in_"+metric, "out_"+metric, folds]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])

    final_perf = extract_best_configuration(df_select, metric, direction=direction)
    
    return final_perf
    
    
    
def boot_all_methods_mean_best_configuration(filename, metric, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    compute the out-of-sample metric of the best configuration for each fold, to finally return the mean value.
    
    params:
        cls_name: name of the classifier whose hyperparameters are optimized
        fs_name: name of the feature selection algorithm whose hyperparameters are optimized
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    folds = "fold"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    
    df_params = df_params[[cls_col, fs_col, key1, key2]]
    df_bootstrap = df_bootstrap[[key1, key2, "in_"+metric, "out_"+metric, folds]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])
    
    #for each fold in each model type, select the best configuration score
    if direction=="max":
        df_best_rows = df_select.loc[df_select.groupby(by=[folds, cls_col, fs_col])["in_"+metric].idxmax()]
    else:
        df_best_rows = df_select.loc[df_select.groupby(by=[folds, cls_col, fs_col])["in_"+metric].idxmin()]
    
    print(df_best_rows)
    #average score across each model type
    df_metric = df_best_rows.groupby(by=[cls_col, fs_col])["out_"+metric].mean().reset_index()
    
    return df_metric
    


    
    
    
def boot_all_methods_best_mean_configuration(filename, metric, direction="max"):
    """
    Given a target file (in the form <dataset_name>_<target_name>.csv),
    compute the out-of-sample metric of the best configuration for each fold, to finally return the mean value.
    
    params:
        filename: name of the file, usually <dataset_short_name>_<target_name>.csv
        metric: name of the evaluated metric, for instance, 'sse'.
        direction: whether to minimize ('min') or maximize ('max') the metric. For instance, for sse, set to "min".
    returns:
        final_perf: mean out-of-sample performance of the chosen CLS and FS algorithms
    """
    
    key1 = "config_name"
    key2 = "start_time"
    cls_col = "CLS.NAME"
    fs_col = "FS.NAME"
    folds = "fold"
    
    bootstrap_dir = "./results/optuna_bootstrap/bootstrap_stats/"
    params_dir = "./results/optuna_bootstrap/params/"
    
    df_bootstrap = pd.read_csv(bootstrap_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    
    df_params = df_params[[cls_col, fs_col, key1, key2]]
    df_bootstrap = df_bootstrap[[key1, key2, "in_"+metric, "out_"+metric, folds]]
    
    df_select = df_params.merge(df_bootstrap, on=[key1, key2])
    
    #mean of each config across all folds
    df_mean_rows = df_select.groupby(by=[key1, key2, cls_col, fs_col])["in_"+metric].mean().reset_index()
    
    #select best configs for each model type
    if direction=="max":
        df_best_config = df_mean_rows.loc[df_mean_rows.groupby(by=[cls_col, fs_col])["in_"+metric].idxmax()]
    else:
        df_best_config = df_mean_rows.loc[df_mean_rows.groupby(by=[cls_col, fs_col])["in_"+metric].idxmin()]
    
    #merge to select only best configs
    df_all_folds = df_best_config[[cls_col, fs_col, key1, key2]].merge(df_select, on=[key1, key2, cls_col, fs_col])
    
    #compute mean across folds of best configs.
    df_metric = df_all_folds.groupby(by=[cls_col, fs_col, key1, key2])["out_"+metric].mean().reset_index()
    
    return df_metric, df_all_folds[[key1, "in_"+metric,"out_"+metric, folds]]
    
    
    
    
#####
#
# Holdout specific part
#
#####    
    
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
    
    results_dir = "../tuning/results/optuna/test_stats/"
    params_dir = "../tuning/results/optuna/params/"
    
    df_results = pd.read_csv(results_dir+filename)
    df_params = pd.read_csv(params_dir+filename)
    
    
    df_params = df_params[[cls_col, fs_col, key1, key2]]
    df_results = df_results[[key1, key2, metric, "FS_time"]]
    
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
    params_dir = "../tuning/results/optuna/params/"
    df_params = pd.read_csv(params_dir+filename)
    df_keys = df_keys[[key1, key2, "R2", "FS_time"]]
    
    rows = df_params.merge(df_keys, on=[key1, key2])
    configs = []
    for i,row in rows.iterrows():
        config = dict()
        for param_name in row.index:
            if row[param_name] is not None and not pd.isna(row[param_name]):
                config = add_value_to_config(config, param_name, row[param_name])
        configs.append(config)
    return configs
    
def run_best_configs_test_set(dataset, filename, target, configs):
    data_config = setup_dataset(dataset, filename, target)
    data_config["DATASET"]["HOLDOUT"]=False
    
    folds_config = {"FOLDS":{"NUMBER_FOLDS": 0,
              "WINDOW_SIZE": 0.9,
              "STRATEGY": "fixed_start"}}
    
    results = []
    params = []
    for config in configs:
        config = {**config, **data_config, **folds_config}
        name = config["config_name"]+str(config["start_time"])
        df_results, df_params = full_experiment(config, name, compute_selected_stats=True, return_selected=True)
        results.append(df_results)
        params.append(df_params)
    
    return pd.concat(results), pd.concat(params)
    
    
if __name__=="__main__":
    #print(boot_all_methods_mean_best_configuration("pairwiseLinear_data_0_X666.csv", "rmse", direction="min"))
    #print(boot_all_methods_best_mean_configuration("pairwiseLinear_data_0_X666.csv", "rmse", direction="min"))
    filename = "7ts2h_data_0_D.csv"
    dataset = "7ts2h"
    fname = "data_0.csv"
    target = "D"
    data_keys = val_all_methods_best_configuration(filename, "R2", direction="max")
    print(data_keys.values)
    best_hyperparams = get_best_hyperparameters_from_keys(filename, data_keys)
    print(best_hyperparams)
    holdout_results, holdout_params = run_best_configs_test_set(dataset, fname, target, best_hyperparams)
    print(holdout_results.values)
    print(holdout_results.columns)
