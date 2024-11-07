import sys
import os

#os.environ['OPENBLAS_NUM_THREAD']='10'
#os.environ['MKL_NUM_THREADS'] = '10'

import time
import pandas as pd
import numpy as np

import optuna
from optuna.samplers import RandomSampler, GridSampler

rootdir = '../'
sys.path.append(rootdir)

from tuning.routines import full_experiment as launch_experiment
from tuning.routines import TargetNotSelectedError

from tuning.first_wave_main import fs_cls_pair_already_optimized, save_append
from tuning.first_wave_main import setup_dataset

from tuning.final_statistics import val_all_methods_best_configuration, get_best_hyperparameters_from_keys

import baselines.estimators
import baselines.feature_selection
from data_opener import open_dataset_and_ground_truth
from models import NotEnoughDataError
from baselines.feature_selection import MaximalSelectedError

optuna.logging.set_verbosity(optuna.logging.WARNING)



def get_best_fs_params(fs_name, proxy_cls_name, best_hyperparams):
    best_hyperparams_copy = [x for x in best_hyperparams if x['FS']["NAME"]==fs_name]
    best_hyperparams_copy = [x for x in best_hyperparams_copy if x['CLS']["NAME"]==proxy_cls_name]
    best_hyperparams_copy = best_hyperparams_copy[0]
    return best_hyperparams_copy["FS"]


def setup_config(trial, res_filename, fs_name, cls_name, proxy_cls_name, metric):
    cls_conf = baselines.estimators.generate_optuna_parameters(cls_name, trial)
    cls_conf = baselines.estimators.complete_config_from_parameters(cls_name, cls_conf)
    
    # find the configuration of the fs_algo from the proxy cls optimization
    all_best_configs = val_all_methods_best_configuration(res_filename, metric, direction="max")
    all_best_hyperparams = get_best_hyperparameters_from_keys(res_filename, all_best_configs)
    
    fs_conf = get_best_fs_params(fs_name, proxy_cls_name, all_best_hyperparams)

    config = {"CLS":{"NAME":cls_name,
           "CONFIG":cls_conf},
     "FS":fs_conf,
     "FOLDS":{"NUMBER_FOLDS": 5,
              "WINDOW_SIZE": 0.4,
              "STRATEGY": "fixed_start"}
    }
    return config



def generate_optuna_objective_function(fs_name, cls_name, proxy_cls_name, dataset_setup, objective="R2"):
    memorize = {"params":[], "results":[]}
    def optuna_objective_function(trial):
        if trial.number%1 == 0:
            print("\t\t\tTrial number", trial.number)
        res_filename = dataset_setup["DATASET"]["NAME"]+"_" +os.path.splitext(dataset_setup["DATASET"]["FILENAME"])[0]+"_"+dataset_setup["DATASET"]["TARGET"]+".csv"
        config_file = setup_config(trial, res_filename, fs_name, cls_name, proxy_cls_name, metric=objective)
        config_file = {**config_file, **dataset_setup}
        
        config_name = fs_name + "_" + cls_name + "_" + str(trial.number)
        
        # sometimes, the configuration is not adapted to the model, and will return an error.
        # rather than check the validity of every configuration, it is simple to catch the exceptions
        # coming from such configurations.
        try:
            df_results, df_params = launch_experiment(config_file, config_name, run_bootstrap=False, compute_selected_stats=True)
        except NotEnoughDataError as e:  # ARDL model was not given enough data
            print("Configuration",trial.number,"failed with error:\n",str(e))
            #raise e
            return np.inf
        except MaximalSelectedError as e:  # SelectFromModel was given too many feature compared to available
            return np.inf
        except TargetNotSelectedError as e:  # no feature selected, cannot compute performance of the forecaster
            print("Configuration",trial.number,"selected no features.")
            return np.inf
        
        # save resulting data
        memorize["params"].append(df_params)
        memorize["results"].append(df_results)
        
        return df_results[objective].mean()
        
    return optuna_objective_function, memorize





def full_experiment(dataset, fs_name, cls_name, proxy_cls_name, experiment_identifier, seed=0):
    ex_datasetup = setup_dataset(dataset, None, None)["DATASET"]
    data_dir = ex_datasetup["PATH"]
    target_extraction = ex_datasetup["TARGET_CHOICE"]
    maximum_target_extraction = ex_datasetup["MAXIMUM_NUMBER_TARGETS"]
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    filelist = list(os.listdir(rootdir + "data/" + data_dir + "/"))
    
    first_evaluation_flag = True  # flag for the first call to objective, to get time estimation.
    count = 0
    
    for i,filename in enumerate(filelist):
        if not os.path.isfile(rootdir + "data/" + data_dir + "/" + filename):
            continue
        print("New file",filename ,"time since begining is", time.time()-start_time, "(", i, "/", len(filelist), ")")
        
        _, var, _, _ = open_dataset_and_ground_truth(data_dir, filename, "parents", rootdir, skip_causal_step=True)
        # make sure to avoid extracting all targets in large datasets
        if target_extraction == "all":
            target_set = var
        elif target_extraction == "sampling":
            target_set = rng.choice(var,size=(maximum_target_extraction,), replace=False, shuffle=False)
        elif target_extraction == "given":
            target_set = var
        
        
        
        for target in target_set:
            print("\tTarget", target, "beggining")
            count+=1
            
            # check if results for the given target of the given file have already been computed
            if fs_cls_pair_already_optimized(dataset, fs_name, cls_name, filename, target, experiment_identifier):
                continue
            
            if first_evaluation_flag: # record time on first iteration
                time_begin = time.time()
            
            dataset_setup = setup_dataset(dataset, filename, target)
        
            # if we are using the GridSampler, we need to specify a search space
            cls_space = baselines.estimators.generate_optuna_search_space(cls_name)
        
            # GridSampler launch
            studylength = np.prod([len(x) for _,x in cls_space.items()])
            if first_evaluation_flag:
                print("\t\tNumber of configurations to be evaluated:",studylength)
            objective, results = generate_optuna_objective_function(fs_name, cls_name, proxy_cls_name, dataset_setup)
            study = optuna.create_study(sampler=GridSampler(cls_space))
            study.optimize(objective, n_trials=studylength)
            
            #results contains all the training information from the trials
            filename_xt = os.path.splitext(filename)[0]
            #print(results)
            params_df = pd.concat(results["params"])
            results_df = pd.concat(results["results"])
            #bootstrap_df = pd.concat(results["bootstrap"])
            
            # the following line identifies each experiment loop, so that the resume operation can occur
            params_df["experiment_identifier"] = experiment_identifier
            params_df["proxy_cls_name"] = proxy_cls_name
            
            save_append(params_df,"./results/optuna/params/"+dataset+"_"+filename_xt+"_"+target+".csv")
            save_append(results_df,"./results/optuna/test_stats/"+dataset+"_"+filename_xt+"_"+target+".csv")
            #save_append(bootstrap_df,"./results/optuna_bootstrap/bootstrap_stats/"+dataset+"_"+filename_xt+"_"+target+".csv")
            
            if first_evaluation_flag:  # give estimation of the procedure length in seconds
                first_evaluation_flag=False
                t = time.time()-time_begin
                print("\t\tOne variable took",t ,"seconds.")
                print("\t\tEstimated time for whole pipeline:",len(target_set)*len(filelist)*t, "seconds")
            if count>20:
                return


if __name__=="__main__":
    _, data, fs, model, proxy_model, experiment_identifier = sys.argv
    print(data)
    full_experiment(data, fs, model,proxy_model, experiment_identifier)
