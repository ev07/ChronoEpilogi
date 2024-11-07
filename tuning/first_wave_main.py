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

import baselines.estimators
import baselines.feature_selection
from data_opener import open_dataset_and_ground_truth
from models import NotEnoughDataError
from baselines.feature_selection import MaximalSelectedError

optuna.logging.set_verbosity(optuna.logging.WARNING)


def fs_cls_pair_already_optimized(dataset, fs_name, cls_name, filename, target, experiment_identifier):
    params_dir = "./results/optuna/params/"
    params_filename = dataset+"_"+os.path.splitext(filename)[0]+"_"+target+".csv"
    if not os.path.isfile(params_dir+params_filename):
        return False
    df_params = pd.read_csv(params_dir+params_filename)
    df_params = df_params[df_params["FS.NAME"]==fs_name]
    if len(df_params)==0:
        return False
    df_params = df_params[df_params["CLS.NAME"]==cls_name]
    if len(df_params)==0:
        return False
    # add an experiment identifier. This allows for different optimisations of same targets, with different HP grid.
    if "experiment_identifier" not in df_params.columns:
        return False
    df_params = df_params[df_params["experiment_identifier"]==experiment_identifier]
    if len(df_params)==0:
        return False
    return True

def save_append(df, path):
    df_0 = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
    df = pd.concat([df_0, df], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


def setup_dataset(dataset_name, filename, target):
    config = dict()
    config["DATASET"] = dict()
    config["DATASET"]["NAME"] = dataset_name
    config["DATASET"]["FILENAME"] = filename
    config["DATASET"]["TARGET"] = target
    config["DATASET"]["HOLDOUT"] = True
    config["DATASET"]["HOLDOUT_RATIO"] = 0.1
    if dataset_name == "VAR10":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "VARSmall/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "all",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "VAR10000":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "VARLarge/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "VAR10000_redundant":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "VARLarge/redundant/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "VARNoisyCopies":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "VARNoisyCopies/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "7ts2h":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "SynthNonlin/7ts2h",
            "CAUSES": "parents",
            "TARGET_CHOICE": "all",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "fMRI":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "fMRI_processed_by_Nauta/returns/our_selection",
            "CAUSES": "parents",
            "TARGET_CHOICE": "all",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "CLIM":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "TestCLIM_N-5_T-250/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "all",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "piecewiseLinear":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "dgp/piecewise_linear/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "monotonic":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "dgp/monotonic/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "trigonometric":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "dgp/trigonometric/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
    elif dataset_name == "wikipedia":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "wikipediaMathEssencials/returns",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 100}
            }
    elif dataset_name == "Appliances":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "Appliances",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "AusMacro":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "AusMacro",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "AusMeteo":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "AusMeteo",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 100}
            }
    elif dataset_name == "weather":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "monash/weather",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.9
    elif dataset_name == "temperature_rain":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "monash/temperature_rain",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.9
    elif dataset_name in ["electricity", "electricity_long"]:
        config = {"DATASET":{**config["DATASET"],
            "PATH": "monash/electricity",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.9
        if dataset_name == "electricity_long":
            config["DATASET"]["HOLDOUT_RATIO"] = 0.3
    elif dataset_name in ["solar", "solar_long"]:
        config = {"DATASET":{**config["DATASET"],
            "PATH": "monash/solar",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.9
        if dataset_name == "solar_long":
            config["DATASET"]["HOLDOUT_RATIO"] = 0.3
    elif dataset_name in ["traffic","traffic_long"]:
        config = {"DATASET":{**config["DATASET"],
            "PATH": "monash/traffic",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.9
        if dataset_name == "traffic_long":
            config["DATASET"]["HOLDOUT_RATIO"] = 0.3
        
    elif dataset_name == "pems-bay":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "GNN_benchmarks/PEMS-BAY",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.8 # like original
    elif dataset_name == "metr-la":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "GNN_benchmarks/METR-LA",
            "CAUSES": "parents",
            "TARGET_CHOICE": "sampling",
            "MAXIMUM_NUMBER_TARGETS": 10}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.8 # like original
    
    elif dataset_name == "size_4_gaussian":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "equivalence_datasets/size_4_gaussian",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": None}
            }
    elif dataset_name == "NoisyVAR_500":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "equivalence_datasets/NoisyVAR",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": 1}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.944444
        
    elif dataset_name == "NoisyVAR_2500":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "equivalence_datasets/NoisyVAR",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": 1}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.7222222
        
    elif dataset_name == "NoisyVAR_8000":
        config = {"DATASET":{**config["DATASET"],
            "PATH": "equivalence_datasets/NoisyVAR",
            "CAUSES": "parents",
            "TARGET_CHOICE": "given",
            "MAXIMUM_NUMBER_TARGETS": 1}
            }
        config["DATASET"]["HOLDOUT_RATIO"] = 0.1111111
    return config


def setup_config(trial, fs_name, cls_name):
    cls_conf = baselines.estimators.generate_optuna_parameters(cls_name, trial)
    cls_conf = baselines.estimators.complete_config_from_parameters(cls_name, cls_conf)
    fs_conf = baselines.feature_selection.generate_optuna_parameters(fs_name, trial)
    fs_conf = baselines.feature_selection.complete_config_from_parameters(fs_name, fs_conf)
    

    config = {"CLS":{"NAME":cls_name,
           "CONFIG":cls_conf},
     "FS":{"NAME":fs_name,
           "CONFIG":fs_conf},
     "FOLDS":{"NUMBER_FOLDS": 5,
              "WINDOW_SIZE": 0.4,
              "STRATEGY": "fixed_start"}
    }
    return config




def generate_optuna_objective_function(fs_name, cls_name, dataset_setup, objective="R2"):
    memorize = {"params":[], "results":[]}
    def optuna_objective_function(trial):
        if trial.number%1 == 0:
            print("\t\t\tTrial number", trial.number)
        config_file = setup_config(trial, fs_name, cls_name)
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

def full_experiment(dataset, fs_name, cls_name, experiment_identifier, seed=0):
    ex_datasetup = setup_dataset(dataset, None, None)["DATASET"]
    data_dir = ex_datasetup["PATH"]
    target_extraction = ex_datasetup["TARGET_CHOICE"]
    maximum_target_extraction = ex_datasetup["MAXIMUM_NUMBER_TARGETS"]
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    filelist = sorted(list(os.listdir(rootdir + "data/" + data_dir + "/")))
    
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
            fs_space = baselines.feature_selection.generate_optuna_search_space(fs_name)
            cls_space = baselines.estimators.generate_optuna_search_space(cls_name)
            space = {**fs_space, **cls_space}
        
            # GridSampler launch
            studylength = np.prod([len(x) for _,x in space.items()])
            if first_evaluation_flag:
                print("\t\tNumber of configurations to be evaluated:",studylength)
            objective, results = generate_optuna_objective_function(fs_name, cls_name, dataset_setup)
            study = optuna.create_study(sampler=GridSampler(space))
            study.optimize(objective, n_trials=studylength)
            
            #results contains all the training information from the trials
            filename_xt = os.path.splitext(filename)[0]
            #print(results)
            params_df = pd.concat(results["params"])
            results_df = pd.concat(results["results"])
            #bootstrap_df = pd.concat(results["bootstrap"])
            
            # the following line identifies each experiment loop, so that the resume operation can occur
            params_df["experiment_identifier"] = experiment_identifier
            
            save_append(params_df,"./results/optuna/params/"+dataset+"_"+filename_xt+"_"+target+".csv")
            save_append(results_df,"./results/optuna/test_stats/"+dataset+"_"+filename_xt+"_"+target+".csv")
            #save_append(bootstrap_df,"./results/optuna_bootstrap/bootstrap_stats/"+dataset+"_"+filename_xt+"_"+target+".csv")
            
            if first_evaluation_flag:  # give estimation of the procedure length in seconds
                first_evaluation_flag=False
                t = time.time()-time_begin
                print("\t\tOne variable took",t ,"seconds.")
                print("\t\tEstimated time for whole pipeline:",len(target_set)*len(filelist)*t, "seconds")


if __name__=="__main__":
    _, data, fs, model, experiment_identifier = sys.argv
    print(data)
    full_experiment(data, fs, model, experiment_identifier)



    
