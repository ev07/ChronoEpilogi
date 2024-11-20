import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import time

sys.path.append("../")
from final_statistics import *
from data_opener import open_dataset_and_ground_truth
from tuning.first_wave_main import setup_dataset
from tuning.routines import compute_stats_selected

from baselines.feature_selection import GroupLasso, NoSelection
from baselines.estimators import ARDLModel, SVRModel, TFTModel, DeepARModel

def full_results_generator(datasets, rootdir = "../", seed=0):
    
    for dataset in datasets:
        ex_datasetup = setup_dataset(dataset, None, None)["DATASET"]
        data_dir = ex_datasetup["PATH"]
        target_extraction = ex_datasetup["TARGET_CHOICE"]
        maximum_target_extraction = ex_datasetup["MAXIMUM_NUMBER_TARGETS"]
        rng = np.random.default_rng(seed)
        
        filelist = list(os.listdir(rootdir + "data/" + data_dir + "/"))
        
        for i,filename in enumerate(filelist):
            if not os.path.isfile(rootdir + "data/" + data_dir + "/" + filename):
                continue
    
            data, var, cause_dict, _ = open_dataset_and_ground_truth(data_dir, filename, rootdir)
            # make sure to avoid extracting all targets in large datasets
            if target_extraction == "all":
                target_set = var
            elif target_extraction == "sampling":
                target_set = rng.choice(var,size=(maximum_target_extraction,), replace=False, shuffle=False)
            elif target_extraction == "given":
                target_set = var
            
            for target in target_set:
                result_filename = dataset+"_"+os.path.splitext(filename)[0]+"_"+target+".csv"
                if not os.path.isfile("../tuning/results/optuna/test_stats/"+result_filename):  # in case the file wasn't computed
                    continue
                
                data_keys = val_all_methods_best_configuration(result_filename, "R2", direction="max")
                
                causal_list = cause_dict[target] if cause_dict is not None else None
                
                yield data_keys, data, dataset, filename, target, result_filename, causal_list


def number_and_selected_sets_structure(fs_multi_instance):
    """
    Compute the number of possibilities for the selected set.
    Also returns a list of list containing the choices of variables for each component, to be used by get_selected_from_multiset_index.
    """
    list_multi_set = [[fs_multi_instance.instance.selected_features[0]]]
    number = 1
    for element in fs_multi_instance.instance.selected_features[1:]:
        list_multi_set.append([element]+fs_multi_instance.instance.equivalent_variables[element])
        number *= len(list_multi_set[-1])
    return number, list_multi_set

def get_selected_from_multiset_index(index, list_multi_set, total_number):
    """
    From the multiset list and an index, find the multiset associated to this index.
    Needs also the total number of subsets that can be made, previously computed by number_and_selected_sets_structure
    """    
    n = index
    denom = total_number
    selected = []
    for candidates in list_multi_set:
        denom = denom // len(candidates)
        selected.append(candidates[n//denom])
        n = n % denom
    return selected

    
def get_splitted_dataset(data, dataset, test_fraction=0.2):
    """
    Split data into train-test, according to hyperparameter tuning holdout config.
    test_fraction parametrizes the size of the test set compared to the train set.
    """
    ex_datasetup = setup_dataset(dataset, None, None)["DATASET"]
    holdout_ratio = ex_datasetup["HOLDOUT_RATIO"]
    train_size = int(len(data)*(1-holdout_ratio))
    
    # make sure to have at least 50 observations in holdout
    if len(data)-train_size<50:
        train_size = len(data)-50
    
    data_train = data.iloc[:train_size]
    
    # make sure that test fraction isn't longer than dataset
    if train_size+int(train_size*test_fraction)>len(data):
        data_test = data[train_size:]
    else:
        data_test = data[train_size:train_size+int(train_size*test_fraction)]
    return data_train, data_test

def get_fs_stats(total_size, fs_set, gt_cause_list, dataset_name):
    if dataset_name != "VARNoisyCopies":
        return compute_stats_selected(total_size, fs_set, gt_cause_list, None)
    else:
        fs_set_transform = [x.split("_")[0] for x in fs_set]
        gt_cause_str = list(map(str,gt_cause_list))
        return compute_stats_selected(total_size, fs_set_transform, gt_cause_str, None)
   
def check_noisy_copies_dataset_anomalies(list_multi_set):
    #verify that the equivalence class of each variable is respected
    diff=0  # number of elements that should not be in the equivalence of a variable.
    for element in list_multi_set:
        ref = element[0].split("_")[0]
        diff += sum([ref!=x.split("_")[0] for x in element[1:]])
    return diff
def check_noisy_copies_dataset_missing(list_multi_set):
    #verify that the equivalence class of each variable is respected
    diff=0  # number of elements that have been correctly included in the equivalence of a variable.
    miss=0  # number of elements that are missing from the equivalence class.
    for element in list_multi_set:
        ref = element[0].split("_")[0]
        diff += sum([ref==x.split("_")[0] for x in element])
        if ref!=element[0]:  #not the target or an element without equivalences
            miss += 10 - diff
    return miss

    

def check_NoisyVAR_solutions_0(list_multi_set, filename):
    res_dict = dict()
    # first, get the ground truth
    df = pd.read_csv("../data/equivalence_datasets/NoisyVAR_characteristics.csv")
    df = df[df["filename"]==filename]
    th_solution = df["theoretical_solution"]
    th_solution = eval(th_solution.values[0])
    th_solution = [list(map(str, l)) for l in th_solution]
    # find irreplaceable, replaceable variables in both GT and exp
    irreplaceable_set_th = set([x[0] for x in th_solution if len(x)==1])
    irreplaceable_set_exp = set([x[0] for x in list_multi_set if len(x)==1])
    replaceable_set_th = set(sum([l for l in th_solution if len(l)>1],[]))
    replaceable_set_exp = set(sum([ l for l in list_multi_set if len(l)>1],[]))
    # then, compute TP, FP, FN on both type of variables
    res_dict["irreplaceable_TP"] = len(irreplaceable_set_th.intersection(irreplaceable_set_exp))
    res_dict["irreplaceable_FN"] = len(irreplaceable_set_th - irreplaceable_set_exp)
    res_dict["irreplaceable_FP"] = len(irreplaceable_set_exp - irreplaceable_set_th)
    res_dict["irreplaceable_precision"] = np.divide(res_dict["irreplaceable_TP"],(res_dict["irreplaceable_TP"]+res_dict["irreplaceable_FP"]))
    res_dict["irreplaceable_recall"] = np.divide(res_dict["irreplaceable_TP"],(res_dict["irreplaceable_TP"]+res_dict["irreplaceable_FN"]))
    res_dict["irreplaceable_f1score"] = np.divide(2*res_dict["irreplaceable_TP"],(2*res_dict["irreplaceable_TP"]+res_dict["irreplaceable_FP"]+res_dict["irreplaceable_FN"]))
    res_dict["replaceable_TP"] = len(replaceable_set_th.intersection(replaceable_set_exp))
    res_dict["replaceable_FN"] = len(replaceable_set_th - replaceable_set_exp)
    res_dict["replaceable_FP"] = len(replaceable_set_exp - replaceable_set_th)
    res_dict["replaceable_precision"] = np.divide(res_dict["replaceable_TP"],(res_dict["replaceable_TP"]+res_dict["replaceable_FP"]))
    res_dict["replaceable_recall"] = np.divide(res_dict["replaceable_TP"],(res_dict["replaceable_TP"]+res_dict["replaceable_FN"]))
    res_dict["replaceable_f1score"] = np.divide(2*res_dict["replaceable_TP"],(2*res_dict["replaceable_TP"]+res_dict["replaceable_FP"]+res_dict["replaceable_FN"]))
    res_dict["th_solution"] = str(th_solution)
    
    # I would have liked to give the number of correct solution overall
    # but not only is it 0 when the length of the reference set is incorrect,
    # computing it might be very long. It is better to rely on variable characterization. 
    
    # one last thing is to add the characteristics of the dataset to the results
    res_dict["th.lags"] = df["lags"].values[0]
    res_dict["th.total_variables"] = df["total_variables"].values[0]
    res_dict["th.mb_size"] = df["mb_size"].values[0]
    
    return res_dict

def is_markov_boundary(fs_set,th_solution):
    if len(fs_set)!=len(th_solution):
        return False
    indexes = list(range(len(th_solution)))
    for v in fs_set:
        for i in range(len(indexes)):
            if v in th_solution[indexes[i]]:
                indexes.remove(indexes[i])
                break
    if len(indexes)==0:
        return True
    return False
    
def is_markov_blanket(fs_set,th_solution):
    if len(fs_set)<len(th_solution):
        return False
    indexes = list(range(len(th_solution)))
    for v in fs_set:
        for i in range(len(indexes)):
            if v in th_solution[indexes[i]]:
                indexes.remove(indexes[i])
                break
    if len(indexes)==0:
        return True
    return False
      

def check_NoisyVAR_solutions(fs_set, filename):
    res_dict = dict()
    # first, get the ground truth
    df = pd.read_csv("../data/equivalence_datasets/NoisyVAR_characteristics.csv")
    df = df[df["filename"]==filename]
    th_solution = df["theoretical_solution"]
    th_solution = eval(th_solution.values[0])
    th_solution = [list(map(str, l)) for l in th_solution]
    # then, find irreplaceable recall (can be compared)
    irreplaceable_set_th = set([x[0] for x in th_solution if len(x)==1])
    replaceable_set_th = set(sum([l for l in th_solution if len(l)>1],[]))
    fs_set = set(fs_set)
    res_dict["irreplaceable_TP"] = len(irreplaceable_set_th.intersection(fs_set))
    res_dict["irreplaceable_FN"] = len(irreplaceable_set_th - fs_set)
    res_dict["irreplaceable_recall"] = np.divide(res_dict["irreplaceable_TP"],(res_dict["irreplaceable_TP"]+res_dict["irreplaceable_FN"]))
    # then, find number of causal variables precision
    res_dict["causal_TP"] = len(irreplaceable_set_th.intersection(fs_set))
    res_dict["causal_FP"] = len(fs_set - irreplaceable_set_th)
    res_dict["causal_precision"] = np.divide(res_dict["causal_TP"],(res_dict["causal_TP"]+res_dict["causal_FP"]))
    # then, find if the set is a markov boundary
    res_dict["is_markov_boundary"] = is_markov_boundary(fs_set,th_solution)
    # then, find if the set is a markov blanket
    res_dict["is_markov_blanket"] = is_markov_blanket(fs_set,th_solution)
    
    res_dict["th_solution"] = str(th_solution)
    # one last thing is to add the characteristics of the dataset to the results
    res_dict["th.lags"] = df["lags"].values[0]
    res_dict["th.total_variables"] = df["total_variables"].values[0]
    res_dict["th.mb_size"] = df["mb_size"].values[0]
    
    return res_dict

def hyperparam_setting(best_hyperparams, fs_version, cls_version):
    """
    Get parameters corresponding to FS algorithm.
    """
    # get config from the appropriate version
    best_hyperparams_copy = [x for x in best_hyperparams if x['FS']["NAME"]==fs_version]
    if cls_version=="all":
        curr = -np.inf
        best_sol = []
        for x in best_hyperparams_copy:
            if x["R2"]>curr:
                curr=x["R2"]
                best_sol = [x]
        best_hyperparams_copy = best_sol
    else:
        best_hyperparams_copy = [x for x in best_hyperparams_copy if x['CLS']["NAME"]==cls_version]
    best_hyperparams_copy = best_hyperparams_copy[0]

    return best_hyperparams_copy


def evaluate_metrics(fs_set, hyperparams, data_train, data_test, gt_cause_list, dataset, filename, target):
    """
    Evaluate all metrics individual to the fs_set.
    Params:
     - fs_set: list of the names of the selected covariates
     - hyperparams: best hyperparam configuration including the forecaster to use
     - data_train: training data
     - data_test: testing data
     - gt_cause_list: list of the ground truth names if any
     - dataset: name of dataset
     - target: name of target
    """
    fs_stats = get_fs_stats(len(fs_set), fs_set, gt_cause_list, dataset)
    
    if target not in fs_set:
        fs_set.append(target)
    
    data_train_selected = data_train[fs_set]
    data_test_selected = data_test[fs_set]

    forecaster_hp = hyperparams["CLS"]["CONFIG"]
    forecaster_constructor = {"ARDLModel":ARDLModel, "SVRModel":SVRModel, "TFTModel":TFTModel, "DeepARModel":DeepARModel}[hyperparams["CLS"]["NAME"]]
    forecaster = forecaster_constructor(forecaster_hp, target)
    
    t = time.time()
    predictedvalues = forecaster.fit_predict(data_train_selected, data_test_selected)
    endtime = time.time()-t
    
    truevalues = data_test_selected[target]
    res = forecaster.compute_metrics(predictedvalues, truevalues)
    res["target"]=target
    res["dataset"]=dataset
    res["filename"]=filename
    res["selected_set"] = fs_set
    res["CLS_time"] = endtime
    res = {**res, **fs_stats}
    return res


def main_loop(dataset_to_use, test_fraction, fs_version, cls_version, recovery=True):
    """
    cls_version: version of the classifier to use.
    """

    # if recovery is true, then initializes the result lists and filter the filename,target pairs
    filter_done = []
    allres = []
    if recovery:
        fname = "./results/fsres-{}-{}-{}-{}.csv".format(dataset_to_use, test_fraction, fs_version, cls_version)
        if os.path.isfile(fname):
            allres = [pd.read_csv(fname)]
            files_and_targets = allres[0][["filename","target"]].values
            filter_done = [(x,str(y)) for x,y in files_and_targets]
    print(filter_done)

    for best_configs, data, dataset, filename, target, results_filename, gt_cause_list in full_results_generator([dataset_to_use]):
        print(filename, target)
        if (filename,target) in filter_done:
            print("file,target computed already")
            continue
        
        # get train and test splits
        data_train, data_test = get_splitted_dataset(data, dataset, test_fraction)
        
        # get best hyperparams for each algorithm
        best_hyperparams = get_best_hyperparameters_from_keys(results_filename, best_configs)
        
        # get best hyperparams for chronomp in a format suited to multiple subset chronomp version
        best_hyperparams_algo = hyperparam_setting(best_hyperparams, fs_version, cls_version)
        
        # launch the multiset learning with the hyperparameters
        config = best_hyperparams_algo["FS"]["CONFIG"] if "CONFIG" in best_hyperparams_algo["FS"] else dict()
        fs = {"GroupLasso":GroupLasso,"NoSelection":NoSelection}[fs_version](config, target)
        
        # fit
        t = time.time()
        fs.fit(data_train)
        end_time = time.time()-t
        
        results = []
        # get reference selected set
        fs_set = fs.get_selected_features()
        # compute forecasting and set selection stats
        res = evaluate_metrics(fs_set, best_hyperparams_algo, data_train, data_test, gt_cause_list, dataset, filename, target)
        # flag this solution as a reference set
        res["reference_set"]=True
        res["FS_time"] = end_time
        res["CLS_name"] = best_hyperparams_algo["CLS"]["NAME"]
        
        # theoretical stats for NoisyVAR
        if dataset_to_use in ["NoisyVAR_500","NoisyVAR_2500", "NoisyVAR_8000"]:
            equiv_stats = check_NoisyVAR_solutions(fs_set, filename)
            res = {**res, **equiv_stats}
        
        results.append(res)
        results = pd.DataFrame(results)
        
        #break
        allres.append(results)
        
        curr = pd.concat(allres)
        curr.to_csv("./results/fsres-{}-{}-{}-{}.csv".format(dataset, test_fraction, fs_version, cls_version), index=False)
                         
        
    allres = pd.concat(allres)
    
    return allres


if __name__=="__main__":
    _, dataset, test_fraction, fs_version, cls_version = sys.argv
    test_fraction = float(test_fraction)

    fsres = main_loop(dataset, test_fraction, fs_version, cls_version)
    fsres.to_csv("./results/fsres-{}-{}-{}-{}.csv".format(dataset, test_fraction, fs_version, cls_version), index=False)

