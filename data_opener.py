import pandas as pd
from collections import defaultdict
import networkx as nx


#! TODO: add support for lagged causes (ancestor, connected) computation with declared max lag.



# extractor functions
def node_to_tuple(node):
    lag, variable = node[1:].split(".")
    return (variable, lag)

def get_all_parents(graph, target):
    if "L0." + target in graph.nodes:
        l = list(graph.predecessors("L0." + target))
        return list(map(node_to_tuple, l))
    else:
        return list(graph.predecessors(target))

def get_all_ancestors(graph, target):
    if "L0." + target in graph.nodes:
        l = list(nx.ancestors(graph, "L0." + target))
        return list(map(node_to_tuple, l))
    else:
        return list(nx.ancestors(graph, target))

def get_all_connected(graph, target):
    if "L0." + target in graph.nodes:
        l = list(nx.node_connected_component(graph.to_undirected(), "L0." + target))
        return list(map(node_to_tuple, l))
    else:
        return list(nx.node_connected_component(graph.to_undirected(), target))

def standardize_df(df):
    return (df - df.mean(axis=0)) / df.std(axis=0)
       



def open_dataset_and_ground_truth(dataset_name: str,
                                  filename: str,
                                  cause_extraction="parents",
                                  rootdir=".",
                                  compute_window_causal_graph=False,
                                  window_size="max_direct",
                                  skip_causal_step=False):
    """
    Open a file in a dataset family, where the ground truth is known:
    Params:
     - dataset_name: name of the dataset family
     - filename: name of the file containing the MTS instance
     (note: /data/<dataset_name>/<filename> is the complete path, with filename including the extension)
     - cause_extraction (optional): the method to compute the relevant variables
     - rootdir (optional): string indicating the root repository
     - computed_window_causal_graph (optional): bool set to True to compute window causal graph
     - window_size (optional): lag selection strategy for the window causal graph. Default is "max_direct" which takes the maximal lag of a cause.
     - skip_causal_step (optional): avoid computing causes_attributes_dict and lagged_causes_attributes_dict, return None, None instead.
    Returns:
     - df: the dataframe containing the MTS
     - possible_targets: the list of attribute names that can be forecasting targets
     - causes_attributes_dict: dictionary associating each attribute to the list of its relevant predictors.
     - lagged_attributes_dict: dictionary associating each attribute to the list of its relevant predictors, containing lag information.
     - (optional) if compute_window_causal_graph is True, return the window causal graph as a networkx Digraph object
    """

    if dataset_name[:11] == "SynthNonlin":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df = df[df.columns[1:]]

    elif dataset_name[:4] == "fMRI":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]

    elif dataset_name[:10] == "FinanceCPT":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, header=None)
        df.columns = [str(i) for i in df.columns]

    elif dataset_name[:18] == "TestCLIM_N-5_T-250":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, header=None, sep=" ")
        df.columns = [str(i) for i in df.columns]
        
    elif dataset_name=="VARProcess/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARProcessNoCorr/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARLarge/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARLarge/redundant/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARNoisyCopies/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARSmall/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="VARVaried/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="dgp/piecewise_linear/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="dgp/monotonic/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="dgp/trigonometric/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="wikipediaMathEssencials/returns":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="Appliances":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="AusMacro":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df = df[df.columns[1:]]
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="AusMeteo":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="monash/electricity" or\
         dataset_name=="monash/traffic" or\
         dataset_name=="monash/weather" or \
         dataset_name=="monash/solar" or\
         dataset_name=="monash/temperature_rain":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="equivalence_datasets/size_4_gaussian":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename)
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="equivalence_datasets/NoisyVAR":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, compression="gzip")
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="GNN_benchmarks/PEMS-BAY":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, compression="gzip")
        df = df[df.columns[2:]]  # exclude time 
        df.columns = [str(i) for i in df.columns]
    elif dataset_name=="GNN_benchmarks/METR-LA":
        df = pd.read_csv(rootdir + "/data/" + dataset_name + "/" + filename, compression="gzip")
        df = df[df.columns[2:]]  # exclude time
        df.columns = [str(i) for i in df.columns]
        
    else:
        raise Exception("Dataset specified in config file is not implemented")
        

    var_names = list(df.columns)
    
    df = standardize_df(df)

    if dataset_name[:11] == "SynthNonlin":
        if dataset_name == "SynthNonlin/7ts2h":
            ground_truth_parents = defaultdict(list)
            ground_truth_lags = 10  # could be anything since we don't care about lags in this project
            ground_truth_parents["A"] = [("D", 1), ("A", 1)] #+ [("B", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["D"] = [("H", 1), ("D", 1)] #+ [("E", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["H"] = [("C", 1), ("H", 1)]
            ground_truth_parents["C"] = [("C", 1)]
            ground_truth_parents["F"] = [("C", 1), ("F", 1)]
            ground_truth_parents["B"] = [("F", 1), ("B", 1)] #+ [("A", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["E"] = [("B", 1), ("E", 1)] #+ [("D", i) for i in range(1, ground_truth_lags + 1)]
        else:
            raise Exception("Dataset specified in argument is not implemented")

    elif dataset_name[:4] == "fMRI":
        index = filename[10:]
        index = index[:-4]
        g_truth_name = "fMRI_processed_by_Nauta/ground_truth/sim" + index + "_gt_processed.csv"

        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 0
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            ground_truth_lags = max(ground_truth_lags, lag)

    elif dataset_name[:10] == "FinanceCPT":

        g_truth_name = "FinanceCPT/ground_truth/" + filename[:filename.find("_returns")] + ".csv"
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 0
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            ground_truth_lags = max(ground_truth_lags, lag)

    elif dataset_name[:18] == "TestCLIM_N-5_T-250":
        g_truth_name = "TestCLIM_N-5_T-250/estimated_ground_truth/" + filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 2
        for cause, effect in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), 1))
            ground_truth_parents[str(effect)].append((str(cause), 2))
    
    elif dataset_name=="VARProcess/returns":
        g_truth_name = "VARProcess/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARProcessNoCorr/returns":
        g_truth_name = "VARProcessNoCorr/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARLarge/returns":
        g_truth_name = "VARLarge/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARLarge/redundant/returns":
        g_truth_name = "VARLarge/redundant/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
    
    elif dataset_name=="VARNoisyCopies/returns":
        g_truth_name = "VARNoisyCopies/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARSmall/returns":
        g_truth_name = "VARSmall/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="VARVaried/returns":
        g_truth_name = "VARVaried/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 5
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="dgp/piecewise_linear/returns":
        g_truth_name = "dgp/piecewise_linear/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 10
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
                        
    elif dataset_name=="dgp/monotonic/returns":
        g_truth_name = "dgp/monotonic/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 10
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
                        
    elif dataset_name=="dgp/trigonometric/returns":
        g_truth_name = "dgp/trigonometric/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 10
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            
    elif dataset_name=="wikipediaMathEssencials/returns":
        g_truth_name = "wikipediaMathEssencials/ground_truths/"+filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        #there is no reason to assume particular lag information from this dataset.
        #therefore lag information in ground truth should not be taken into account.
        #a default value is put here
        ground_truth_lags = 10
        for cause, effect in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), 1))
    
    # datasets without ground truth
    elif dataset_name=="Appliances":
        var_names = ["Appliances"]
        return df, var_names, None, None

    elif dataset_name=="AusMacro":
        if filename == "data_0_original.csv":
            var_names = ["RGDP","CPI-ALL"]
        elif filename == "data_1_added_IBR.csv":
            var_names = ["IBR"]
        return df, var_names, None, None
    elif dataset_name=="AusMeteo":
        return df, var_names, None, None
    elif dataset_name in ["monash/solar", "monash/electricity", "monash/traffic", "monash/weather", "monash/temperature_rain"]:
        return df, var_names, None, None
    elif dataset_name in ["GNN_benchmarks/PEMS-BAY", "GNN_benchmarks/METR-LA"]:
        return df, var_names, None, None
    
    
    # equivalence datasets (no GT due to nonunique MB and no representation for now)
    elif dataset_name=="equivalence_datasets/size_4_gaussian":
        var_names = ["T"]
        return df, var_names, None, None
    elif dataset_name=="equivalence_datasets/NoisyVAR":
        var_names = ["0"]
        return df, var_names, None, None
    else:
        raise Exception("Dataset specified in argument is not implemented")
    
    # define targets when restricted
    possible_targets = var_names
    if dataset_name=="VARLarge/redundant/returns":
        possible_targets = possible_targets[:100]
    elif dataset_name=="VARNoisyCopies/returns":
        possible_targets = possible_targets[:10]
    
    ################
    #
    #   Creating the causal graphs
    #
    ################
    
    if skip_causal_step:
        return df, var_names, None, None

    node_names = dict()
    for var in var_names:
        for lag in range(ground_truth_lags + 1):
            node_names[(var, lag)] = "L" + str(lag) + "." + str(var)

    # graph with only parents
    ground_truth_graph = nx.DiGraph()
    for key in node_names:
        ground_truth_graph.add_node(node_names[key])
    for key in ground_truth_parents:
        child_name = "L0." + str(key)
        for parent in ground_truth_parents[key]:
            parent_name = "L" + str(parent[1]) + "." + str(parent[0])
            ground_truth_graph.add_edge(parent_name, child_name)

    # summary graph (no lags)
    summary_graph = nx.DiGraph()
    summary_graph.add_nodes_from(var_names)
    maxlag = 0
    for cause, effect in ground_truth_graph.edges:
        lag = cause[1:cause.find(".")]
        maxlag = max([maxlag, int(lag)])
        cause = cause[cause.find(".") + 1:]
        effect = effect[effect.find(".") + 1:]
        if not summary_graph.has_edge(cause, effect):
            summary_graph.add_edge(cause, effect, lags=[lag])
        else:
            summary_graph[cause][effect]["lags"].append(lag)

    # window causal graph
    if compute_window_causal_graph:
        if window_size=="max_direct":
            nlags = maxlag
        window_graph = nx.DiGraph()
        node_names = []
        for var in df.columns:
            for lag in range(0, nlags+1):
                node_names.append("L" + str(lag) + "." + str(var))
        window_graph.add_nodes_from(node_names)
        for cause, effect in ground_truth_graph.edges:
            lag = int(cause[1:cause.find(".")])
            cause = cause[cause.find(".") + 1:]
            effect = effect[effect.find(".") + 1:]
            for L in range(lag, nlags+1):
                window_graph.add_edge("L" + str(L) + "." + cause, "L" + str(L - lag) + "." + effect)
        #window_graph_pos = dict([(node,
        #                          (1 - int(node[1:node.find(".")]) / data_config['pastPointsToForecast'],
        #                           1 - VAR_NAMES.index(node[node.find(".") + 1:]) / len(VAR_NAMES)))
        #                         for node in window_graph.nodes()])
    
    
    # get causes variables    
    
    causes_attributes_dict = defaultdict(list)

    for target_name in var_names:
        if cause_extraction == "parents":
            causes_attributes = get_all_parents(summary_graph, target_name)
        elif cause_extraction == "ancestors":
            causes_attributes = get_all_ancestors(summary_graph, target_name)
        elif cause_extraction == "connected":
            causes_attributes = get_all_connected(summary_graph, target_name)
        else:
            raise Exception("causeExtraction method specified in config file is not implemented")

        causes_attributes_dict[target_name] = list(map(str,causes_attributes))
        
    # get lagged causes (variables, lag)
    # due to the difficulty of specifying a maximum lag, only parent set is returned
    
    lagged_causes_attributes_dict = defaultdict(list)
    for target_name in var_names:
        lagged_causes_attributes = get_all_parents(ground_truth_graph, target_name)

        lagged_causes_attributes_dict[target_name] = lagged_causes_attributes
    
    if compute_window_causal_graph:
        return df, possible_targets, causes_attributes_dict, lagged_causes_attributes_dict, window_graph
        
    return df, possible_targets, causes_attributes_dict, lagged_causes_attributes_dict



