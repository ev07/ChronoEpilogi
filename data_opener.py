import pandas as pd


def standardize_df(df):
    return (df - df.mean(axis=0)) / df.std(axis=0)
       



def open_dataset_and_ground_truth(dataset_name: str,
                                  filename: str,
                                  rootdir=".",
                                  **kwargs):
    """
    Open a file in a dataset family, where the ground truth is known:
    Params:
     - dataset_name: name of the dataset family
     - filename: name of the file containing the MTS instance
     (note: /data/<dataset_name>/<filename> is the complete path, with filename including the extension)
     - rootdir (optional): string indicating the root repository
    Returns:
     - df: the dataframe containing the MTS
     - possible_targets: the list of attribute names that can be forecasting targets
     - None (retrocompatibility with earlier code)
     - None (retrocompatibility with earlier code)
    """

    if dataset_name=="monash/electricity" or\
         dataset_name=="monash/traffic" or\
         dataset_name=="monash/solar" or\
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


    if dataset_name in ["monash/solar", "monash/electricity", "monash/traffic"]:
        return df, var_names, None, None
    elif dataset_name in ["GNN_benchmarks/PEMS-BAY", "GNN_benchmarks/METR-LA"]:
        return df, var_names, None, None
    
    
    # equivalence datasets (no GT due to nonunique MB and no representation for now)
    elif dataset_name=="equivalence_datasets/NoisyVAR":
        var_names = ["0"]
        return df, var_names, None, None
    else:
        raise Exception("Dataset specified in argument is not implemented")
    




