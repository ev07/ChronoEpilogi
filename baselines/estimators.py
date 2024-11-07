import sys
import os

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoLars
from xgboost import XGBRegressor

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, RMSE
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

rootdir = '../'
sys.path.append(rootdir)

import data_opener
import models



class Estimator():
    # whether the model supports begin given a list (variable, lag)
    # to filter the MTS window with.
    support_feature_filtering = None
    #define if the model needs the target column to be in the input
    is_autoregressive = False
    
    def __init__(self, config, target):
        self.config = config
        self.target = target
    
    def fit_predict(self, data_train, data_test):
        # to be implemented within classes  
        return None
    
    
    def compute_metrics(self, y_pred, y_true, align=True):
        if align:
            y_true = y_true.loc[y_pred.index]  # align time series
        statistics = dict()
        statistics["sse"] = np.sum((y_true - y_pred)**2)
        statistics["R2"] = 1-statistics["sse"]/np.std(y_true)**2/len(y_true)
        statistics["rmse"] = np.sqrt(statistics["sse"]/len(y_true))
        statistics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        statistics["mfe"] = np.mean(y_true - y_pred)
        return statistics     
    
    def get_metrics(self, fittedvalues, data_test):
        return self.compute_metrics(fittedvalues, data_test[self.target])
        
    def compute_BBCCV(self, y_pred, y_true, K=100, seed=0):
        #return a list of rows corresponding to the in/out metrics in different bootstrap folds.
        # note that our bbccv is sampling N-1 samples and not N compared to the original method.
        rng = np.random.default_rng(seed=seed)
        results = []
        for fold in range(K):
            insample_index = rng.choice(y_pred.index, size=len(y_pred)-1, replace=True)
            outsample_index = [x for x in y_pred.index if x not in insample_index]
            insample_p = y_pred.loc[insample_index]
            insample_t = y_true.loc[insample_index]
            outsample_p = y_pred.loc[outsample_index]
            outsample_t = y_true.loc[outsample_index]
            inmetrics = self.compute_metrics(insample_p, insample_t, align=False)
            outmetrics = self.compute_metrics(outsample_p, outsample_t, align=False)
            
            row = dict(("in_"+key, value) for key, value in inmetrics.items())
            row = {**row, "fold":fold}
            row = {**row, **(dict(("out_"+key, value) for key, value in outmetrics.items()))}
            results.append(row)
        return results



class ARDLModel(Estimator):
    support_feature_filtering = False
    is_autoregressive = True

    def __init__(self, config, target):
        self.model = models.ARDLModel(config, target)
    def fit_predict(self, data_train, data_test):
        self.model.fit(data_train)
        fittedvalues = self.model.fittedvalues(data_test)
        return fittedvalues


class SKLearnVectorized(Estimator):
    support_feature_filtering = True
    # Base class for vectorized approaches in sklearn.
    def fit_predict(self, data_train, data_test, selected_features=None):
        X, y, _ = self.prepare_data_vectorize(data_train, selected_features)
        self.model = self.model.fit(X,y)
        X, y, indexes = self.prepare_data_vectorize(data_test, selected_features)
        fittedvalues = self.model.predict(X)
        fittedvalues = pd.Series(fittedvalues, index=indexes)
        return fittedvalues
        
    def prepare_data_vectorize(self,data, selected_features = None):
        # used to vectorize several timesteps in a dimension 1 vector.
        # vectorize
        y = data[self.target].iloc[self.lags:]
        indexes = y.index
        y = y.values
        window_X = [data.values[i:i+self.lags].reshape((-1,)) for i in range(len(data)-self.lags)]
        X = np.array(window_X)
        # if selected features were given, build filter
        if selected_features is not None:
            mask = np.zeros((self.lags, len(data.columns)), dtype=bool)
            column_index = dict([(name, i) for i, name in enumerate(data.columns)])
            for name, lag in selected_features:
                if lag<=self.lags:  # only count lags in the self.lags window
                    mask[-lag,column_index[name]]=True
            mask = mask.reshape((-1,))
            # apply filter
            X = X[:,mask]
        
        return X, y, indexes
    
    def fit(self, data_train, selected_features=None):
        X, y, _ = self.prepare_data_vectorize(data_train, selected_features)
        self.model = self.model.fit(X,y)
    def predict(self, data_test, selected_features=None):
        X, y, indexes = self.prepare_data_vectorize(data_test, selected_features)
        fittedvalues = self.model.predict(X)
        fittedvalues = pd.Series(fittedvalues, index=indexes)
        return fittedvalues

class SVRModel(SKLearnVectorized):
    def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model =  SVR(**config["skconfig"])

class KNeighborsRegressorModel(SKLearnVectorized):
    #same fit_predict as SVRModel, so inheritance is easier.
    def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model = KNeighborsRegressor(**config["skconfig"])
        
class LassoLarsModel(SKLearnVectorized):
     def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model = LassoLars(**config["skconfig"])

class XGBRegressorModel(SKLearnVectorized):
    def __init__(self, config, target):
        self.target = target
        self.lags = config["lags"]
        self.model = XGBRegressor(**config["skconfig"])
        
        
##################################################################
#                                                                #
#          Deep models with pytorch forecasting                  #
#                                                                #
##################################################################


class PytorchForecaster(Estimator):
    
    def _prepare_data(self, data, train=True, offset=0):
        # get dataloader
        orig_to_new = dict([(c,str(i)) for i,c in enumerate(data.columns)])
        data = data.rename(columns=orig_to_new)
        newtarget = orig_to_new[self.target]
        data["time"]=range(offset,len(data)+offset)
        data["groups"]=0
        
        dts = TimeSeriesDataSet(
            data,
            target=newtarget,
            time_idx="time",
            group_ids=["groups"],
            max_encoder_length = self.config["lags"],
            time_varying_unknown_reals = list(data.columns[:-2]),
            time_varying_known_reals = ["time"],
            add_relative_time_idx = False,
            add_target_scales = False,
            add_encoder_length = False,
            target_normalizer = None,
            scalers=dict([(var,None) for var in data.columns if var!=newtarget]),
            )
        
        if train:
            return dts, dts.to_dataloader(train=True, batch_size=32)
        
        indextime = pd.DataFrame({"time":data["time"],"index":data.index})
        return  dts, dts.to_dataloader(train=False, batch_size=320), indextime

    def _create_model(self, dts):
        return None
    
    def fit(self, data):
        dts, train_loader = self._prepare_data(data)
        self.offset=len(data)
        self.model = self._create_model(dts)
        self.lightning_trainer = pl.Trainer(max_epochs=self.config["epochs"],
                                       accelerator="gpu", devices=1,
                                       enable_progress_bar=False,
                                       enable_model_summary=False,
                                       enable_checkpointing=False)
        
        self.lightning_trainer.fit(
            self.model,
            train_dataloaders=train_loader
            )
    
    def predict(self, data):
        dts, val_loader, index_time = self._prepare_data(data, train=False, offset=self.offset)
        res = self.model.predict(val_loader, return_index=True, trainer_kwargs={"accelerator":"gpu", "devices":1})
        y_pred = res.output
        dfinfo = res.index
        y_pred = y_pred.cpu().detach().numpy().flatten()
        dfinfo["y_pred"]=y_pred
        dfinfo = dfinfo.merge(index_time,on="time")
        dfinfo.index = dfinfo["index"]
        return dfinfo["y_pred"]

    def fit_predict(self, data_train, data_test):
        self.fit(data_train)
        return self.predict(data_test)


class TFTModel(PytorchForecaster):
    def _create_model(self, dts):
        model = TemporalFusionTransformer.from_dataset(
            dts,
            loss=RMSE(),
            **self.config["config"]
        )
        return model

class DeepARModel(PytorchForecaster):

    def _prepare_data(self, data, train=True, offset=0):
        # get dataloader
        orig_to_new = dict([(c,str(i)) for i,c in enumerate(data.columns)])
        data = data.rename(columns=orig_to_new)
        newtarget = orig_to_new[self.target]
        data["time"]=range(offset,len(data)+offset)
        data["groups"]=0
        
        cols_without_target = [x for x in data.columns[:-2] if x!=newtarget]
        
        dts = TimeSeriesDataSet(
            data,
            target=newtarget,
            time_idx="time",
            group_ids=["groups"],
            max_encoder_length = self.config["lags"],
            time_varying_unknown_reals = [newtarget],
            time_varying_known_reals = ["time"]+cols_without_target,
            add_relative_time_idx = False,
            add_target_scales = False,
            add_encoder_length = False,
            target_normalizer = None,
            scalers=dict([(var,None) for var in data.columns if var!=newtarget]),
            )
        
        if train:
            return dts, dts.to_dataloader(train=True, batch_size=32)
        
        indextime = pd.DataFrame({"time":data["time"],"index":data.index})
        return  dts, dts.to_dataloader(train=False, batch_size=320), indextime
    def _create_model(self, dts):
        model = DeepAR.from_dataset(
            dts,
            **self.config["config"]
        )
        return model



##################################################################
#                                                                #
#   Create configs for completion and optuna                     #
#                                                                #
##################################################################

def complete_config_from_parameters(name, hyperparameters):
    if name == "ARDLModel":
        config = { "constructor" : {"lags":hyperparameters.get("forecaster_lags", 10),
                                    "order":hyperparameters.get("order", 10),
                                    "causal":True,
                                    "trend":hyperparameters.get("trend", "ct"),
                                    "seasonal":hyperparameters.get("seasonal", False),
                                    "period":hyperparameters.get("period", None),
                                    "missing":"drop"},
                   "fit" : {"cov_type":hyperparameters.get("cov_type", "HC0"),
                            "cov_kwds":hyperparameters.get("cov_kwds", None)}
                 }
    elif name == "SVRModel":
        config = {"lags":hyperparameters.get("forecaster_lags", 10),
                  "skconfig":{"kernel":hyperparameters.get("kernel", "rbf"),
                              "degree":hyperparameters.get("degree", 3),
                              "gamma":hyperparameters.get("gamma", "scale"),
                              "coef0":hyperparameters.get("coef0", 0.),
                              "tol":hyperparameters.get("tol", 0.001),
                              "C":hyperparameters.get("C", 1.0),
                              "epsilon":hyperparameters.get("epsilon", 0.1),
                              "shrinking":True}}
    elif name == "KNeighborsRegressorModel":
        config = {"lags":hyperparameters.get("forecaster_lags", 10),
                  "skconfig":{"n_neighbors":hyperparameters.get("n_neighbors", 5),
                              "weights":hyperparameters.get("weights", "uniform"),
                              "algorithm":hyperparameters.get("algorithm", "auto"),
                              "leaf_size":hyperparameters.get("leaf_size", 30),
                              "p":hyperparameters.get("p", 2),
                              "metric":"minkowski"}}
    elif name == "LassoLarsModel":
        config = {"lags":hyperparameters.get("lags", 10),
                  "skconfig":{"alpha":hyperparameters.get("alpha", 1.),
                              "fit_intercept":True,
                              "fit_path":False}}
    elif name == "XGBRegressorModel":
        config = {"lags":hyperparameters.get("lags", 10),
                  "skconfig":{"n_estimators":hyperparameters.get("n_estimators",10),
                              "verbosity":hyperparameters.get("verbosity",0),
                              "n_jobs":hyperparameters.get("n_jobs",5),
                              "colsample_bytree":hyperparameters.get("colsample_bytree",1),
                              "reg_alpha":hyperparameters.get("reg_alpha",0),
                              "reg_lambda":hyperparameters.get("reg_lambda",1)
                  }}
    elif name == "TFTModel":
        config = {"lags":hyperparameters.get("forecaster_lags", 70),
                  "epochs":hyperparameters.get("epochs", 5),
                  "config":{"hidden_size":hyperparameters.get("hidden_size", 16),
                            "lstm_layers":hyperparameters.get("lstm_layers", 2),
                            "attention_head_size":hyperparameters.get("attention_head_size", 4),
                            "dropout":hyperparameters.get("dropout", 0.1),
                            "hidden_continuous_size":hyperparameters.get("hidden_continuous_size", 8),
                           
                  }
            }
        
    elif name == "DeepARModel":
        config = {"lags":hyperparameters.get("forecaster_lags", 70),
                  "epochs":hyperparameters.get("epochs", 5),
                  "config":{"cell_type" : hyperparameters.get("cell_type", "LSTM"),
                            "hidden_size" : hyperparameters.get("hidden_size", 8),
                            "rnn_layers" : hyperparameters.get("rnn_layers", 2),
                            "dropout" : hyperparameters.get("dropout", 0.1)}
            }
    return config

def generate_optuna_parameters(name, trial):
    hp = dict()
    if name == "ARDLModel":
        hp["forecaster_lags"] = trial.suggest_int("forecaster_lags",5,20,step=1,log=False)
        hp["order"] = hp["forecaster_lags"]
        hp["trend"] = trial.suggest_categorical("trend",["n","t","c", "ct"])
    elif name == "SVRModel":
        hp["forecaster_lags"] = trial.suggest_int("forecaster_lags",5,20,step=1,log=False)
        hp["kernel"] = trial.suggest_categorical("kernel",["linear","rbf","poly", "sigmoid"])
        #hp["degree"] = trial.suggest_int("degree",2,5,1,log=False)
        hp["coef0"] = trial.suggest_float("coef0", 0.0, 2.)
        hp["C"] = trial.suggest_float("C", 0.05, 20., log=True)
    elif name == "KNeighborsRegressorModel":
        hp["forecaster_lags"] = trial.suggest_int("forecaster_lags",5,20,step=1,log=False)
        hp["n_neighbors"] = trial.suggest_int("n_neighbors",3,20,1,log=True)
        hp["weights"] = trial.suggest_categorical("weights",["uniform", "distance"])
        hp["leaf_size"] = trial.suggest_int("leaf_size",20,50,1,log=True)
        hp["p"] = trial.suggest_int("p", 1, 2, 1, log=False)
    elif name == "LassoLarsModel":
        hp["lags"] = trial.suggest_int("lags",5,20,1,log=False)
        hp["alpha"] = trial.suggest_float("alpha", 0.001, 10., log=True)
    elif name == "XGBRegressorModel":
        hp["lags"] = trial.suggest_int("lags",5,96,log=True)
        hp["n_estimators"] = trial.suggest_int("n_estimators",5,100,log=True)
        hp["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.1, 1., log=False)
        hp["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 10., log=False)
        hp["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10., log=True)
    elif name == "TFTModel":
        hp["forecaster_lags"] = trial.suggest_int("forecaster_lags",5,70,step=1,log=False)
        hp["epochs"] = trial.suggest_int("epochs",5,10,step=1,log=False)
        hp["hidden_size"] = trial.suggest_int("hidden_size",8,64,log=True)
        hp["attention_head_size"] = trial.suggest_int("attention_head_size",1,4,step=1,log=False)
        hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.5, log=False)
        hp["hidden_continuous_size"] = trial.suggest_int("hidden_continuous_size",4,32,log=True)
        hp["lstm_layers"] = trial.suggest_categorical("lstm_layers",[1,2,3])
    elif name == "DeepARModel":
        hp["forecaster_lags"] = trial.suggest_int("forecaster_lags",5,70,step=1,log=False)
        hp["epochs"] = trial.suggest_int("epochs",5,10,step=1,log=False)
        hp["cell_type"] = trial.suggest_categorical("cell_type",["LSTM", "GRU"])
        hp["hidden_size"] = trial.suggest_int("hidden_size",8,64,log=True)
        hp["rnn_layers"] = trial.suggest_categorical("rnn_layers",[1,2,3])
        hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.5, log=False)
    return hp


def generate_optuna_search_space(name):
    hp = dict()
    if name == "ARDLModel":
        hp["forecaster_lags"] = [10]
        hp["trend"] = ["c"]
    elif name == "SVRModel":
        hp["forecaster_lags"] = [96]
        hp["kernel"] = ["rbf", "sigmoid"]
        hp["coef0"] = [0.0]
        hp["C"] = [ 0.01, 0.1, 1., 10., 100.]
    elif name == "KNeighborRegressorModel":
        hp["forecaster_lags"] = [96]
        hp["n_neighbors"] = [5,  10,  50]
        hp["weights"] = trial.suggest_categorical("weights",["uniform", "distance"])
        hp["leaf_size"] = [20, 50]
        hp["p"] = [ 1, 2]
    elif name == "LassoLarsModel":
        hp["lags"] = [20]
        hp["alpha"] = [0.001,0.01, 0.1,  1.,  10.]
    elif name == "XGBRegressorModel":
        hp["lags"] = [10]
        hp["n_estimators"] = [10]
        hp["colsample_bytree"] = [0.3, 1.]
        hp["reg_alpha"] = [0.]
        hp["reg_lambda"] = [0.1, 1., 10.]
    elif name == "TFTModel":
        hp["forecaster_lags"] = [96]
        hp["epochs"] = [5,10]
        hp["hidden_size"] = [8,16,32,64]
        hp["attention_head_size"] = [1,2,4]
        hp["dropout"] = [0.2]
        hp["hidden_continuous_size"] = [8]
        hp["lstm_layers"] = [1,2]
    elif name == "DeepARModel":
        hp["forecaster_lags"] = [96]
        hp["epochs"] = [5,10]
        hp["cell_type"] = ["LSTM", "GRU"]
        hp["hidden_size"] = [8, 16, 32, 64]
        hp["rnn_layers"] = [1,2,3]
        hp["dropout"] = [0.1, 0.2, 0.3]
    return hp



