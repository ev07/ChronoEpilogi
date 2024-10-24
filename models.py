from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults, AutoRegResultsWrapper
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

from scipy.stats import f as fdistrib, wilcoxon, chi2

from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
import pandas as pd
import pydash


import baselines.estimators


# custom error for ARDL model

class NotEnoughDataError(ValueError):
    def __init__(self, datasize, lags, orders):
        self.message = "Model needs more lags of the data than provided to make predictions.\nPlease consider increasing the test data size.\nData length: {}, number of lags: {}, number of orders: {}".format(datasize, lags, orders)
        super().__init__(self.message)
    


################################
#
#   Learning wrapper
#
################################


class LearningModel:
    def __init__(self, config, target):
        self.config = config
        self.target = target
        
        self.data = None
        self.model = None

    #
    # part that need to be implemented for each learning model    
    #
    
    def fit(self, data):
        pass

    def fittedvalues(self, data=None):
        # return the fitted values of the model.
        # should be a pd.Series with corresponding index to original data
        # the series should not contain NaN timestamps.
        pass
        
    def stopping_metric(self, previous_model, method):
        # should return a metric that corresponds more or less to p-values.
        # the lower, the more incentive to keep adding new variables to the selected set
        pass
    
    def has_too_many_parameters(self, ratio):
        # part of the stopping criterion: verify if there are ratio times more timestamps in the data
        # than parameters in the model.
        pass
        
    #
    # part that can be let as-is or modified after inheritance.
    #
    
    def residuals(self, data=None):
        """
        Compute residuals.
        Uses training data if "data" argument is None, otherwise specified data is used.
        
        Params:
            data (optional): None or pd.DataFrame, where data.columns should be identical to self.data.columns
        Returns:
            pd.dataframe with a single column corresponding to the target. The index of this dataframe coincidates
            with the index of the "data" dataframe, over the forecasted points.
        """
        # output should be a pd.DataFrame, rows index should correspond to the original data
        # the series should not contain NaN timestamps.
        if data is None:
            data = self.data
        fittedvalues = self.fittedvalues(data)
        targetdata = data[self.target]
        targetdata = targetdata[fittedvalues.index]
        residuals = targetdata - fittedvalues
        # make the results a dataframe adressable by target.
        df = pd.DataFrame({self.target: residuals})
        return df

    def sse(self, data=None):
        """
        Compute the Sum of Squared Errors of the residuals of the target series.
        """
        residuals = self.residuals(data)[self.target]
        return pd.Series.sum(residuals**2)

    def total_variation(self, data=None):
        """
        Compute the Total Variation of the target series, for all timestamps for which residuals are computed.
        Total Variation = len(series) * series.std()**2
        Uses training data if "data" argument is None, otherwise specified data is used.
        
        Params:
            data (optional): None or pd.DataFrame, where data.columns should be identical to self.data.columns
        """
        indexes = self.residuals(data).index  # only take timestamps over which residuals exist
        if data is None:
            data = self.data
        originals = data.loc[indexes][self.target]
        return len(indexes)*(originals.std()**2)

    def statistics(self, data=None):
        # how to get statistics for the training set, or a test set.
        # current global statistics are sse, rmse, R2, mfe, mape. 
        # They can be obtained only from the data and the fitted values.
        if data is None:
            data = self.data
        
        statistics = dict()
        residuals = self.residuals(data)
        statistics["sse"] = self.sse(data)
        statistics["rmse"] = np.sqrt(statistics["sse"]/len(residuals))
        statistics["R2"] = 1 - statistics["sse"]/self.total_variation(data)
        statistics["mfe"] = np.mean(residuals[self.target])
        
        fittedvalues = self.fittedvalues(data)
        rangefitted = fittedvalues.index  # use only timestamps for which we have outputs
        statistics["mape"] = mean_absolute_percentage_error(data.loc[rangefitted][self.target], fittedvalues)
        
        return statistics


################################
#
#   ARDL model wrapper
#
################################



class ARDLModel(LearningModel):
    """
    Adapted from statsmodels.tsa.ardl.ARDL.
     - due to arguments in both instance declaration and fit routine, config must contain:
       - arguments to pass to ARDL constructor
       - arguments to pass to ARDL.fit
     - does not use lag estimation
     - target variable is included in the observed variables
     - uses minus aic of final model as significance

    config (dict):
     - "constructor" (dict): arguments to pass to the ARDL constructor
       - see https://www.statsmodels.org/dev/generated/statsmodels.tsa.ardl.ARDL.html#statsmodels.tsa.ardl.ARDL
     - "fit" (dict): arguments to pass to the ARDL.fit method
       - see https://www.statsmodels.org/dev/generated/statsmodels.tsa.ardl.ARDL.fit.html#statsmodels.tsa.ardl.ARDL.fit
    """

    def __init__(self, config, target):
        super().__init__(config, target)
        self.results = None  # to store the ARDLResults instance

    def fit(self, data):
        """Make sure that number of parameters are enough compared to the data size
        """
        if isinstance(self.config["constructor"]["order"], int) or isinstance(self.config["constructor"]["order"], float):
            maxlag = self.config["constructor"]["order"]
        else:
            maxlag = max(self.config["constructor"]["order"])
        maxlag = max([maxlag, self.config["constructor"]["lags"]])
        if len(data.index) - maxlag < maxlag*len(data.columns)+4:
            raise NotEnoughDataError(len(data.index), self.config["constructor"]["lags"], self.config["constructor"]["order"])
    
        self.data = data
        self.model = self.createModel(data)
        self.results = self.model.fit(**self.config["fit"])
        
    def createModel(self, data):
        """
        Creates the model from the data provided.
        Created model is not trained.
        """
        if len(data.columns)>1:
            model = ARDL(endog=data[self.target],
                              exog=data.loc[:, data.columns != self.target],
                              **self.config["constructor"])
        else:
            model = ARDL(endog=data[self.target],
                              exog=None,
                              order=None,
                              **pydash.omit(self.config["constructor"], "order"))
        return model

    def stopping_metric(self, previous_model, method):
        """
        Computes the metric associated to the model type.
        The lower the better the new model
        """
        metric = None
        if method == "aic":  # compare models significances
            previous_model_significance = previous_model.aic()
            current_model_significance = self.aic()
            metric = current_model_significance - previous_model_significance
            
        elif method == "f-test":
            constraint_matrix = []
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    new_constraint = np.zeros((len(self.results.params.index),))
                    new_constraint[i]=1
                    constraint_matrix.append(new_constraint)
            r_matrix = np.array(constraint_matrix)
            metric = self.results.f_test(r_matrix).pvalue
            
        elif method == "by_hand_f-test":
            fstat_top = (previous_model.sse() - self.sse()) / (previous_model.dof() - self.dof())
            fstat_bot = self.sse() / self.dof()
            fstat = fstat_top / fstat_bot
            pvalue = 1 - fdistrib.cdf(fstat, previous_model.dof() - self.dof(), self.dof())
            metric = 0 if np.isnan(pvalue) else pvalue
        
        elif method == "wald-test":
            constraint_matrix = []
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    new_constraint = np.zeros((len(self.results.params.index),))
                    new_constraint[i]=1
                    constraint_matrix.append(new_constraint)
            r_matrix = np.array(constraint_matrix)
            metric = self.results.wald_test(r_matrix, use_f=False, scalar=True).pvalue
        
        elif method == "lr-test":
            diff_dof = 0
            for i, param_name in enumerate(self.results.params.index):
                if param_name not in previous_model.results.params.index:
                    diff_dof+=1
            cstat = -2*(previous_model.llh() - self.llh())
            metric = chi2.sf(cstat,df=diff_dof)
        
        return metric

    def _pad_test_data_to_create_model(self, data):
        """
        The not-so-nice thing about creating a copy model for test data,
        is that model instanciation checks that the data is large enough to learn.
        Hence, when test data is large enough to be evaluated (timesteps > lags)
        but not enough to be learned (timesteps < lags*variables + 1), it blocks.
        The solution I found is to simply pad the data. The fittedvalue method will select
        the right timestamps at the end.
        This method is handling the padding.
        
        Returns:
             the padded or nonpadded data
        Raises NotEnoughDataError if the data is too small for even 1 prediction.
        """
        if len(data)<=self.config["constructor"]["lags"] or len(data)<=self.config["constructor"]["order"]:
            # estimation is impossible, not enough descriptors for 1 prediction
            raise NotEnoughDataError(len(data), self.config["constructor"]["lags"], self.config["constructor"]["order"])
        
        period = self.config["constructor"]["period"] if "period" in self.config["constructor"] else 1
        period = period+1 if period is not None else 2
        needed_regressors = (len(data.columns) - 1)*self.config["constructor"]["order"] 
        needed_regressors += self.config["constructor"]["lags"] + 4
        needed_regressors *= period
        if needed_regressors - len(data)>0:
            zeros = np.zeros((needed_regressors - len(data), len(data.columns)))
            index = range(len(zeros))
            zeros = pd.DataFrame(zeros, columns=data.columns, index=index)
            data = pd.concat([data, zeros])
            
        return data

    def fittedvalues(self,data=None):
        if data is not None:
            index = data.index  # keep track of original index
            pad_data = self._pad_test_data_to_create_model(data)  # pad just in case test size is small
            pad_data = pad_data.reset_index(drop=True)  # predict works better for rangeindex starting at 0
            model = self.createModel(pad_data)
            # use previous parameters 
            fittedvalues = model.predict(self.results._params, start=0,end=len(data)-1, dynamic=False)
            fittedvalues_nona = fittedvalues.dropna()
            fittedvalues_nona.index = index[fittedvalues_nona.index]
            if len(fittedvalues_nona)==0: # debug handling
                print(fittedvalues_nona)
                print(fittedvalues)
                print(pad_data)
                print("\n")
            return fittedvalues_nona
        else:
            return self.results.fittedvalues

    
    def residuals(self, data=None):
        if data is not None:
            df = super().residuals(data)
        else:
            residuals = self.results.resid.dropna()
            df = pd.DataFrame({self.target: residuals})
        return df

    def aic(self):
        return self.results.aic
        
    def llh(self):
        return self.results.llf

    def dof(self):
        return self.results.df_resid
    
    def has_too_many_parameters(self, ratio):
        nbparams = len(self.results.params)
        nobs = self.results.nobs
        return nobs/nbparams<ratio
    
    def statistics(self, data=None):
        # handle statistics on both training and test sets
        # the .apply method does not work for distributed lags, so I have to rely on a hack.
        # also, I did not implement llf computation for the test model.
        statistics = super().statistics(data)
        if data is None:
            #add train-specific stats
            statistics["llh"] = self.llh()
            statistics["aic"] = self.aic()
        return statistics
        



#########################
#
#   SVR model for ChronoEpilogi_train_val. Not used in the paper.
#
#########################

class SVRModel(LearningModel):
    def __init__(self, config, target):
        super().__init__(config, target)
        self.test_fittedvalues = None
        self.test_residuals = None

    def fit(self, data):
        self.model = baselines.estimators.SVRModel(self.config, self.target)
        self.model.fit(data, None)

    def fittedvalues(self, data=None, test=True):
        # data=None, test=True: if test fittedvalues computed already, return it, otherwise return train fittedvalues
        # data=None, test=False: compute the train fittedvalues
        # data given, test=True: if test fittedvalues computed already, return it, otherwise compute it and send back
        # data given, test=False: compute the fittedvalues for that given dataset.
        no_data_given = data is None
        if test and self.test_fittedvalues is not None:
            return self.test_fittedvalues
        if no_data_given:
            data = self.data
        fittedvalues = self.model.predict(data, None)
        if test and not no_data_given:
            self.test_fittedvalues = fittedvalues
        return fittedvalues
    
    def residuals(self, data=None, test=True):
        # data=None, test=True: if test residuals computed already, return it, otherwise return train residuals
        # data=None, test=False: compute the train residuals
        # data given, test=True: if test residuals computed already, return it, otherwise compute it and send back
        # data given, test=False: compute the residuals for that given dataset.
        no_data_given = data is None
        if test and self.test_residuals is not None:
            return self.test_residuals
        fittedvalues = self.fittedvalues(data, test=test)
        if no_data_given:
            data = self.data
        
        targetdata = data[self.target]
        targetdata = targetdata[fittedvalues.index]
        residuals = targetdata - fittedvalues
        # make the results a dataframe adressable by target.
        df = pd.DataFrame({self.target: residuals})
            
        if test and not no_data_given:
            self.test_residuals = df
        return df
    
    def stopping_metric(self, previous_model, method):
        # should return a metric that corresponds more or less to p-values.
        # the lower, the more incentive to keep adding new variables to the selected set
        if method=="rmse_diff":
            residuals_full = self.residuals(test=True)[self.target]
            residuals_partial = previous_model.residuals(test=True)[self.target]
            rmse_full = np.sqrt(pd.Series.sum(residuals_full**2))/len(residuals_full)
            rmse_partial = np.sqrt(pd.Series.sum(residuals_partial**2))/len(residuals_partial)
            return rmse_full - rmse_partial
    
    def has_too_many_parameters(self, ratio):
        # part of the stopping criterion: verify if there are 10 times more timestamps in the data
        # than parameters in the model.
        # for now unused
        return False

