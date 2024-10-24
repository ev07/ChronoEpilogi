import copy
import matplotlib.pyplot as plt
import time
import numpy as np

##
#
#   Main class
#
##




class ChronoEpilogi():
    def __init__(self, config, target, verbosity=0):
        """
        In this version, the first residuals are computed on an autoregressive model.
        Thus, the target variable is by default in the selected set.
        
        config contains:
         - equivalent_version: ["f","fb","fe", "fe_m", "fbg","fbe","fbe_m"] the version of the algorithm to apply.
             "f" is a simple forward, "fb" a forward backward, without equivalent set discovery.
             "fe" is a forward phase with equivalent set during forward phase.
             "fbg" is a forward backward phase before applying a greedy equivalent search
             "fbe" is a forward backward phase before applying a comprehensive (theory-sound) search.
             "fe_m"/"fbe_m" is fe/fbe with a final full model check that the equivalence found are truly non-redundant variables.
         - method: define how the stopping criterion will be computed if applicable.
         - significance_threshold: define the difference in model significance to stop.
         - max_features: the maximal number of selected features
         - association: the class constructor of the multivariate association to use
         - association.config: parameters to give to the association constructor
         - model: the model constructor
         - model.config: the parameters to give to the model constructor.
        config might optionally contains, depending on 
        verbosity set at 1 keeps track of the algorithm whole history.
        """
        
        self.target = target
        self.fitted = False
        self.selected_features = []
        self.config = config
        self.verbosity = verbosity
        self.history = []
        self.association_objects = None
        self.check_config()
        self._prebuild_association_objects()
        
        self.equivalent_variables=dict()
        self.partial_correlation_objects = None
    
    def _reset_fit(self):
        self.selected_features=[]
        self.history = []
        self.fitted = False
        self.equivalent_variables=dict()
    
    def check_config(self):
        # configuration parameters required for forward phase
        assert "association" in self.config
        assert "association.config" in self.config
        assert "max_features" in self.config
        assert "valid_obs_param_ratio" in self.config
        assert "significance_threshold" in self.config
        assert "method" in self.config
        assert "model" in self.config
        assert "model.config" in self.config
        # version of the algorithm (f, fb, fe, fe_m, fbg, fbe, fbe_m)
        assert "equivalent_version" in self.config
        # configuration parameters required for backward phase
        if "b" in self.config["equivalent_version"]:
            assert "significance_threshold_backward" in self.config
            assert "method_backward" in self.config
        # configuration parameters required for equivalent set discovery
        if self.config["equivalent_version"] in ["fe", "fe_m", "fbg", "fbe", "fbe_m"]:
            assert "partial_correlation" in self.config
            assert "partial_correlation.config" in self.config
            assert "equivalence_threshold" in self.config
            assert "equivalence_method" in self.config
    
    def _prebuild_association_objects(self):
        "Initialize association object to respect the main algorithm structure"
        association_constructor = self.config["association"]
        association_object = association_constructor(self.config["association.config"])
        self.association_objects = association_object
        
        # equivalence search objects
        if self.config["equivalent_version"] in  ["fe", "fe_m", "fbg", "fbe", "fbe_m"]:
            partial_corr_constructor = self.config["partial_correlation"]
            partial_correlation_object = partial_corr_constructor(self.config["partial_correlation.config"])
            self.partial_correlation_objects = partial_correlation_object
    
    
    def _stopping_criterion(self, current_model, previous_model, len_selected_features):
        """return True if we should continue to include variables, False to stop"""
        # if enough features were selected, stop
        if len_selected_features >= self.config["max_features"]:
            return False
        # if the number of observations is too low compared to the number of parameters, stop
        if current_model.has_too_many_parameters(self.config["valid_obs_param_ratio"]):
            return False
        # if this is the first iteration, continue
        if len_selected_features == 1:
            return True
        
        threshold = self.config["significance_threshold"]
        metric = current_model.stopping_metric(previous_model, self.config["method"])
        return metric < threshold
        
    def _train_model(self, data, selected_set):
        """Creates a model on the selected variables."""
        current_model = self.config["model"](self.config["model.config"], target=self.target)
        current_model.fit(data[selected_set])

        return current_model
        
    
    def _initialize_fit(self, initial_selected, data):
    
        #prepare the initial regression set
        if initial_selected == []:
            initial_selected = [self.target]
        elif self.target not in initial_selected:
            initial_selected = [self.target] + initial_selected
    
        # initialization of candidate variables
        candidate_variables = set(data.columns)
        for variable in initial_selected:
            candidate_variables.remove(variable)
            # remove equivalent variables if computed already
            if self.config["equivalent_version"] in ["fe", "fe_m"]:
                for equiv in self.equivalent_variables.get(variable,[]):
                    candidate_variables.remove(equiv)
            
        # initialize the N-1 model so that the stopping criterion can be used from the first step
        if len(initial_selected)==1:
            previous_model = None  # will be defined during the first iteration
        else:  # train on the selected, removing the last included variable.
            previous_model = self._train_model(data, initial_selected[:-1])
        
        # initialize the current model
        time_modeltrain_start = time.time()
        current_model = self._train_model(data, initial_selected)
        time_modeltrain_end = time.time()
        
        residuals = current_model.residuals()
        
        return initial_selected, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end


    def _equivalent_set_greedy(self, data, chosen_variable, residuals, candidate_variables):
        """
        Compute for each candidate variable, if it is equivalent to the chosen variable by partial correlation with residuals.
        This version tests for each C in candidate_variables:
          - C indep residuals | chosen_variable
          - chosen_variable indep residuals | C
        If both hold, C is considered equivalent to chosen_variable.
        
        Tests conducted with linear models and the likelihood ratio test.
            X indep R | Y is tested by llr(m1,m2), m1<- R~Y, m2<- R~Y,X
        """
        equivalence_threshold = self.config["equivalence_threshold"]
        equivalent_list = []
        for candidate in candidate_variables:
            candidate_df = data[[candidate]]
            residuals_df = residuals
            condition_df = data[[chosen_variable]]
            pvalue = self.partial_correlation_objects.partial_corr(residuals_df, candidate_df, condition_df)
            if pvalue > equivalence_threshold:  # no relation found between candidate and residuals given condition
                pvalue = self.partial_correlation_objects.partial_corr(residuals_df, condition_df, candidate_df)
                if pvalue > equivalence_threshold:  # no relation found between condition and residuals given candidate
                    equivalent_list.append(candidate)
        return equivalent_list
    
    def _equivalent_set_comprehensive_resid(self, data, chosen_variable, residuals, candidate_variables):
        """
        Compute for each candidate variable, if it is equivalent to the chosen variable by partial correlation with residuals.
        
        Tests conducted with linear models and the likelihood ratio test.
            X indep R | Y is tested by llr(m1,m2), m1<- R~Y, m2<- R~Y,X
        """
        equivalence_threshold = self.config["equivalence_threshold"]
        equivalent_list = []
        for candidate in candidate_variables:
            candidate_df = data[[candidate]]
            residuals_df = residuals
            condition_df = data[[chosen_variable]]
            pvalue = self.partial_correlation_objects.partial_corr(residuals_df, candidate_df, condition_df)
            if pvalue > equivalence_threshold:  # no relation found between candidate and residuals given condition
                equivalent_list.append(candidate)
        return equivalent_list
    
    def _equivalent_set_comprehensive_model(self, data, chosen_variable, candidate_variables, conditioning_set):
        """
        Compute for each candidate variable, if it is equivalent to the chosen variable by granger causality test.
        
        Tests conducted with the provided model and test metric:
            X indep T | C, Z is tested by stopping_metric(m1,m2), m1<- T~C,Z, m2<- T~X,C,Z
            X=chosen_variable, C\in candidate_variables, Z=conditioning_set, T=self.target
        """
        equivalence_threshold = self.config["equivalence_threshold"]
        equivalence_method = self.config["equivalence_method"]
        equivalent_list = []
        for candidate in candidate_variables:
            restricted_model = self._train_model(data, conditioning_set + [candidate])
            full_model = self._train_model(data, conditioning_set + [candidate, chosen_variable])
            pvalue = full_model.stopping_metric(restricted_model, equivalence_method) # low p-value indicates models are different
            if pvalue > equivalence_threshold:  # no relation found between X and T given Y,Z
                equivalent_list.append(candidate)
        return equivalent_list


    def _forward_verbose(self, data, initial_selected=[]):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # train first model with the list of covariate given.
        selected_features, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end = self._initialize_fit(initial_selected, data)
        
        # keep track of the equivalent covariates to each covariate.
        equivalent_variables = self.equivalent_variables

        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            candidate_variable_list = list(candidate_variables) 
            if len(candidate_variable_list)==0:  # verify that we still have candidates
                break
            
            # compute associations
            time_association_start = time.time()
            measured_associations = self.association_objects.association(residuals, data[candidate_variable_list])
            time_association_end = time.time()
            
            chosen_index = np.argmax(measured_associations)
            chosen_variable = candidate_variable_list[chosen_index]
                                  
            # put the chosen variable in the selected feature set
            selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            
            # logging
            new_history_row = {"step": len(selected_features),
            "model": current_model,
            "associations": list(zip(measured_associations, candidate_variable_list)),
            "associations_time": time_association_end - time_association_start,
            "association_chosen":chosen_variable,
            "model_time": time_modeltrain_end - time_modeltrain_start
            }
            
            # compute equivalent set and remove equivalent features
            time_equivset_start = time.time()
            if self.config["equivalent_version"] in ["fe"]:  # version of the algorithm that will require equivalence testing during forward
                equivalent_variables[chosen_variable] = self._equivalent_set_greedy(data, chosen_variable, residuals, candidate_variables)
                for to_remove in equivalent_variables[chosen_variable]:
                    candidate_variables.remove(to_remove)
            time_equivset_end = time.time()
            
            # logging
            new_history_row["equiv_time"] = time_equivset_end - time_equivset_start
            
            
            # compute new model with this variable
            previous_model = current_model
            time_modeltrain_start = time.time()
            current_model = self._train_model(data, selected_features)
            time_modeltrain_end = time.time()
            # change residuals
            residuals = current_model.residuals()
            
            # logging
            # measuring the p-value of the chosen variable model-based test
            new_history_row["stopping_metric_chosen_variable"] = current_model.stopping_metric(previous_model, self.config["method"])
            self.history.append(new_history_row)
            

        # create selected feature set
        self.selected_features = selected_features
        self.equivalent_variables = equivalent_variables
        # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
        if len(self.selected_features)>1:
            if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                removed = self.selected_features.pop(-1)
                if self.config["equivalent_version"] in ["fe", "fe_m"]:
                    self.equivalent_variables.pop(removed)
 
        
    def _forward_from_selected_list(self, data, selected_list):
        """
        Given a list of ordered column names, corresponding to a run of the algorithm with the same config except for the selection threshold,
        run the current configuration by using the list to avoid computing the correlations, shortening the process.
        
        Params:
            data: pd.DataFrame, the dataframe containing the forecasted MTS. Must be ordered by timestamp increasing.
            selected_list: list of str, ordered list of the covariates (columns in the dataframe),
                 that have been insered in order by a previous run of the algorithm with same parameters except the stopping thresholds
                 (significance_threshold, max_features, valid_obs_params_ratio).
        Returns:
            forward_ended
        """
        equivalent_variables = self.equivalent_variables
        forward_ended=False
        
        # initial model
        selected_features, candidate_variables, previous_model, current_model, residuals, _, _ = self._initialize_fit([], data)
        if self.target == selected_list[0]:
            selected_list = selected_list[1:]
        elif self.target in selected_list:
            selected_list.remove(self.target)
            
        # first step, runing each model and verifying that the stopping criterion does not stop the algorithm before end of given sequence
        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            if len(selected_list)==0:  # verify that we still have candidates in the history
                break
            chosen_variable = selected_list.pop(0)
            selected_features.append(chosen_variable)
            
            # compute equivalent set and remove equivalent features
            if self.config["equivalent_version"] in ["fe","fe_m"]:  # version of the algorithm that will require equivalence testing during forward
                equivalent_variables[chosen_variable] = self._equivalent_set_greedy(data, chosen_variable, residuals, candidate_variables)
                for to_remove in equivalent_variables[chosen_variable]:
                    candidate_variables.remove(to_remove)
            
            # update model for stopping criterion
            previous_model = current_model
            current_model = self._train_model(data, selected_features)
            # change residuals
            residuals = current_model.residuals()
            
        else:  # if termination due to stopping criterion, remove the eventual superflous 
            self.selected_features = selected_features
            # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
            if len(self.selected_features)>1:
                if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                    removed = self.selected_features.pop(-1)
                    if self.config["equivalent_version"] in ["fe","fe_m"]:
                        self.equivalent_variables.pop(removed)
            forward_ended = True
        return forward_ended

            

    def _forward(self, data, initial_selected=[]):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # train first model with the list of covariate given.
        selected_features, candidate_variables, previous_model, current_model, residuals, time_modeltrain_start, time_modeltrain_end = self._initialize_fit(initial_selected, data)
        
        # keep track of the equivalent covariates to each covariate.
        equivalent_variables = self.equivalent_variables

        while self._stopping_criterion(current_model, previous_model, len(selected_features)):
            # find maximally associative variable to current residuals
            candidate_variable_list = list(candidate_variables) 
            if len(candidate_variable_list)==0:  # verify that we still have candidates
                break
            
            # compute associations
            measured_associations = self.association_objects.association(residuals, data[candidate_variable_list])
            
            chosen_index = np.argmax(measured_associations)
            chosen_variable = candidate_variable_list[chosen_index]
                                  
            # put the chosen variable in the selected feature set
            selected_features.append(chosen_variable)
            candidate_variables.remove(chosen_variable)
            
            # compute equivalent set and remove equivalent features
            if self.config["equivalent_version"] in [ "fe", "fe_m"]:  # version of the algorithm that will require equivalence testing during forward
                equivalent_variables[chosen_variable] = self._equivalent_set_greedy(data, chosen_variable, residuals, candidate_variables)
                for to_remove in equivalent_variables[chosen_variable]:
                    candidate_variables.remove(to_remove)
            
            # compute new model with this variable
            previous_model = current_model
            current_model = self._train_model(data, selected_features)
            # change residuals
            residuals = current_model.residuals()

        # create selected feature set
        self.selected_features = selected_features
        self.equivalent_variables = equivalent_variables
        # remove the last selected feature if irrelevant and if not the target itself (always send back one variable at least)
        if len(self.selected_features)>1:
            if current_model.stopping_metric(previous_model, self.config["method"]) >= self.config["significance_threshold"]:
                removed = self.selected_features.pop(-1)
                if self.config["equivalent_version"] in [ "fe","fe_m"]:
                    self.equivalent_variables.pop(removed)
    
    def _backward(self, data):
        """
        Backward pass testing each of the covariate for nonzero coefficient.
        The test is Y_t indep X_t-L...X_t-1 | Z_t-L..Z_t-1 where Z are all other covariates including Y_t-L...Y_t-1
        While there is such a change, keep on conducting backward tests.
        
        Params:
            data: pd.DataFrame, containing the time series that will be forecasted. Must be ordered by timestamp increasing.
        """
        selected_set_has_changed = True
        while selected_set_has_changed:
            selected_set_has_changed = False  # flag reset
            selected_features = copy.copy(self.selected_features)
            full_model = self._train_model(data, selected_features)
            for column in [x for x in selected_features if x!=self.target]:  # never remove target
                restricted_model = self._train_model(data, [x for x in selected_features if x!=column])
                threshold = self.config["significance_threshold_backward"]
                metric = full_model.stopping_metric(restricted_model, self.config["method_backward"])
                if metric >= threshold:  # there is no significative difference between the models.
                    self.selected_features.remove(column)
                    selected_set_has_changed = True  # set change flag to true

    def _equivalent_search(self, data):
        # data: pandas dataframe
        #      index is the timestamp
        #      column is the feature name
        
        # train first model with the list of covariate given.
        candidate_variables = set(data.columns)
        for variable in self.selected_features:
            candidate_variables.remove(variable)
        
        equivalent_variables = dict()
        
        for index in range(1,len(self.selected_features)):
            # create the conditioning set model
            if self.config["equivalent_version"]=="fbg":
                selected_variables = self.selected_features[:index]
            elif self.config["equivalent_version"] in ["fbe", "fbe_m"]:
                selected_variables = self.selected_features[:index]+self.selected_features[index+1:]
            current_model = self._train_model(data, selected_variables)
            residuals = current_model.residuals()
            
            chosen_variable = self.selected_features[index]
            
            # compute equivalent set 
            if self.config["equivalent_version"]=="fbg":
                equivalent_variables[chosen_variable] = self._equivalent_set_greedy(data, chosen_variable, residuals, candidate_variables)
            elif self.config["equivalent_version"] in ["fbe", "fbe_m"]:  # different function since C indep X1...Xn verified due to MB property
                # also, here, do the proper test with full model computation, not residual use.
                equivalent_variables[chosen_variable] = self._equivalent_set_comprehensive_model(data, chosen_variable, candidate_variables, selected_variables)
            # remove equivalent variables
            for to_remove in equivalent_variables[chosen_variable]:
                candidate_variables.remove(to_remove)
        
        self.equivalent_variables = equivalent_variables
        
    def _verify_equivalence_relevance(self,data):
        # verify for each equivalent variable if it is truly relevant.
        # allows more control since this property is never checked directly,
        # only indirectly with the backward phase on the reference set.
        statistical_test = "method_backward" if "method_backward" in self.config else "method"
        threshold = self.config["significance_threshold_backward"] if "significance_threshold_backward" in self.config else self.config["significance_threshold"]
        
        for key in self.equivalent_variables:
            covariates = [var for var in self.selected_features if var!=key]
            restricted_model = self._train_model(data, covariates)
            to_remove_list = []
            for candidate in self.equivalent_variables[key]:
                full_model = self._train_model(data, covariates+[candidate])
                metric = full_model.stopping_metric(restricted_model, self.config[statistical_test])
                if metric>=threshold:
                    to_remove_list.append(candidate)
            for to_remove in to_remove_list:
                self.equivalent_variables[key].remove(to_remove)
            
                
    def fit(self, data, initial_selected=[], check_stopping_on_initial=False):
        """
        Fit the specified version of the algorithm.
        Params:
            data: pd.DataFrame, the MTS ordered by timestamp increasing.
            initial_selected (optional): list of str, the list of columns to use as initial set of the forward phase. Default is empty.
            check_stopping_on_initial (optional): boolean, if True, the stopping criterion is applied iteratively to each element of the initial_selected list.
                If the stopping criterion isn't reached, it continues with a standard forward phase using association to choose variables to include.
                If it is reached, the result of the forward set is the subset of initial_selected before the stopping criterion is met.
                If check_stopping_on_initial is False, the algorithm does not check the stopping criterion of the initial_set.
        """
        # reset previous fit
        self._reset_fit()
        
        # forward phase
        forward_ended = False
        if check_stopping_on_initial:
            forward_ended = self._forward_from_selected_list(data, initial_selected)
        if not forward_ended:
            if self.verbosity:
                self._forward_verbose(data, initial_selected)
            else:
                self._forward(data, initial_selected=initial_selected)
        
        # backward phase
        if "b" in self.config["equivalent_version"]:
            self._backward(data)
        
        # post-equivalent search
        if self.config["equivalent_version"] in ["fbg", "fbe", "fbe_m"]:
            self._equivalent_search(data)
        
        # final relevance verification
        if self.config["equivalent_version"] in ["fbe_m", "fe_m"]:
            self._verify_equivalence_relevance(data)
        
        self.fitted = True
    
    def transform(self, data):
        if self.fitted:
            return data[self.selected_features]
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def get_selected_features(self):
        if self.fitted:
            return self.selected_features




class ChronoEpilogi_train_val(ChronoEpilogi):
    """
    Adaptation to models that need validation sets for performance estimation.
    Not used in the paper.
    """    
    
    def check_config(self):
        super().check_config()
        assert "validation_ratio" in self.config
    def _train_val_data_split(self, data):
        validation_size = int(self.config.get("validation_ratio", 0.1)*len(data))
        data_train = data.iloc[:-validation_size]
        data_test = data.iloc[-validation_size:]
        return data_train, data_test
    def _train_model(self, data, selected_set):
        """Creates a model on the selected variables.
        In this version, also uses a train val split to compute residuals."""
        current_model = self.config["model"](self.config["model.config"], target=self.target)
        data_train, data_test = self._train_val_data_split(data[selected_set])
        current_model.fit(data_train)
        _=current_model.residuals(data_test, test=True)
        return current_model
