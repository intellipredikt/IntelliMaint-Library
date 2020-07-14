# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:16:45 2020

@author: Ajay
"""

#%% 
import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector

#%% Feature selection (Corelation features- Used when very less amount of features)
class feature_selection:
    def __init__(self,data):
        self.data=data
        
    def correlated_feature_sel(self,data,target):
        """
        Correlated feature selection:
        This function will select correlated features based on target feature in descending order.\n
        
        Input Parameters
        ----------------
        data,target:
            data : Input data (format: Dataframe) \n
            target : Target name (format : Str) \n
        Returns
        -------
            features : Correlated feature based on target feature in descending order \n
        """
        data=data.select_dtypes(exclude=[object]) # removing str/object columns
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = data.corr()
        c=corr_matrix[data.columns==target]
        c=c.iloc[0].sort_values(axis=0,ascending=False)
        features=c.index[1:]
        return features

#% To check function
# corr_features = correlated_feature_sel(data, 'y')
# print('correlated features: ', len(set(corr_features)))

#%% Select KBest Score
    def univariate_statistic(self,data,target,score_func=f_classif,method=SelectKBest,k=5):
        """
        Univariate statistics feature selection:
        This function works by selecting the best features based on the univariate statistical tests (ANOVA). \n
        
        Input Parameters
        ----------------
        data,target,score_func,method,k=5:
            data : Input data (format: Dataframe) \n
            target : Target name (format : Str) \n
            score_func : Scoring function (For classiifer: f_classif, For regression: f_regression) \n
            method : Method (For classiifer : SelectKBest, For Reggresion: SelectPercentile) \n
            k : Number of feature selected (Features are in descending order \n
        Returns
        -------
        f_col, sel_fea_data:
            f_col : Selected n feature \n
            sel_fea_data : Selected n features transformed data \n
        """
        data=data.select_dtypes(exclude=[object]) # removing str/object columns
        x = data[data.columns[data.columns!=target]]
        sel_fea = method(score_func=score_func, k=k).fit(np.array(x), data[target].values)
        f_col=x.columns[sel_fea.get_support()]
        
        if  isinstance(k, int):
            f_col= f_col[:k]
            sel_fea_data = sel_fea.transform(x[:k])    
        elif isinstance(k, float):
            print('Feature selected number is invalid')
        else:
            sel_fea_data = sel_fea.transform(x)
        return f_col, sel_fea_data
    
# #% To check function
# f_col, sel_fea_data=Univariate_Statistic(data,'y',k=5)

#%% Wrapper based feature selection
    def feature_selector(self,data,target,model,n_features=5,threshold=0.95,forward=True,scoring='roc_auc',cv=3):
        """
        Feature selector has two algorithm:
        Step forward feature selection: The performance of the classifier is evaluated
        with respect to each feature. The feature that performs the best is 
        selected out of all the features. \n
    
        In the second step, the first feature is tried in combination with all the other features. 
        The combination of two features that yield the best algorithm performance is selected. 
        The process continues until the specified number of features are selected.
        
        Step backward feature selection:  This algorithm is similar to forward selection.Here
        feartures are selected from back. \n
        Input Parameters
        ----------------
        data,target,model,n_features,forward,scoring,cv):
            data : Input data (format: Dataframe) \n
            target : Target name (format : Str) \n
            model : Classification/Reggresion model (Any scikit-learn classifier or regressor) \n
            n_features : Number of features to select, where n_features < input feature set (default=5)\n
            threshold  : Threshold is used to remove highly correlated features (default=0.9, format=int) \n
            forward : Forward selection if True, backward selection otherwise (default= True) \n
            scoring :  It uses a sklearn scoring metric string identifier, for example {accuracy, f1, precision, recall, roc_auc}
                    for classifiers, {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error', 'median_absolute_error', 'r2'}
                    for regressors. (default :roc_auc) \n
            cv :  k-fold cross-validation is performed (default=3) \n
        Returns
        -------
        selected_feat,features:
            selected_feat : Selected features names\n
            features : Sequential feature selector api \n
        """
        data=data.select_dtypes(exclude=[object])
        corr_features = set()  # Set of all the names of correlated columns
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    corr_features.add(colname)
    
        data.drop(labels=corr_features, axis=1, inplace=True)
        print('Removing',len(set(corr_features)),' highly correlated feature')
        feature_selector = SequentialFeatureSelector(model,k_features=n_features,forward=forward,verbose=2,
                scoring=scoring,cv=cv)
        features = feature_selector.fit(np.array(data[data.columns[data.columns!=target]]), data[target].values)
        selected_feat= data.columns[list(features.k_feature_idx_)]
        print('%%%-------------Feature Selection process completed-----------------%%%')
        print('Selected Features:', selected_feat)
        return selected_feat,features
    
    def exhaustive_feature_selector(self,data,target,model,min_features=2,max_features=4,threshold=0.95,scoring='roc_auc',cv=3):
        """
        Exhaustive Feature selector \n 
        The performance of a machine learning algorithm is evaluated against all possible
        combinations of the features in the dataset. The feature subset that yields best
        performance is selected. The exhaustive search algorithm is the most greedy algorithm 
        of all the wrapper methods since it tries all the combination of features and selects the best. \n
    
        A downside to exhaustive feature selection is that it can be slower compared to step forward and
        step backward method since it evaluates all feature combinations \n
        
        Input Parameters
        ----------------
        data,target,model,min_features,max_features,threshold,scoring,cv): 
            data : Input data (format: Dataframe) \n
            target : Target name (format : Str) \n
            model : Classification/Reggresion model (Any scikit-learn classifier or regressor) \n
            min_features : Minimun number of features (default = 2, format : Int) \n
            max_features : Maximum number of features (default = 4, format : Int) \n        
            threshold  : Threshold is used to remove highly correlated features (default=0.9, format=int) \n        
            scoring :  It uses a sklearn scoring metric string identifier, for example {accuracy, f1, precision, recall, roc_auc}
                    for classifiers, {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error', 'median_absolute_error', 'r2'}
                    for regressors. (default :roc_auc) \n
            cv :  k-fold cross-validation is performed (default=3) \n
        Returns
        -------
        selected_feat,features:
            selected_feat : Selected features \n
            features : Exhaustive feature selector api \n
        """
        data=data.select_dtypes(exclude=[object]) # removing str/object columns
        corr_features = set()  # Set of all the names of correlated columns
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    corr_features.add(colname)
    
        data.drop(labels=corr_features, axis=1, inplace=True)
        print('Removing',len(set(corr_features)),' highly correlated feature')
        feature_selector = ExhaustiveFeatureSelector(model,min_features=2,max_features=4,
                        scoring=scoring,print_progress=True,cv=2)
        features = feature_selector.fit(np.array(data[data.columns[data.columns!=target]]), data[target].values)
        selected_feat= data.columns[list(features.best_idx_)]
        print('%%%-------------Feature Selection process completed-----------------%%%')
        # print('Selected Features:', selected_feat)
        return selected_feat,features
    
# #% To check function
# a,b=Feature_Selector(data,'y',RandomForestClassifier(),scoring='roc_auc',cv=3)
    
#%%  Univariate feature selection using various sklearn classifer/regressor
    def univariate_features(self,data,target,model,method,k):
        """
        Univariate feature selection \n
        This function will sort features in descending order. It uses sklearn classifier or regression methods
        to sort based on sklearn metrics.
        
        Input Parameters
        ----------------
        data,target,model,method,k):
            data : Input data (format: Dataframe) \n
            target : Target name (format : Str) \n
            model : Classification/Reggresion model (Any scikit-learn classifier or regressor) \n
            method : Scoring function to decide feature selection (sklearn metrics) \n
                    Usually for Classification: roc_auc_score and for regression mean_squared_error is used\n
            k : Number of feature selected (Features are in descending order \n
        Returns
        -------
        fe_sel:
            fe_sel : First n selected features in descending order \n
        """
        data=data.select_dtypes(exclude=[object]) # removing str/object columns
        scoring = []
        x_train, x_test, y_train, y_test = train_test_split(
        data.drop(labels=target, axis=1),data[target],test_size=0.3,random_state=0)
        for feature in x_train.columns:
            model.fit(x_train[feature].to_frame(), y_train)
            pred = model.predict(x_test[feature].to_frame())
            scoring.append(method(y_test, pred))
        scoring = pd.Series(scoring)
        scoring.index = x_train.columns
        scoring=scoring.sort_values(ascending=False)
        if  isinstance(k, int):
            fe_sel=scoring[:k]
        elif isinstance(k, float):
            print('Feature selected number is invalid')
        else:
            fe_sel=scoring
        return fe_sel
    
#% To check function
# a=Univariate(data,'y',RandomForestClassifier(),metrics.accuracy_score,10)
    
#%% Remove constant and quasi constant features, duplicate features and highly 
### corelated features.
    def feature_preprocessing(self,data,var_threshold=0.01,cor_threshold=0.95):
        """
        This function will remove four kinds of unwanted features from the input dataset. \n
        1) Constant features : Features which have constant value i.e same number in a column \n
        2) Quasi-constant features : Features which are almost constant. In other words, 
            these features have the same values for a very large subset of the outputs. \n
        3) Duplicate features : Features which has duplicate values or similar values\n
        4) Highly Correlated features : Features which are highly correlated and doesn't make any sence 
            in prediction. This process is carried by setting the cor_threshold \n 
        Input Parameters
        ----------------
        data, var_threshold, cor_threshold:
            data : Input data (format: Dataframe) \n
            var_threshold : Variance threshold is set for removing quasi constant features (default: 0.01) \n
            cor_threshold : Correlation threshold is set for removing highly correlated features (default: +/-0.95) \n
        Returns
        -------
        Data:
            Data : Returns data after feature removal \n
        """
        # Removing object/str columns from dataframe
        data=data.select_dtypes(exclude=[object])
        print('------------Searching for Constant features------------------')
        constant_features = [feat for feat in data.columns if data[feat].std() == 0]
        data=data.drop(labels=constant_features, axis=1)
        print('Removing', len(constant_features), 'constant features') 
        print('------------Searching for Quasi-constant features------------')
        sel = VarianceThreshold(threshold=var_threshold)  # 0.1 indicates 99% of observations are similar approximately
        sel.fit(data)  # fit finds the features with low variance
        features_selected = data.columns[sel.get_support()]
        a=len(data.columns)-len(features_selected)
        print('Removing',a,'Quasi-Constant features')
        data=pd.DataFrame(sel.transform(data),columns=features_selected)
        print('------------Searching for Duplicate features-----------------')
        duplicated_feat = []
        for i in range(0, len(data.columns)):
            col_1 = data.columns[i]
            for col_2 in data.columns[i + 1:]:
                if data[col_1].equals(data[col_2]):
                    duplicated_feat.append(col_2)
        data=data.drop(labels=duplicated_feat, axis=1)
        print('Removing',len(duplicated_feat),'Duplicate features')
        print('------------Searching for Highly correlated features---------')
        corr_features = set()  # Set of all the names of correlated columns
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >= cor_threshold:
                    colname = corr_matrix.columns[i]  # getting the name of column
                    corr_features.add(colname)
        print('Removing',len(corr_features),'correlated features based on threshold')
        data=data.drop(labels=corr_features, axis=1)
        print('Total features removed:',len(constant_features)+a+len(duplicated_feat)+len(corr_features))
        return data
    
# To check function
# d=Feature_Preprocess(f.fillna(0),cor_threshold=.999)
    



