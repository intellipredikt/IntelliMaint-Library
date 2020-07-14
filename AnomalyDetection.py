# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:23:23 2020

@author: IPTLP0018
@author: anup
"""
from IntelliMaint.anomaly.individual_anomaly import IndividualAnomalyInductive, IndividualAnomalyTransductive
from IntelliMaint.anomaly.group_anomaly import GroupAnomaly
import pandas as pd
from IntelliMaint.anomaly.group_anomaly.datasets import data_loader

class AnomalyDetection:
    """
    AnomalyDetection:
        Detect Anolamies in Streaming / Historical Time Series Data for Indivudual Source or Group of Sources
            For Individual Stationary Data:
                    usage: AnomalyDetection.statinaryAnalysis(df,training_window,method_name,neighbours,w_deviation,dev_threshold,columns, plot)
                    
            For Individual Non Stationary Data:
                    usage: AnomalyDetection.nonstationaryPeriodic(df, ref_group, external_percentage, method_name, neighbours, w_deviation, dev_threshold, plot)
                    
            For Historical Time Series:
                    usage: AnomalyDetection.groupAnomaly(dataPath, nb_units, ids_target_units, transformer, w_transform, w_ref_group, method_name, neighbours, w_deviation, dev_threshold, plot)
    """

    def __init__(self):
        pass
    
    
    def stationaryAnalysis(self, df, training_window = 80, method_name = "knn", neighbours = 20, w_deviation = 15, dev_threshold = 0.6, columns = None, plot='Y'):

        """
        StationaryAnalysis: Detect Anomalies for Individual Stationary Data
        
        Input Parameters:
        ----------------
                df: data
                
                training_window(int): Initial Number of rows without anomalies to be considered for reference
                
                method_name(char): # Strangeness measure: "median" or "knn" or "lof"
                
                neighbours(int): Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
                
                w_deviation(int): Window used to compute the deviation level based on the last w_deviation samples.Window used to compute the deviation level based on the last w_martingale samples
                
                dev_threshold(float): Threshold in [0,1] on the deviation level
                
                plot(char): Plot Required 'Y' / 'N'
                
        Returns:
        -------
            dictionary:  Time, Strangeness, Pvalues, Deviation
            
                        strangeness : Strangeness of x with respect to samples in Xref
        
                        Pvalues : p-value that represents the proportion of samples in Xref that are stranger than x.
        
                        Deviation : Normalized deviation level updated based on the last w_deviation steps

        """
        info_dict = []
        cols = ['Time', 'Strangeness', 'P-Value', 'Deviation']
        # Create a model using IndividualAnomalyInductive
        model = IndividualAnomalyInductive(non_conformity = method_name, k = neighbours, w_martingale = w_deviation, dev_threshold = dev_threshold, columns = None)
        model.fit(df.head(n=training_window).values)
        # At each time step t, a data-point x comes from the stream
        for t, x in zip(df.index, df.values):
            info = model.predict(t, x)
            print("Time: {} ==> strangeness: {}, deviation: {}".format(t, info.strangeness, info.deviation), end="\r")
            info_dict.append({'Time': t, 
                            'Strangeness': info.strangeness, 
                            'P-Value': info.pvalue, 
                            'Deviation': info.deviation})
        df_new = pd.DataFrame(info_dict, columns=cols)
        df_new = df_new.set_index('Time')

        # Plot strangeness and deviation level over time
        if(plot == 'Y'):
            model.plot_deviations(figsize=(8, 6), plots=["strangeness", "deviation", "pvalue", "threshold"])
            
        return df_new


    def nonstationaryPeriodic(self, df, ref_group = ["day-of-week"], external_percentage = 0.3, method_name = "knn", neighbours = 20, w_deviation = 15, dev_threshold = 0.6, plot="Y"):
        """
        nonstationaryPeriodic: Detect Anomalies for Individual Non Stationary Data
        
        Input Parameters:
        ----------------
                df: data
                
                ref_group('char'): ["hour-of-day", "day-of-week", "day-of-month", "week-of-year", "month-of-year", "season-of-year"] 
                
                method_name(char): # Strangeness measure: "median" or "knInitial Number of rows without anomalies to be considered for referencen" or "lof"
                
                neighbours(int): Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
                
                w_deviation(int): Window used to compute the deviation level based on the last w_deviation samples.Window used to compute the deviation level based on the last w_martingale samples
                
                dev_threshold(float): Threshold in [0,1] on the deviation level
                
                plot(char): Plot Required 'Y' / 'N'
                
        Returns:
        -------
            dictionary:  Time, Strangeness, Pvalues, Deviation
            
                        strangeness : Strangeness of x with respect to samples in Xref
        
                        Pvalues : p-value that represents the proportion of samples in Xref that are stranger than x.
        
                        Deviation : Normalized deviation level updated based on the last w_deviation steps

        """

        info_dict = []
        cols = ['Time', 'Strangeness', 'P-Value', 'Deviation']
        model = IndividualAnomalyTransductive(ref_group = ref_group, external_percentage = external_percentage, non_conformity = method_name, k = neighbours, w_martingale = w_deviation, dev_threshold = dev_threshold)
        
        for t, x in zip(df.index, df.values):
            info = model.predict(t, x)
            print("Time: {} ==> strangeness: {}, deviation: {}".format(t, info.strangeness, info.deviation), end="\r")
            info_dict.append({'Time': t, 
                            'Strangeness': info.strangeness, 
                            'P-Value': info.pvalue, 
                            'Deviation': info.deviation})
        df_new = pd.DataFrame(info_dict, columns=cols)
        df_new = df_new.set_index('Time')

        # Plot strangeness and deviation level over time
        if(plot == 'Y'):
            model.plot_deviations(figsize=(8, 6), plots=["strangeness", "deviation", "pvalue", "threshold"])
            
        return df_new


    def groupAnomaly(self, dataPath, nb_units = 0, ids_target_units = 0, transformer = None, w_transform = 30, w_ref_group = "7days", method_name = "median", neighbours = 20, w_deviation = 15, dev_threshold = 0.6, plot="Y"):
        """
        groupANomaly: Detect Anomalies for Group
        
        Input Parameters:
        ----------------
                dataPath: Path to the dataset
                
                nb_units(int): Number of units. Must be equal to len(x_units), where x_units is a parameter of the method self.predict
                
                ids_target_units(list): List of indexes of the target units (to be diagnoised). Each element of the list should be an integer between 0 (included) and nb_units (excluded).

                w_ref_group(string): Time window used to define the reference group, e.g. "7days", "12h" ...
                                     Possible values for the units can be found in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html

                method_name(char): # Strangeness measure: "median" or "knInitial Number of rows without anomalies to be considered for referencen" or "lof"
                
                neighbours(int): Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
                
                w_deviation(int): Window used to compute the deviation level based on the last w_deviation samples.Window used to compute the deviation level based on the last w_martingale samples
                
                dev_threshold(float): Threshold in [0,1] on the deviation level
                
                plot(char): Plot Required 'Y' / 'N'
                
        Returns:
        -------
            dictionary:  Time, Strangeness, Pvalues, Deviation
            
                        strangeness : Strangeness of x with respect to samples in Xref
        
                        Pvalues : p-value that represents the proportion of samples in Xref that are stranger than x.
        
                        Deviation : Normalized deviation level updated based on the last w_deviation steps

        """

        dataset = data_loader.DataCSVs(dataPath).load()
        cols = ['Time', 'Strangeness', 'P-Value', 'Deviation']
        list_of_components = [[] for _ in range(nb_units)]
        list_of_dfs = []
        
        model = GroupAnomaly(nb_units = nb_units, ids_target_units = ids_target_units, transformer = transformer, w_transform = w_transform, w_ref_group = w_ref_group, non_conformity = method_name, k = neighbours, w_martingale = w_deviation)

        for dt, x_units in dataset.stream():
            devContextList = model.predict(dt, x_units)
            for uid, devCon in enumerate(devContextList):
                list_of_components[uid].append({'Time': dt,
                'Strangeness': devCon.strangeness,
                'P-Values':devCon.pvalue,
                'Deviation':devCon.deviation})

        for item in list_of_components:       
            df = pd.DataFrame(item, columns=cols)
            df = df.set_index('Time')
            list_of_dfs.append(df)

        # Plot strangeness and deviation level over time
        if(plot == 'Y'):
            model.plot_deviations(figsize=(8, 6), plots=["strangeness", "deviation", "pvalue", "threshold"])

        return list_of_dfs
