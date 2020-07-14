# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:09:22 2020

@author: IPTLP0018
"""
from IntelliMaint.utils.Dimensions import Dimensions
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

class TimeDomain:
    '''
    TimeDomain : Extracts Time Domain Features RMS, Mean, Variance, Skewness, Kurtosis, Crest_Factor
       
    StatisticalFeatures.TimeDomain(df)
    '''
    def __init__(self, df):
        self.df = df
        
    def _func_compute(self, func):
        param = {}
        param['data'] = self.df
        dim = Dimensions(func, param)
        output = dim.choose_dim()
        return output

    def _rms_compute(self, data):
        rms = np.sqrt(np.mean(data**2, axis = 0))
        return rms
    
    def get_rms(self):
        """
        get_rms: 
            Perform Root Mean Square
            
        Returns:
            Dictionary
        
        """
        output = self._func_compute(self._rms_compute)
        return output
        

    def _mean_compute(self, data):
        rms = np.mean(data, axis = 0)
        return rms
    
    def get_mean(self):
        """
        get_mean: 
            Perform Mean across column
            
        Returns:
            Dictionary
        
        """

        output = self._func_compute(self._mean_compute)
        return output


    def _variance_compute(self, data):
        rms = np.var(data, axis = 0)
        return rms
    
    def get_variance(self):
        """
        get_variance: 
            Perform Variance across column
            
        Returns:
            Dictionary
        
        """

        output = self._func_compute(self._variance_compute)
        return output


    def _crestfactor_compute(self, data):
        peaks = np.max(data, axis=0)
        rms = self._rms_compute(data)
        try:
            output = np.divide(peaks, rms)
        except ZeroDivisionError:
            print('You Can\'t divide by Zero')
        return output
    
    def get_crestfactor(self):
        """
        get_crestfactor: 
            Perform CrestFactor analysis across column
            
        Returns:
            Dictionary
        
        """

        output = self._func_compute(self._crestfactor_compute)
        return output

    def _kurtosis_compute(self,data):
        output = kurtosis(data, axis=0)
        return output
    
    def get_kurtosis(self):
        
        """
        get_kurtosis: 
            Perform Kurtosis Analysis across column
            
        Returns:
            Dictionary
        
        """

        output = self._func_compute(self._kurtosis_compute)
        return output
        
    
    def _skewness_compute(self,data):
        output = skew(data, axis=0)
        return output
    
    def get_skewness(self):
        """
        get_skewness: 
            Perform Skewness analysis across column
            
        Returns:
            Dictionary
        
        """

        output = self._func_compute(self._skewness_compute)
        return output
        

    def extract_features(self, method_name='all'):
        """
        extract_features: 
            Perform Statistical Feature calculation across column
            
        Input Parameters:
        ----------------
            method_name(Char) = 'all':(default) Includes calculation of 'RMS', 'Variance', 'Crest_Factor', 'Kurtosis','Skewness'
                                'RMS': Calculates only RMS
                                'Variance': Calculates only Variance
                                'Crest_Factor': Calculates only Crest Factor
                                'Kurtosis': Calculates only Kurtosis
                                'Skewness': Calculates only Skewness
        Returns:
            Dictionary
        
        """

        if(method_name=='all'):
            method_name = ['RMS', 'Variance', 'Crest_Factor', 'Kurtosis','Skewness']
            
        output = dict()
        for method in method_name:
            if(method == 'RMS'):
                out = self.get_rms()
                output.update({'RMS':out})
            if(method == 'Variance'):
                out = self.get_variance()
                output.update({'Variance':out})
            if(method == 'Crest_Factor'):
                out = self.get_crestfactor()
                output.update({'Crest_Factor':out})
            if(method == 'Kurtosis'):
                out = self.get_kurtosis()
                output.update({'Kurtosis':out})
            if(method == 'Skewness'):
                out = self.get_skewness()
                output.update({'Skewness': out})
        return output
