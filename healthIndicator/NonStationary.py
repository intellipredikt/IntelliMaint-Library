# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:10:49 2020

@author: IPTLP0018
"""
from IntelliMaint.utils.Dimensions import Dimensions
import numpy as np

class Nonstationary:
    '''
    Nonstationary: Extracts features (Empherical Mode Decomposition[EMD], Wavelet packet decomposition[WPD] for Non Stationary Time Series Data
    Nonstationary.extract_features(df, fs)
    '''

    def __init__(self, df):
        self.df = df
    
    def _func_compute(self, func):
        param = {}
        param['data'] = self.df
        dim = Dimensions(func, param)
        output = dim.choose_dim()
        return output
    
    def _emd(self, data):
        emd = EMD()
        IMFs = emd(data)
        return IMFs
    
    def get_emd(self):
        """
        get_emd: 
            Perform Empherical Mode Decomposition.
            
        Input Paramenters:
        -----------------
        data(n-dim dataframe)
                            
        Returns:
       -------
       dict : Dictionary 
        """

        output = self._func_compute(self._emd)
        return output
    
    '''
    def _wpd(self):
        emd = EMD()
        IMFs = emd(data)
        return IMFs
    
    def get_emd(self):
        output = self._func_compute(self._emd)
        return output
    #Emphirical Mode Decomposition
    '''
    def get_wpd(self, wavelet_name ='db4', mode='symmetric', level=2):
        """
        get_wpd: 
            Perform Wavelet Packet Decomposition.
            
        Input Paramenters:
        -----------------
        wavelet_name, mode:
             
           wavelet_name(char)(Optional): 
           mode(char)(Optional):
               
        Returns:
       -------
       dict : Dictionary 
        """

        wd = pywt.WaveletPacket(self.df,wavelet=wavelet_name, mode=mode)
        wd.get_level(level)

        path = []
        ds = []
        for n in wd.get_leaf_nodes(False):
            path.append(n.path)
            ds.append(n.data)

        path = np.asarray(path)
        ds = np.asarray(ds)
        return path, ds

    def extract_features(self, method_name = ['WPD','EMD'], wavelet_name='db4', mode='symmetric', level=4):
        """
        extract_features: 
            Extracts  Features like WPD and EMD.
            
        Input Paramenters:
        -----------------
        method_name, wavelet_name, mode:
             
            method_name*(char): 'WPD' - Returns Wavelet Packet Decomposition features
                               'EMD' - Returns Empherical Mode Decomposition features
                               ['WPD','EMD'] - (default) Returns both features
           wavelet_name(char)(Optional): 
           mode(char)(Optional):
               
        Returns:
       -------
       dict : Dictionary of selected features
        """

        output = dict()
        for method in method_name:
        #print(method)
            if(method == 'WPD'):
                out = self.get_wpd(self.df,wavelet_name,mode,level)
                output.update({'WPD':out})
            if(method == 'EMD'):
                out = self.get_emd(self.df)
                output.update({'EMD': out})
        return output
