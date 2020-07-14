# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:10:01 2020

@author: IPTLP0018
"""
from IntelliMaint.utils.Dimensions import Dimensions
import numpy as np
class FrequencyDomain:
    ''' 
    Frequency domain: Extracts Frequency Domiain features Spectrum and Cepstrum 
        
        FrequencyDomain.extractfeatures(df,fs,method_name)
        
    '''

    def __init__(self, df):
        self.df = df

    def _nextpow2(self, data):
        return np.ceil(np.log2(len(data)))

    
    def _update_parameters(self, fs, nfft, window):
        param = {}
        if(nfft==None):
            nfft = int(2**self._nextpow2(data)*2)        
        
        if(window == 'hanning'):
            window=np.hanning(nfft)
        elif(window == 'hamming'):
            window=np.hamming(nfft)
        elif(window == 'blackman'):
            window=np.blackman(nfft)
        elif(window == 'bartlett'):
            window == np.bartlett(nfft)
        else:
            print('This window type is not available')
        freqs = np.linspace(0,int(fs/2),num=int(np.floor(nfft/2))+1)
        
        lags = np.linspace(0,(nfft/2)-0.5, int(nfft/2))/fs
        
        param['data'] = self.df
        param['fs'] = fs
        param['nfft'] = nfft
        param['window_coeff'] = window
        return param, freqs, lags
    
    def get_spectrum(self, fs=16000, nfft=None, window = 'hanning'):
        """
        get_spectrum: 
            Estimates Power Spectral Desities using Fourier Trasformation
            
        Input Parameters:
        -----------------
        data,fs,nfft,window:
            data(n-dim numpy array): Input data could be of multiple dimensions of numpy array
            fs(int): Frame Shift, default = 16000
            nfft(): Nonuniform Fast Fourier Trasform
            window(char): Standard tapering window (hann, hamming, blackman, bartlett), default = hanning
            
        Returns:
        --------
        dictionary: Consists Spectrum Features
            
        """
        param, freqs, _ = self._update_parameters(fs, nfft, window)
        dim = Dimensions(self._spectrum, param)
        output = dim.choose_dim()
        return output, freqs

    def get_cepstrum(self, fs=16000, nfft=None, window='hanning'):
        """
        get_cepstrum: 
            Log magnitude of the Spectrum followed by an inverse Fourier transform
            
        Input Parameters:
        -----------------
        data,fs,nfft,window:
            data(n-dim numpy array): Input data could be of multiple dimensions of numpy array
            fs(int): Frame Shift, default = 16000
            nfft(): Nonuniform Fast Fourier Trasform
            window(char): Standard tapering window (hann, hamming, blackman, bartlett), default = hanning
            
        Returns:
        --------
        dictionary: Consists Cepstrum Features
            
        """

        param, _, lags = self._update_parameters(fs, nfft, window)
        dim = Dimensions(self._cepstrum, param)
        output = dim.choose_dim()
        return output, lags
    
    def _spectrum(self, data=[], fs=16000, nfft=None, window_coeff = []):
        zeroPadded = np.append(data,np.zeros(int(nfft)-len(data)))
        spectrum = (np.abs(np.fft.rfft(zeroPadded*window_coeff,nfft)))**2/nfft
        return spectrum
    
    def _cepstrum(self, data = [], fs=16000, nfft=None, window_coeff = []):
        zeroPadded = np.append(data,np.zeros(int(nfft)-len(data)))
        spectrum = (np.abs(np.fft.rfft(zeroPadded*window_coeff,nfft)))**2/nfft
        
        temp = np.fft.irfft(np.log(spectrum))
        cepstrum = temp[:int(len(temp)/2)]
        return cepstrum

    def extract_features(self, fs, method_name='all'):
        
        """
        extract_features: 
            Extracts Frequency Domain Features like spectrum and cepstrum.
            
        Input Paramenters:
        -----------------
        fs,method_name:
            fs(int): Frame Shift 
            method_name(char): 'all' - Returns both spectrum and ceptrum features
                               'spectrum' - Returns only spectral densities
                               'cepstrum' - Returns only cepstral coordinates
       Returns:
       -------
       dict : Dictionary of selected features
        """
        if(method_name == 'all'):
            method_name = ['spectrum', 'cepstrum']
        
        output = dict()
        for method in method_name:
            if(method == 'spectrum'):
                out = self.get_spectrum()
                output.update({'Spectrum':out})
            if(method == 'Cepstrum'):
                out = self.get_cepstrum()
                output.update({'Cepstrum':out})
        return output
