# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:06:25 2020

@author: IPTLP0018
"""
import numpy as np
import sys
class ChunkData:
    
    def __init__(self, data, frame_size=100, frame_shift = None):
        self.data = data
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        if(self.frame_shift == None):
            self.frame_shift = self.frame_size
    
    def _check1(self):
    # Check if the provided input data is atleast an 1D array
        if not isinstance(self.data, np.ndarray):
            data = np.array(self.data)
        if(self.data.ndim < 1):
            sys.exit('Oops !!! Error with the input vector')
        elif(self.data.ndim == 1):
            self.data = np.reshape(self.data,(self.data.shape[0],1))
        return self.data
    

    
    def make_chunks(self):
    # initialize the number of chunks defined.
        chunks = []
        self.data = self._check1()
    #Check if the provided size of the input data is greater than frame_size
        if(self.data.shape[0]<self.frame_size):
            sys.exit('Oops !!! The length of input vector is smaller than the analysis window length')
    
        for j in range(0,len(self.data)-self.frame_size,self.frame_shift):
            
            chunks.append(self.data[j:j+self.frame_size,:])
        frames = np.array(chunks)
        return frames



