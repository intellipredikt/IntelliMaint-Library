# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:07:03 2020

@author: IPTLP0018
"""
import numpy as np
class Dimensions:
    
    def __init__(self, func, param):
        self.func = func
        self.param = param
        self.df = param['data']

    def choose_dim(self):
        if(self.df.ndim == 3):
            output = self._handle_3D()
        elif(self.df.ndim == 2):
            output = self._handle_2D()
        elif(self.df.ndim == 1):
            output = self._handle_1D()
        else:
            print('This dimension is not handled')
        return output
    
    
    def _handle_3D(self):
        output_val = []
        append_3D_output = []
        
        for chunks in self.df:
            append_2D_output = []
            for chunk in chunks.T:
                self.param['data'] = chunk
                func_output =  self.func(**self.param)
                append_2D_output.append(func_output)
            np_format_2D_output = np.array(append_2D_output)
            append_3D_output.append(np_format_2D_output.T)
            output_val = np.array(append_3D_output)
        return output_val
    
    def _handle_2D(self):
        output_val = []
        for chunk in self.df.T:
            self.param['data'] = chunk
            func_output = self.func(**self.param)
            output_val.append(func_output)
        output_val = np.array(output_val)
        return output_val
    
    def _handle_1D(self):
        output_val = []
        func_output, freq = self.func(**self.param)
        output_val.append(func_output)
        output_val = np.array(output_val)
        return output_val
