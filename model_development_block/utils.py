# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:32:23 2020

@author: Admin
"""

#%% Deep  learning utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle as pk
from keras.models import load_model

class DL_utils:
    def convert3D(self,xt,yt,lookback,f_steps=0):    # f_step is future steps  
        if len(yt)==0:
            x = []
            for i in np.arange(0,len(xt)-lookback+1-f_steps):
                t = []
                for j in range(0,lookback):
                    # Gather past records upto the lookback period
                    t.append(xt[[(i+j)], :])
                x.append(t)
            y=[]
        else:                   
            x,y = [],[]
            if f_steps>lookback:
                print('Warning: Multi step forcasting should be less than the lookback, else model will underfit the data.')
            
            for i in np.arange(0,len(xt)-lookback-f_steps):
                t = []
                for j in range(0,lookback):
                    # Gather past records upto the lookback period
                    t.append(xt[[(i+j)], :])
                x.append(t)
                if f_steps==0:
                    y.append(yt[i+lookback-1:i+lookback].reshape(f_steps+1,-1))
                else:
                    y.append(yt[i+lookback-1:i+lookback+f_steps].reshape(f_steps+1,-1))
        return x, y
    
    def flatten(self,X):
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return (flattened_X)
    
    def scaling(self,X, scaler): 
        Y=np.zeros(np.shape(X))
        for i in range(X.shape[0]):
            Y[i, :, :]= scaler.transform(X[i, :, :])   
        return Y  
    
    def training_plot(self,AE):
        training_loss = AE.history['loss']
        test_loss = AE.history['val_loss']
        plt.figure()
        plt.plot(training_loss, 'r--')
        plt.plot(test_loss, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation loss Vs Epoch') 
        
    def save_model(self,models,scaler,mu,sigma,path=None,model_name=None):
        print('-------------------------------Saving Model--------------------------------------')
        now = datetime.now()
        now= now.strftime("%d_%m_%Y_%H_%M_%S")
        if path==None:
            print('Error: Please enter path for saveing model. Example: ..Desktop\folder')
            path=os.getcwd()
        else:
            os.chdir(path)
        # Saving the model
        if model_name==None:
            m_name='model'
        else:
            m_name=model_name
        
        if len(scaler['target_scaler'])==1:
            model=models[0]
            model.save(m_name+'_'+now+'.h5')
        else:
            for i in range(0,len(models)):
                model=models[i]
                model.save(m_name+'_'+scaler['target_names'][i]+'_'+now+'.h5')       
        mu=pd.Series(mu)
        sigma=pd.Series(sigma)
        mu_sigma=pd.concat([mu.to_frame('mu'),sigma.to_frame('sigma')],axis=1)
        pk.dump([scaler,mu_sigma], open(m_name+'_'+'scaler'+'_'+now+'.pkl', 'wb'))
        print('Saved model,scaler,mu and sigma in:',path)
        
    def model_loading(self,path=None,model_name=None):
        print('--------------------------Loading Model--------------------------------')    
        if path==None:
            path=os.getcwd()
            print('Searching model in default path or current directory:',path)
        else:
            os.chdir(path)
            print('Searching model in:',path)
        try:
            m_name=model_name
            model= load_model(m_name)
            model._make_predict_function()            
            s_name=m_name.split('/')[-1].split('_')[0]+'_scaler_'+m_name.split('/')[-1].split('_',2)[2].split('.')[0]  
            scaler,mu_sigma = pk.load(open(s_name+'.pkl', 'rb'))
            mu=list(mu_sigma['mu'])
            sigma=list(mu_sigma['sigma'])
            print('Model, scaler, mu and sigma loaded succesfully')
        except:
            print('Error: Entered model name is not available in defined path, please try again with correct model name. Example: model_03_05_2020.h5')
            model,scaler,mu,sigma=[],[],[],[]
        return model,scaler,mu,sigma

