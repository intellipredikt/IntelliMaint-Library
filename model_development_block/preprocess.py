# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 00:23:16 2020

@author: Admin
"""
#%%
import numpy as np
from IntelliMaint.model_development_block.utils import DL_utils
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class data_preprocessing:
    def __init__(self,data,scale='StandardScaler',target_names=None,input_indices=None,
                     train_val_test_ratio=[0.6,0.2,0.2],random_shuffle=False):
        self.target_names=target_names
        self.input_indices=input_indices
        self.train_val_test_ratio=train_val_test_ratio
        self.scale=scale
        self.random_shuffle=random_shuffle
        self.data=data
    
    def sequential_preprocess(self,lookback=2,forward_steps=0):
        convert=DL_utils() # 3D matrix creation
        data=self.data   
        if self.target_names==None:
            print('Error: Enter target variable name')
            sys.exit()
        if self.input_indices==None: # When column names are not provided
            col=data.columns.drop(self.target_names)
            if len(col)==0:
                col=self.target_names
        else:
            col=self.input_indices
        data_y=data[self.target_names].values
        data_y=np.reshape(data_y,(len(data_y),-1))
        n_features=np.size(data[col],1)       
        X,Y=convert.convert3D(data[col].values,data_y,lookback,forward_steps)                
        if self.random_shuffle==True:
            np.random.seed(32)
            mapIndexPosition = list(zip(X, Y))
            np.random.shuffle(mapIndexPosition)
            X,Y = zip(*mapIndexPosition)
            X,Y=list(X),list(Y)       
        X,y=np.array(X),np.array(Y)
        data_x=X.reshape(X.shape[0],lookback, n_features)
        if np.size(y,1)==1:
            y=np.reshape(y,(np.size(y,0),np.size(y,2)))
        # Spliting input data into train, val and test data
        tr_split=np.int(self.train_val_test_ratio[0]*len(data_x))
        x_t=data_x[0:tr_split]
        y_train=y[:tr_split] 
        
        val_split=np.int(self.train_val_test_ratio[1]*len(data_x))+tr_split
        x_v=data_x[tr_split:val_split]
        y_val=y[tr_split:val_split]
        
        x_te=data_x[val_split:]
        y_test=y[val_split:]
    
        # Scaling train val and test with respect to train data      
        input_scaler=eval(self.scale)().fit(convert.flatten(x_t)) #scaling training data        
        x_train=convert.scaling(x_t,input_scaler)
        x_val=convert.scaling(x_v,input_scaler)
        x_test=convert.scaling(x_te,input_scaler)
        target_scaler=[]
        if len(np.shape(y_train))==2:
            target_scaler.append(eval(self.scale)().fit(y_train))
        else:
            for i in range(0,np.size(y_train,2)):
                target_scaler.append(eval(self.scale)().fit(y_train[:,:,i]))
        
        def target_scaling(da,target_scaler):
            t_scaled=[]
            if len(np.shape(da))==2:
                t_scaled.append(target_scaler[0].transform(da))
            else:
                for i in range(0,np.size(target_scaler,0)):
                    t_scaled.append(target_scaler[i].transform(da[:,:,i]))
            return t_scaled
        y_train=target_scaling(y_train,target_scaler)       
        y_test=target_scaling(y_test,target_scaler)
        y_val=target_scaling(y_val,target_scaler)
        scaler={'input_scaler':input_scaler,'target_scaler':target_scaler,
                'input_column_names':col,'target_names':self.target_names,
                'lookback':lookback,'f_steps':forward_steps}
        processed_data={'x_train':x_train,'x_val':x_val,'x_test':x_test,
           'y_train':y_train,'y_val':y_val,'y_test':y_test}
        return processed_data,scaler
