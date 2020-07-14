# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:08:52 2020

@author: Admin
"""


#%%-----------------Section1: Importing fuction -------------------------------#
import numpy as np
import os as os
import matplotlib.pyplot as plt
import warnings
from keras.layers import Dense,LSTM, Dropout
from keras.models import Sequential
import mplcursors
import seaborn as sns
import sys
from IntelliMaint.model_development_block.preprocess import data_preprocessing as dp
from IntelliMaint.model_development_block.utils import DL_utils
sns.set(style="whitegrid")
warnings.filterwarnings("ignore")
plt.close('all')
np.random.seed(203)

#%%-----------------Section2: Deep Learning(LSTM) -------------------------------#
class LSTM_NN:
    def __init__(self,data):
        self.data=data
    def create_model(self,layers=[100,80,30],activation_func=['linear','relu','linear'],dropout=[0.2,0.2,0.2]):
        data=self.data
        if 'x_train' in data:
            total_target=len(data['y_train'])
            len_target=np.size(data['y_train'][0],1)
            lookback=np.size(data['x_train'],1)
            n_features=np.size(data['x_train'],2)           
        else:
            print('Error: Please provide scaled input data with train, validation and test data')
            sys.exit()
            
        a_func=activation_func
        i_layer=layers
        drop=dropout
        # if only one Layers    
        if np.size(i_layer,0)!=np.size(a_func,0):
            print('Error: Number of activation function should be equal to number of layers')
            sys.exit()
        if np.size(i_layer,0)!=np.size(drop,0):
            print('Warning: Number of dropout should be equal to number of layers, appending with zeros')    
            for i in range(0,np.size(i_layer,0)-np.size(drop,0)):
                drop.append(0)
        if np.size(i_layer,0)<=2:
            print('Error: Please add minimum three layers with activation fuction for model development')        
            sys.exit()
        else:
            model = Sequential()
            # First layer adding
            model.add(LSTM(i_layer[0], activation=a_func[0], input_shape=(lookback, n_features),
                                return_sequences=True))
            model.add(Dropout(drop[0]))
            # Remainig layer added but not last two layer
            for i in np.arange(1,np.size(i_layer,0)-2):
                model.add(LSTM(i_layer[i], activation=a_func[i],return_sequences=True))
                model.add(Dropout(drop[i]))
            # Second last Layer adding
            model.add(LSTM(i_layer[-2], activation=a_func[-2],return_sequences=False))
            model.add(Dropout(drop[-2]))
            # Last layer adding
            if i_layer[-1]!=len_target:
                print('Warning: Number of neurons in last layers should be equal to number of target i.e,',
                      len_target)
            model.add(Dense(len_target,activation=a_func[-1]))
        model.summary()
        return model
    
    def train(self,model,scaler,optimizer='adam',loss='mse',metrics=['mse'],
              epoch=50,batch_size=50): 
        data=self.data
        if 'x_train' in data:
            total_target=len(scaler['target_names'])      
        else:
            print('Error: Please provide scaled input data with train, validation and test data')
            sys.exit()        
        
        mu,sigma,RE_train,models,model_history=[],[],[],[],[]
        for i in range(0,len(data['y_train'])):
            print('---------------------------------------------------------------------')
            if len(scaler['target_scaler'])==1:
                print('Training model for: ',scaler['target_names'])
            else:            
                print('Training model for: ',scaler['target_names'][i])
            print('---------------------------------------------------------------------')
            model.compile(optimizer = optimizer, loss = loss,metrics=metrics)
            model_history.append(model.fit(data['x_train'], data['y_train'][i], epochs = epoch, batch_size = batch_size,
                            shuffle = False,validation_data = (data['x_val'], data['y_val'][i])))
            training=DL_utils()
            training.training_plot(model_history[i])
            print('Training completed and loss vs epoch graph is plotted')
            pr=model.predict(data['x_train'])       
            pred=scaler['target_scaler'][i].inverse_transform(pr)
            actual=scaler['target_scaler'][i].inverse_transform(data['y_train'])[i]
            RE_train.append(np.abs(actual-pred))
            mu.append(np.mean(RE_train[i],0))
            sigma.append(np.std(RE_train[i],0))
            score=model.evaluate(data['x_test'],data['y_test'][i])
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            models.append(model)
        return mu,sigma,RE_train,models,model_history
    
    def prediction(self,models,scaler,RE_plot=True,plot_title=None,plot_column_num=0):
        f_steps=scaler['f_steps']
        target_names=scaler['target_names']
        col=scaler['input_column_names']
        lookback=scaler['lookback']
        convert=DL_utils()
        data=self.data
        n_features=np.size(data[col],1)
        
        def forcasting(da):
            a=np.transpose(da)
            c=[]
            for i in range(0,np.size(a,0)):
                if i==0:
                    c.append(a[i,i])
                else:
                    f=list(range(0,i+1))
                    k=f.copy()
                    f.reverse()
                    p=[]
                    for j in range(0,len(f)):
                        p.append(a[k[j],f[j]])
                    c.append(np.sum(p)/np.size(f,0))  
            for j in range(np.size(a,0),np.size(a,1)):
                k=list(range(0,np.size(a,0)))
                l=list(np.linspace(j-np.size(a,0)+1,j,np.size(a,0)))
                l.reverse()
                p=[]
                for j in range(0,len(k)):
                    p.append(a[k[j],int(l[j])])
                c.append(np.sum(p)/np.size(a,0))
            return np.array(c).reshape(len(c),-1)
                             
        if len([i for i in list(data.columns) if i in target_names])==0:   
            yt=[]
            X,_=convert.convert3D(data[col].values,yt,lookback,f_steps=0)        
            X=np.array(X)
            data_x=X.reshape(X.shape[0], lookback, n_features)
            if n_features==len(scaler['input_scaler'].mean_):
                data_x=convert.scaling(data_x,scaler['input_scaler'])
                
                pre=models[0].predict(data_x)   
                pr=scaler['target_scaler'][0].inverse_transform(pre) # scaling back to original target     
                RE=[]
            else:
                print('Invalid dataframe. Number of features are not matching with trained model')
                pr,RE=[],[]
            RE_plot=False
        else:
            target=target_names
            data_y=data[target_names].values
            data_y=np.reshape(data_y,(len(data_y),-1))
            y_data=data_y.copy()
            X,Y=convert.convert3D(data[col].values,data_y,lookback,f_steps=f_steps)        
            X,data_y=np.array(X),np.array(Y)
            data_x=X.reshape(X.shape[0], lookback, n_features)
            data_x=convert.scaling(data_x,scaler['input_scaler'])
            RE,pr=[],[]
            for i in range(0,len(scaler['target_scaler'])):
                model=models[i]
                pre=model.predict(data_x)
                print(np.shape(data_y))
                print(np.shape(pre))
                pr.append(scaler['target_scaler'][i].inverse_transform(pre)) # scaling back to original target
                if scaler['f_steps']==0 and len(scaler['target_names'])>1:
                    act=np.reshape(data_y,(len(pr[i]),np.size(data_y,2)))
                else:
                    act=np.reshape(data_y[:,:,i],(len(pr[i]),-1))  
                RE.append(np.abs(act-pr[i]))   
                score=model.evaluate(data_x,scaler['target_scaler'][i].transform(act))
                print('Prediction loss:', score[0])
                print('Prediction accuracy:', score[1])
                
        if RE_plot==True:
            plt.figure()
            if len(target)==1:
                ax1=plt.subplot(2,1,1)
                plt.plot(y_data)
                plt.plot(forcasting(pr[0]))
                plt.ylabel(str(target),fontsize=12)
                plt.legend(['Data','Reconstructed'])
                if plot_title==None:
                    title='Model prediction'
                else:
                    title=plot_title           
                plt.title(title,fontsize=16,fontweight='bold')
                plt.xticks(rotation=0, ha='right')
                plt.subplot(2,1,2,sharex=ax1)
                plt.plot(forcasting(RE[0]),'k')
                plt.ylabel(str(target),fontsize=12)
                plt.title('Reconstruction error',fontsize=16,fontweight='bold')
                plt.xticks(rotation=0, ha='right')
                mplcursors.cursor()
                plt.tight_layout()
            else:
                ax1=plt.subplot(2,1,1)
                if plot_column_num==None:
                    c_n=0
                    print('Warning: Plotting default target:' ,target[c_n])
                else:
                    c_n=plot_column_num     
                if c_n<len(target):
                    print('--------------------Plotting reconstruction error----------------------')
                else:
                    print('--------------------Plotting reconstruction error----------------------')
                    print('Entered column is invalid, plotting default column number 0')
                    c_n=0
                plt.plot(y_data[:,c_n])
                if scaler['f_steps']==0 and len(scaler['target_names'])>1:
                    plt.plot(pr[0][:,c_n])
                else:
                    plt.plot(forcasting(pr[c_n]))
                plt.ylabel(target[c_n],fontsize=12)
                plt.legend(['Data','Reconstructed'])
                if plot_title==None:
                    title='Model prediction'
                else:
                    title=plot_title
                plt.title(title,fontsize=16,fontweight='bold')
                plt.xticks(rotation=0, ha='right')
                plt.subplot(2,1,2,sharex=ax1)
                
                if scaler['f_steps']==0 and len(scaler['target_names'])>1:
                    print(np.shape(RE))
                    plt.plot(RE[0][:,c_n],'k')
                else:
                    plt.plot(forcasting(RE[c_n]),'k')
                plt.ylabel(target[c_n],fontsize=12)
                plt.title('Reconstruction error',fontsize=16,fontweight='bold')
                plt.xticks(rotation=0, ha='right')
                mplcursors.cursor()
                plt.tight_layout()
        return RE,pr
    
    def save_model(self,models,scaler,mu,sigma,save_path=None,model_name=None):
        s_m=DL_utils()
        s_m.save_model(models,scaler,mu,sigma,save_path,model_name)
        
    def model_loading(self,load_path=None,model_name=None):
        l_m= DL_utils()
        models,scaler,mu,sigma=l_m.model_loading(load_path,model_name)
        return models,scaler,mu,sigma