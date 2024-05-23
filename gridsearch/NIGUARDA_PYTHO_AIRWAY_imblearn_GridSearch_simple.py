# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:13:35 2023

@author: stapolitis
"""

from sklearn import ensemble
from sklearn import metrics
import sklearn as skl
import pandas as pd
import numpy.random as rd
import scipy as sp
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import imblearn as imb
import numpy as np
import math
# alternative functions for undersampling
import sys
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.metrics import log_loss


def undersampling_primitive(df,percent,target,analogy):
    #split between intervantion and non-intervantion
   zero_df = df[df[target] < 0.5]
   one_df = df[df[target] > 0.5]
   #select randomnly according to one value-intervantion
   train_one=one_df.sample(frac=percent)
   test_one=one_df.drop(train_one.index)
   sa=train_one.inc_key.size
   train_zero=zero_df.sample(n=math.ceil(analogy*sa))
   test_zero=zero_df.drop(train_zero.index)
   #Combine balanced sets
   train = pd.concat([train_one,train_zero])
   test = pd.concat([test_one,test_zero])
   train = train.sample(frac=1).reset_index(drop=True)
   test = test.sample(frac=1).reset_index(drop=True)
   return train, test

def undersampling_primitive_2(df,percent,target,analogy):
    #split between intervantion and non-intervantion
   zero_df = df[df[target] < 0.5]
   one_df = df[df[target] > 0.5]
   #select randomnly according to one value-intervantion
   train_one=one_df.sample(frac=percent)
   test_one=one_df.drop(train_one.index)
   sa=train_one.inc_key.size
   zero_sa=zero_df.inc_key.size
   zero_sa_int=math.ceil((zero_sa)*(1-percent))
   train_zero=zero_df.sample(n=math.ceil(analogy*sa))
   test_zero=zero_df.drop(train_zero.index)
   test_zero=test_zero.sample(n=zero_sa_int)
   #Combine balanced sets
   train = pd.concat([train_one,train_zero])
   test = pd.concat([test_one,test_zero])
   train = train.sample(frac=1).reset_index(drop=True)
   test = test.sample(frac=1).reset_index(drop=True)
   return train, test

def nearmiss(df,percent,target,ver):
    from imblearn.under_sampling import NearMiss
    train=df.sample(frac=percent)
    test=df.drop(train.index)
    x_train=train[['SEX','AGEyears','EMSSBP','EMSPULSERATE','EMSRESPIRATORYRATE','EMSPULSEOXIMETRY','EMSTOTALGCS']].to_numpy()
    y_train=train[target].to_numpy()
    nr = NearMiss(version=ver) 
    X_near, Y_near= nr.fit_resample(x_train, y_train.ravel())
    Y_near=Y_near.reshape((Y_near.size,1))
    train_ar=np.hstack((X_near, Y_near))
    train=pd.DataFrame(train_ar,columns=['SEX','AGEyears','EMSSBP','EMSPULSERATE','EMSRESPIRATORYRATE','EMSPULSEOXIMETRY','EMSTOTALGCS',target])
    train = train.sample(frac=1).reset_index(drop=True)
    return train, test

#this is too slow, at least at my computer
def condensednene(df,percent,target):
    from imblearn.under_sampling import CondensedNearestNeighbour
    train=df.sample(frac=percent)
    test=df.drop(train.index)
    x_train=train[['SEX','AGEyears','EMSSBP','EMSPULSERATE','EMSRESPIRATORYRATE','EMSPULSEOXIMETRY','EMSTOTALGCS']].to_numpy()
    y_train=train[target].to_numpy()
    cnn = CondensedNearestNeighbour() 
    X_near, Y_near= cnn.fit_resample(x_train, y_train.ravel())
    Y_near=Y_near.reshape((Y_near.size,1))
    train_ar=np.hstack((X_near, Y_near))
    train=pd.DataFrame(train_ar,columns=['SEX','AGEyears','EMSSBP','EMSPULSERATE','EMSRESPIRATORYRATE','EMSPULSEOXIMETRY','EMSTOTALGCS','SUPPLEMENTALOXYGEN'])
    train = train.sample(frac=1).reset_index(drop=True)
    return train, test

def minmaxscale(x):
    xx=(x-x.min())/(x.max()-x.min())
    return(xx)
def standartscale(x):
    xx=(x-x.mean())/x.std()
    return(xx)
def robustscale(x):
    scaler=RobustScaler()
    x_scaler=scaler.fit_transform(x)
    xx=pd.DataFrame(x_scaler,columns=x.columns)
    return(xx)
#distribution is a string, 'normal' or 'uniform' n is an int suggestion 100, can be from 1 to 1000
def quantiletransformer(x,n,distribution):
    trans = QuantileTransformer(n_quantiles=n, output_distribution=distribution)
    data = trans.fit_transform(x)
    xx=pd.DataFrame(data,columns=x.columns)
    return(xx)
import warnings
warnings.simplefilter('ignore', FutureWarning)
#load data
niguarda_pre_hospital=pd.read_csv('../niguarda_pre_ts_NOBMI.csv')
#keep clean
clean=niguarda_pre_hospital.dropna().reset_index(drop=True)
#replace triage with numbers
mapping = {'green': 0, 'yellow': 1, 'red': 2}
clean.loc[:, 'Triage_code'] = clean['Triage_code'].replace(mapping)
#replace trauma type with numbers
mapping = {'blunt': 0, 'non noto': 1, 'penetrating': 2}
clean.loc[:, 'Trauma_type'] = clean['Trauma_type'].replace(mapping)
#replace male female with numbers
mapping = {'female': 0, 'male': 1}
clean.loc[:, 'Sex'] = clean['Sex'].replace(mapping)
#features_alone
features=clean[['Sex','Age','Trauma_type','Antplatelet','Oral_anticoagulant','Pregnancy','Heart_rate','Respiratory_rate','SpO2','Systolic_BP','Diastolic_BP','GCS_eyes','GCS_verbal','GCS_motor','GCS_total','RTS_Resp_Rate','RTS_Syst_BP','RTS_GCS','RTS','Triage_code']]


#targets
targets=clean[['Cristalloid','Colloid','Fluid_volume','Prehosp_cardiac_arrest','Airways_control','Cricotirodotomy','Chest_decompression','Intraosseous_access','Cardiac_massage','Cardiac_Arrest_ED','Massive_transfusion_protocol_activation','Right_thoracostomy_tube','Left_thoracostomy_tube']]
targets=targets[['Cristalloid','Colloid','Fluid_volume','Prehosp_cardiac_arrest','Airways_control','Chest_decompression','Cardiac_Arrest_ED','Massive_transfusion_protocol_activation','Right_thoracostomy_tube','Left_thoracostomy_tube']]
targets['chest_tube_needle'] = targets[['Chest_decompression', 'Right_thoracostomy_tube','Left_thoracostomy_tube']].max(axis=1).astype(int)
targets['arrest'] = targets[['Prehosp_cardiac_arrest', 'Cardiac_Arrest_ED']].max(axis=1).astype(int)
targets['fluids'] = (targets['Fluid_volume'] > 0).astype(int)
targets['airways']= (targets['Airways_control'] != 'not intubated').astype(int)
first_targets=targets[['fluids','arrest','airways','chest_tube_needle','Massive_transfusion_protocol_activation']]
print('choose target  between 1 fluids 2 arrest 3 airways 4 chest_tube_needle and 5 Massive_transfusion_protocol_activation')
target = input("choose: ")
if target=='1':
    name='fluids'
    clean_tr = pd.concat([features,first_targets['fluids']], axis=1)
elif target=='2':
    name='arrest'
    clean_tr = pd.concat([features,first_targets['arrest']], axis=1)
elif target=='3':
    name='airways'
    clean_tr = pd.concat([features,first_targets['airways']], axis=1)
elif target=='4':
    name='chest_tube_needle'
    clean_tr = pd.concat([features,first_targets['chest_tube_needle']], axis=1)
elif target=='5':
    name='Massive_transfusion_protocol_activation'
    clean_tr = pd.concat([features,first_targets['Massive_transfusion_protocol_activation']], axis=1)

clean_tr.insert(0,'inc_key',range(1,clean_tr.Sex.size+1))
x_train=clean_tr.drop(columns=[name,'inc_key'])
y_train=clean_tr[name].to_numpy()
x_test=clean_tr.drop(columns=[name,'inc_key'])
y_test=clean_tr[name].to_numpy()
x_train_norm=minmaxscale(x_train)
x_test_norm=minmaxscale(x_test)
x_train_norm=x_train_norm.to_numpy()
x_test_norm=x_test_norm.to_numpy()
learning_rate=0.1
random_state=100

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.8, 0.9, 1.0],
    'max_depth': [3, 4, 5],
    'max_features': [20, 25, 30],
}

clf = ensemble.GradientBoostingClassifier(loss='log_loss', random_state=random_state)

#clf.fit(x_train_norm, y_train)

print("GridSearchCV")

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_norm, y_train)
print("GridSearchCV after fit")

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

results = grid_search.cv_results_

# Print parameter values at each step
#for i in range(len(results['params'])):
 #   print(f"Step {i + 1}: {results['params'][i]} - Mean Test Score: {results['mean_test_score'][i]}")

# Create a DataFrame to store hyperparameters and reports
results_df = pd.DataFrame(columns=['Parameters', 'Report'])

# Print classification report for each parameter combination
for i, params in enumerate(results['params']):
    # Get predictions for each fold
    y_pred = cross_val_predict(grid_search.best_estimator_, x_train_norm, y_train, cv=5)

    # Compute classification report
    report = classification_report(y_train, y_pred)

    print(f"Parameters: {params}\n{report}")
    params_str = str(params)
    #results_df = results_df.append({'Parameters': params, 'Report': report}, ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame({
        'Parameters': params_str, 'Report': report})], index=True)

results_df.to_csv('csv/'+name+ '/'+ name +'_' + 'GradientBoost' + '_'+ 'hyperparameter_reports.csv', index=False)


