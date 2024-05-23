# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:13:35 2023

@author:
Undersampling within crossvalidation will be performed as crossvalidation keeps the initial analogy
in the folds.
If no undersampling is applied in the folds, then the precision is 0 increases, this is not correct
I think
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
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sn
from matplotlib import pyplot as plt


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

# Import Required Modules
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn import datasets
from sklearn import metrics
# FEATCHING FEATURES AND TARGET VARIABLES IN ARRAY FORMAT.

# Input_x_Features.
x = features.to_numpy()
    
# Input_ y_Target_Variable.
y =clean_tr.iloc[:,20].to_numpy()
      
     
# Feature Scaling for input features.
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)

undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
     
   # Create  classifier object.
random_state=100
ss=0.01
clf =ensemble.GradientBoostingClassifier(max_features = 'sqrt',learning_rate = ss,\
                                  loss = 'log_loss',min_samples_split=900,min_samples_leaf=50,n_estimators = 160,subsample = 0.9,max_depth=4,
                                   random_state = random_state)  
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
lst_accu_stratified = []
lst_roc_stratified = []
lst_re0_stratified = []
lst_re1_stratified = []
lst_pr0_stratified= []
lst_pr1_stratified = []
lst_f1_stratified=[]

y_true = []
predictions = []
for train_index, test_index in skf.split(x, y):
       x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
       y_train_fold, y_test_fold = y[train_index], y[test_index]
      # x_resampled, y_resampled = undersampler.fit_resample(x_train_fold, y_train_fold)
       x_resampled, y_resampled = x_train_fold , y_train_fold
       clf.fit(x_resampled, y_resampled)
       y_pred = clf.predict(x_test_fold)
       lst_accu_stratified.append(clf.score(x_test_fold, y_test_fold))
       lst_re0_stratified.append( metrics.recall_score(y_test_fold, y_pred, pos_label=0))
       lst_re1_stratified.append(metrics.recall_score(y_test_fold, y_pred, pos_label=1))
       lst_pr0_stratified.append(metrics.precision_score(y_test_fold, y_pred, pos_label=0))
       lst_pr1_stratified.append(metrics.precision_score(y_test_fold, y_pred, pos_label=1))
       lst_f1_stratified.append(metrics.f1_score(y_test_fold, y_pred, average='macro'))
       y_pred_proba = clf.predict_proba(x_test_fold)[::,1]
       lst_roc_stratified.append(metrics.roc_auc_score(y_test_fold, y_pred_proba))

       y_true.append(list(y_test_fold))
       predictions.append(list(y_pred))

y_true_combined = np.concatenate(y_true)
predictions_combined = np.concatenate(predictions)

fig, ax = plt.subplots(1,1, figsize=(16,7))
cm = metrics.confusion_matrix(y_true_combined, np.round(predictions_combined), normalize='true')
df_cf=pd.DataFrame(np.round(cm, 2), index=['No', 'Yes'], columns=['No', 'Yes'])
sn.heatmap(df_cf, annot=True, cmap="viridis", ax=ax)
ax.set_title('Airway Undersampling')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True label')
plt.show()

# Print the output.
print('for the target '+name+' :')
print('List of possible accuracy:', lst_accu_stratified)
print('\nOverall Accuracy:',
       mean(lst_accu_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))
print('List of possible roc accuracy:', lst_roc_stratified)
print('\nOverall roc Accuracy:',
        mean(lst_roc_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_roc_stratified))
print('List of possible f1 macro:', lst_f1_stratified)
print('\nOverall f1 macro:',
         mean(lst_f1_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_f1_stratified))
print('List of possible recall for 0:', lst_re0_stratified)
print('\nOverall recall for 0:',
         mean(lst_re0_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_re0_stratified))
print('List of possible recall for 1:', lst_re1_stratified)
print('\nOverall recall for 1:',
         mean(lst_re1_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_re1_stratified))
print('List of possible precission for 1:', lst_pr1_stratified)
print('\nOverall precission for 1:',
         mean(lst_pr1_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_pr1_stratified))
