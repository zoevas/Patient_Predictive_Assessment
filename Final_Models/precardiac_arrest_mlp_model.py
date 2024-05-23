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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier



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
niguarda_pre_hospital=pd.read_csv('../../niguarda_pre_ts_NOBMI.csv')
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

x = features.to_numpy()
y = targets[name].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.125, random_state=42)
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

scaler_minmax=MinMaxScaler()
x_train = scaler_minmax.fit_transform(x_train)
x_test = scaler_minmax.fit_transform(x_test)

print('dimensions')
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)


# Define MLPClassifier
mlp_classifier = MLPClassifier(
    random_state=1,
    max_iter=2000,
    activation='tanh',
    batch_size=32,
    solver='adam',
    hidden_layer_sizes=(50, ),
    learning_rate='invscaling',
    alpha = 0.0001,
    early_stopping=True
 )

x_resampled, y_resampled = undersampler.fit_resample(x_train, y_train)

gbpClassifier=GradientBoostingClassifier(n_estimators=50,
    learning_rate=0.001,
    max_depth=3,
    random_state=1)
# gbp does not work it gives 0 in class 1
# clf = gbpClassifier
clf = mlp_classifier

clf.fit(x_resampled, y_resampled)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)

# Print the classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)

# Plot the confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# Confusion Matrix
cm = confusion_matrix(y_test, np.round(y_pred), normalize='true')
df_cf = pd.DataFrame(np.round(cm, 2), index=['No', 'Yes'], columns=['No', 'Yes'])
sns.heatmap(df_cf, annot=True, cmap="viridis", ax=ax[0])
ax[0].set_title('Confusion Matrix')
ax[0].set_xlabel('Predicted Label')
ax[0].set_ylabel('True label')

# Classification Report Heatmap
class_rep_heatmap = sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T, annot=True, cmap="viridis", ax=ax[1])
ax[1].set_title('Classification Report Heatmap')
plt.show()

filename = "models/precardiac_arrest_model.pickle"
pickle.dump(clf, open(filename, "wb"))

explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(x_resampled, 100))

shap_values = explainer.shap_values(x_test)

pickle.dump(explainer, open('shap_models/precardiac_arrest_model.pkl', "wb"))







