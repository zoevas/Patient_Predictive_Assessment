# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:13:35 2023

@author: stapolitis
"""

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_predict


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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler_minmax=MinMaxScaler()
x_train = scaler_minmax.fit_transform(x_train)
x_test = scaler_minmax.fit_transform(x_test)

print('dimensions')
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)

learning_rate=0.1
random_state=100

gb_classifier = GradientBoostingClassifier(learning_rate=learning_rate, random_state=random_state)
mlp_classifier = MLPClassifier(random_state=1, max_iter=300)
samplers = [
    #SMOTEENN(),
    #SMOTETomek(),
    RandomUnderSampler()
]

classifiers = [
                ('GradientBoosting', gb_classifier),
                ('MLP', mlp_classifier)
]

pipelines = []
param_grids = []
for classifier_name, classifier in classifiers:
    for sampler in samplers:
        pipeline = ImbPipeline([('sampler', sampler), ('classifier', classifier)])
        pipelines.append((classifier_name,pipeline))
        param_grid = {}
        if classifier_name == 'GradientBoosting':
             param_grid = {
             f'classifier__{key}': value
             for key, value in {
               'n_estimators': [50, 100, 200],
               'learning_rate': [0.01, 0.1, 0.2],
               'max_depth': [3, 5, 7]
             #  'subsample': [0.8, 1.0],
              # 'min_samples_split':[2, 5, 10],
            #   'min_samples_leaf':[1, 2, 4]
              }.items()
        }
        elif classifier_name == 'MLP':
            param_grid = {
                f'classifier__{key}': value
                for key, value in {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate':['constant', 'invscaling', 'adaptive'],
                    'solver': ['adam', 'sgd'],
                    'batch_size': [32, 64, 128],
                    'early_stopping': [True, False],
                    'max_iter': [2000]
                    # Add other MLP hyperparameters
                }.items()
            }

        param_grids.append((classifier_name, param_grid))

print('type of y:', type(y_train))
missing_values = pd.isna(y).sum()
print('Number of missing values in y:', missing_values)
unique_values = np.unique(y)
print('Unique values in y:', unique_values)

print("GridSearchCV")

results_df= pd.DataFrame(columns=['Model', 'Sampler', 'Best_Params', 'Classification_Report'])

for (model, pipeline),(_, param_grid) in zip(pipelines, param_grids):
    print('model:', model, 'pipeline: ', pipeline, 'param_grid', param_grid)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(x_train, y_train)

    print("GridSearchCV after fit", pipeline)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(f'Best Hyperparameters for {model}: {best_params}')

    best_model = grid_search.best_estimator_

    y_pred = grid_search.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

    results = grid_search.cv_results_

    best_params_str = str(best_params)
   # Append the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame({
        'Model': model,
        'Sampler': type(pipeline.named_steps['sampler']).__name__ if pipeline.named_steps['sampler'] else 'NoSampler',
        'Best_Params': best_params_str,
        'Classification_Report': report
    })], ignore_index=True)

    # Create a list to store intermediate results DataFrames
    intermediate_results_dfs = []
    # Create a DataFrame to store hyperparameters and reports
    intermediate_results_df = pd.DataFrame(columns=['Model', 'Sampler', 'Parameters', 'Report'])
    for i, params in enumerate(results['params']):
        # Get predictions for each fold
        y_pred = cross_val_predict(grid_search.best_estimator_, x_train, y_train, cv=5)

        # Compute classification report
        report = classification_report(y_train, y_pred)

        print(f"Parameters: {params}\n{report}")
        params_str = str(params)
        #results_df = results_df.append({'Parameters': params, 'Report': report}, ignore_index=True)
        i
        # Create a DataFrame to store intermediate results
        intermediate_results_df_temp = pd.DataFrame({
        'Model': model,
        'Sampler': type(pipeline.named_steps['sampler']).__name__ if pipeline.named_steps['sampler'] else 'NoSampler',
        'Parameters': [params_str],
        'Report': [report],
        })

        # Save intermediate results DataFrame to the list
        intermediate_results_dfs.append(intermediate_results_df_temp)

    # Concatenate all intermediate results DataFrames
    intermediate_results_df = pd.concat(intermediate_results_dfs, ignore_index=True)

    intermediate_results_df.to_csv('csv/'+name+ '/'+ name + '_' + model + '_' + type(pipeline.named_steps['sampler']).__name__ +"_" +'hyperparameter_reports.csv', index=False)

# Save the DataFrame to a CSV file
results_df.to_csv('csv/'+ name + '/' + 'gridcsv_overallResults.csv', index=False)

#
def model_predict_function(x):
    return best_model.predict_proba(x)[:, 1]

explainer = shap.KernelExplainer(model_predict_function, shap.sample(x_train, 10))
pickle.dump(explainer, open('shap_explainer.pkl', "wb"))

shap_values = explainer.shap_values(x_test)
print('shap values: ', shap_values)
print('shap values size: ', shap_values.size)
#shap.plots.bar(shap_values)
feature_name=['Sex', 'Age', 'Trauma_type', 'Antplatelet', 'Oral_anticoagulant',
       'Pregnancy', 'Heart_rate', 'Respiratory_rate', 'SpO2', 'Systolic_BP',
       'Diastolic_BP', 'GCS_eyes', 'GCS_verbal', 'GCS_motor', 'GCS_total',
       'RTS_Resp_Rate', 'RTS_Syst_BP', 'RTS_GCS', 'RTS', 'triage_code']
print('feature_name size: ', len(feature_name))
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, x_test, feature_names=feature_name,plot_type='bar' ,show=False)
plt.title('SHAP Summary Plot')
plt.show()




