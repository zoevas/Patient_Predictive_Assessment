# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:10:12 2024

@author: stapolitis
"""

from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
# alternative functions for undersampling
import shap
from statistics import mean, stdev


import warnings
warnings.simplefilter('ignore', FutureWarning)
#load data
niguarda_pre_hospital=pd.read_csv('niguarda_pre_ts_NOBMI.csv')
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
    
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import Normalizer, PowerTransformer, QuantileTransformer   
    
# Standard Scaler
scaler_standard = StandardScaler()

# Min-Max Scaler
scaler_minmax = MinMaxScaler()

# Robust Scaler
scaler_robust = RobustScaler()
scaler_n = Normalizer()
scaler_qt = QuantileTransformer(output_distribution='uniform')  # or 'normal'
    
sc_list=[scaler_minmax] 
sc_names=['minmax']   
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss, ClusterCentroids
from imblearn.over_sampling import SMOTE

imb_list = [RandomUnderSampler(random_state=42)]
sample=RandomUnderSampler(random_state=42)
imb_names=['RANDOM']
from sklearn.ensemble import GradientBoostingClassifier

for scaler, scaler_name in zip(sc_list,sc_names):
    for sampler, sampler_name in zip(imb_list,imb_names):
        print('scaling technique: ',scaler_name)
        print('sampling technique:', sampler_name)
        # Input_x_Features.
        x = features.to_numpy()       
         
        # Input_ y_Target_Variable.
        y =clean_tr.iloc[:,20].to_numpy()
        
        x_scaled = scaler.fit_transform(x)

        from sklearn.model_selection import StratifiedKFold

        # Define the number of folds for cross-validation
        n_splits = 10  # Choose the number of folds

        # Initialize StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize the classifier
        n_estimators= [ 100]
        learning_rate=[ 0.1]
        max_depth=[3]
        min_samples_split=[ 4]
        min_samples_leaf=[1]
        subsample=[ 0.9]

        for ne in n_estimators:
            for lr in learning_rate:
                for md in max_depth:
                    for ms in min_samples_split:
                        for ml in min_samples_leaf:
                            for sbs in subsample:
                                params = {
    'n_estimators': ne,
    'learning_rate': lr,
    'max_depth': md,
    'min_samples_split': ms,
    'min_samples_leaf': ml,
    'subsample': sbs,
    'random_state': 42
} 
                                clf= GradientBoostingClassifier(**params)
                                lst_accu_stratified = []
                                lst_roc_stratified = [] 
                                lst_re0_stratified = [] 
                                lst_re1_stratified = []
                                lst_pr0_stratified= []
                                lst_pr1_stratified = []
                                lst_f1_stratified=[]
                                # Iterate through each fold for cross-validation
                                for train_index, test_index in skf.split(x_scaled, y):
                                    # Split the data into train and test sets for this fold
                                    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
                                    y_train_fold, y_test_fold = y[train_index], y[test_index]

                                    # Apply undersampling to the training set within each fold
                                    x_train_resampled, y_train_resampled = sampler.fit_resample(x_train_fold, y_train_fold)

                                    # Train your classifier on the resampled data
                                    clf.fit(x_train_resampled, y_train_resampled)
                                    y_pred = clf.predict(x_test_fold)
                                    lst_accu_stratified.append(clf.score(x_test_fold, y_test_fold))
                                    lst_re0_stratified.append( metrics.recall_score(y_test_fold, y_pred, pos_label=0))
                                    lst_re1_stratified.append(metrics.recall_score(y_test_fold, y_pred, pos_label=1))
                                    lst_pr0_stratified.append(metrics.precision_score(y_test_fold, y_pred, pos_label=0))
                                    lst_pr1_stratified.append(metrics.precision_score(y_test_fold, y_pred, pos_label=1))
                                    lst_f1_stratified.append(metrics.f1_score(y_test_fold, y_pred, average='macro'))
                                    y_pred_proba = clf.predict_proba(x_test_fold)[::,1]
                                    lst_roc_stratified.append(metrics.roc_auc_score(y_test_fold, y_pred_proba))
                                
                                print(' ', scaler_name,' ',sampler_name,' ',ne,' ',lr,' ',md,' ',ms,' ',ml,' ',sbs)
                                print('\nOverall f1 macro:',
                                      mean(lst_f1_stratified)*100, '%')
                                print('\nOverall std/mean f1 macro:',
                                      100*stdev(lst_f1_stratified)/mean(lst_f1_stratified), '%')
                                print('\nOverall accuracy macro:',
                                      mean(lst_accu_stratified)*100, '%')
                                print('\nOverall std/mean accu:',
                                      100*stdev(lst_accu_stratified)/mean(lst_accu_stratified), '%')
                                print('\nOverall roc accuracy macro:',
                                      mean(lst_roc_stratified)*100, '%')
                                print('\nOverall std/mean roc:',
                                      100*stdev(lst_roc_stratified)/mean(lst_roc_stratified), '%')
                                print('\nOverall recall for 0:',
                                      mean(lst_re0_stratified)*100, '%')
                                print('\nOverall std/mean r0:',
                                      100*stdev(lst_re0_stratified)/mean(lst_re0_stratified), '%')
                                print('\nOverall recall for 1:',
                                      mean(lst_re1_stratified)*100, '%')
                                print('\nOverall std/mean re1:',
                                      100*stdev(lst_re1_stratified)/mean(lst_re1_stratified), '%')
                                print('\nOverall precission for 1:',
                                      mean(lst_pr1_stratified)*100, '%')
                                print('\nOverall std/mean pr1:',
                                      100*stdev(lst_pr1_stratified)/mean(lst_pr1_stratified), '%')
                                print('\nOverall precission for 0:',
                                            mean(lst_pr0_stratified)*100, '%')
                                print('\nOverall std/mean pr0:',
                                      100*stdev(lst_pr0_stratified)/mean(lst_pr0_stratified), '%')  

 # Initialize the Gradient Boosting classifier
                                
                                 
                        
x_all= features.to_numpy()       
         
        # Input_ y_Target_Variable.
y_all =clean_tr.iloc[:,20].to_numpy()
from sklearn.model_selection import train_test_split

x_scaled = scaler_minmax.fit_transform(x_all)


x_train_1, x_test, y_train_1, y_test = train_test_split(
    x_scaled, y_all, test_size=0.01, random_state=42, stratify=y_all
)

x_train, y_train = sample.fit_resample(x_train_1, y_train_1)
clf=GradientBoostingClassifier(n_estimators=100, 
    learning_rate=0.1,
    max_depth=3, 
    min_samples_split=4, 
    min_samples_leaf=1, 
    subsample=0.9,
    random_state=42, )
# Train the model on all data
clf.fit(x_train, y_train)     
import pickle

explainer = shap.TreeExplainer(clf)

pickle.dump(explainer, open('shap_models/shap_explaine_hemopneumo.pkl', "wb"))

shap_values = explainer.shap_values(x_test)
print('shap values: ', shap_values)

feature_name=['Sex', 'Age', 'Trauma_type', 'Antplatelet', 'Oral_anticoagulant',
       'Pregnancy', 'Heart_rate', 'Respiratory_rate', 'SpO2', 'Systolic_BP',
       'Diastolic_BP', 'GCS_eyes', 'GCS_verbal', 'GCS_motor', 'GCS_total',
       'RTS_Resp_Rate', 'RTS_Syst_BP', 'RTS_GCS', 'RTS','Triage_code']
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, x_test, feature_names=feature_name,plot_type='bar' ,show=False)
plt.title('SHAP Summary Plot')
plt.show()
              

# Save the trained model
with open('models/gb_for_hepn.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
print('load')    
with open('models/gb_for_hepn.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)    
   
with open('shap_models/shap_explaine_hemopneumo.pkl', 'rb') as model_file:
    loaded_expl = pickle.load(model_file) 
shap_values = loaded_expl.shap_values(x_test[42:43,:])
print('shap values: ', shap_values)

feature_name=['Sex', 'Age', 'Trauma_type', 'Antplatelet', 'Oral_anticoagulant',
       'Pregnancy', 'Heart_rate', 'Respiratory_rate', 'SpO2', 'Systolic_BP',
       'Diastolic_BP', 'GCS_eyes', 'GCS_verbal', 'GCS_motor', 'GCS_total',
       'RTS_Resp_Rate', 'RTS_Syst_BP', 'RTS_GCS', 'RTS','Triage_code']
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, x_test[42:43,:], feature_names=feature_name,plot_type='bar' ,show=False)
plt.title('SHAP Summary Plot')
plt.show()


y_pred = loaded_model.predict(x_test)
print('accu',loaded_model.score(x_test, y_test))
print('re0  ',metrics.recall_score(y_test, y_pred, pos_label=0))
print('re1 ',metrics.recall_score(y_test, y_pred, pos_label=1))
print('pre0 ',metrics.precision_score(y_test, y_pred, pos_label=0))
print('pre1 ',metrics.precision_score(y_test, y_pred, pos_label=1))
print('f1 ',metrics.f1_score(y_test, y_pred, average='macro'))
y_pred_proba = loaded_model.predict_proba(x_test)[::,1]
print('roc ',metrics.roc_auc_score(y_test, y_pred_proba))
    
