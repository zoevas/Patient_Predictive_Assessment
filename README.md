# Patient_Predictive_Assessment
PPA is a ML tool that is trained on a health dataset containing information of trauma patients admissioned in Niguarda hospital.
The dataset is private and thus not contained in this repository.
Input features that are used:
['Sex','Age','Trauma_type','Antplatelet','Oral_anticoagulant','Pregnancy','Heart_rate','Respiratory_rate','SpO2','Systolic_BP','Diastolic_BP','GCS_eyes','GCS_verbal','GCS_motor','GCS_total','RTS_Resp_Rate','RTS_Syst_BP','RTS_GCS','RTS','Triage_code']

Four different target predictions are made:
1. Airway help
2. Chest needle usage which predicts hemopneumothorax pathology
3. Massive Blood Transfusion
4. Precardiac arrest

The GridSearch folder contains the fine tuning of the four different models. The user inputs in the prompt which target class models wants to be fine tuned.
The fine tuning results are saved in csv files, and then the models with the best hyperparameters are trained in 'Final Models' folder.

##  Versions
python 3.8
```
cloudpickle 2.2.1
scikit-learn 1.3.0
imblearn 0.0
pandas 10.2.0
shap 0.43.0
seaborn 0.13.2
```
