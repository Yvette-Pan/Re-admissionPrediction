"""
Created on Tue Nov  6 11:08:18 2018
@author: 
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import re
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import os
os.chdir('E:\\Fordham\\2018 Fall\\CISC 6930\\Final Project')

dataset_raw = pd.read_csv('diabetic_data.csv')
dataset_raw.dtypes
dataset_raw.describe(include = 'all')

### create a new to combine "> 30" and "no" to "no" ###
#getting the number of target
sns.set() # seaborn setting
count_by_readmitted = dataset_raw.groupby('readmitted').size()
print(count_by_readmitted)
count_by_readmitted.plot(kind = 'bar')# inbalanced data


### convert '?' to 'NaN' and make a dataframe copy dataset,all work we done should on this set ###
dataset = dataset_raw.replace('?', np.nan).copy() 

#making the target to be 0,1: 0 means no readmitted, 1 means readmitted.



dataset['readmitted'].replace('<30',1, inplace = True)
dataset['readmitted'].replace('NO',0, inplace = True)
dataset['readmitted'].replace('>30',0, inplace = True)

plt.hist(dataset['readmitted'])


### fix feature types ####

# numeric to nominal
dataset.encounter_id = dataset.encounter_id.astype('object')
dataset.patient_nbr = dataset.patient_nbr.astype('object')

# convert numeric code to category code
dataset['admission_type_id'] = dataset['admission_type_id'].astype('str')
dataset['discharge_disposition_id'] = dataset['discharge_disposition_id'].astype('str')
dataset['admission_source_id'] = dataset['admission_source_id'].astype('str')


dataset.info() # check again



########################################
##### Exploring Categorical Variables ##
########################################

n = dataset.shape[0]
features_all= dataset.columns
'''
  features_all =
        ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'payer_code', 'medical_specialty',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
       'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted'],
      dtype='object']
 '''

### check percentage of missing values ###
df_feature_catg = dataset.select_dtypes(exclude= ['int64'])
feature_catg = df_feature_catg.columns # object features
d_catg = df_feature_catg.shape[1]

# df.isnull().sum(),and How many percerage of missings values we have
dataset_pct_missing = dataset.isnull().sum()/n
dataset_pct_missing.sort_values(ascending=[False])

''' 
insights: there are 7 catg feature have missing values, these are
['weight', 'medical_specialty', 'payer_code', 'race', 'diag_3', 'diag_2','diag_1'
insights: Weight has 96.86% of missing values
'''

### Histogram ###

##########################################
#Plotting numeric hist
numeric = np.array(dataset.describe().columns)

'''
'time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses'
'''
hist_matrix = dataset[numeric].hist()
#dataset['number_outpatient'].plot(kind='hist')


#distributions graphes for numeric features:
dist_matrix = dataset[numeric].plot(kind='density', subplots=True, layout=(2,4), sharex=False, sharey = False)
plt.show()    # No graphs showing

#boxplot graphes for numeric features:
boxplot_matrix = dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()
##########################################




### plot scatterplot matrix for numeric data ###
sns.set(style="ticks")
#scatters between numeric features and color by targets
sns.pairplot(dataset[numeric],palette='husl',hue = 'readmitted', plot_kws={'alpha': 0.5})


### plot heatmap of coeffiecient for numeric data###
sns.heatmap(dataset[numeric].corr()) 



#######################################
####### Handling Missing Values #######
#######################################


# drop weights, very sparse 
dataset = dataset.drop(labels='weight',axis = 1)
# drop observation that has gender of unknown value
dataset = dataset[dataset['age']!= 'Unknown/Invalid']


# explore payercode, calcuting readmissoin distribution in each of the group 
payercode_bar = sns.countplot(x = 'payer_code', hue = 'readmitted', data = dataset)
# seems like payercode is not a signficant factor that affect readmisson 
#drop payercode
dataset = dataset.drop(labels='payer_code',axis = 1)



# assign medical specialty with mode based on age 

count_by_medspe = dataset.groupby('medical_specialty').size()
sns.countplot(x = 'age',hue = 'medical_specialty', data = dataset)


def conditional_fillna_base_on_mode(conditioned_feature,missing_value_feature, dataset):
    pair = {}
    age_grouped = dataset.groupby(conditioned_feature)
    for age, group in age_grouped:
        medical_grouped = group.groupby([missing_value_feature]).size().sort_values(ascending=False)  
        mode = medical_grouped.keys()[0]
        pair[age] = mode
    dataset[missing_value_feature] = dataset[missing_value_feature].replace(np.nan, pair[age])
    return dataset

dataset = conditional_fillna_base_on_mode('age','medical_specialty',dataset)



# race: fill in missing data with the mode
dataset['race'].replace(np.nan,'Caucasian', inplace = True)

# drop the duplicated encounter_ID. Keep the last entry(Needs to check)
#dataset.drop_duplicates(subset = ['patient_nbr'], keep = 'last', inplace = True)
# decided to keep both records if duplicated. will make a feature to flag it.
# show how many duplicates there are for each patient_nbr

duplicates = dataset[dataset.duplicated(['patient_nbr'], keep=False)]
duplicated_count = duplicates.groupby('patient_nbr').size()
duplicated_count.to_csv('duplicated.csv') # only run this once


# drop paitence death
dead_code = ['11','12','13','14','19','20','21','25','26']
drop_discharged_index = dataset[dataset['discharge_disposition_id'].isin(dead_code)].index
dataset.drop(drop_discharged_index,inplace = True)                                       
#101766 - 3415 = 98351



#2.dropping the instances who have missing values in diag1,2,3
dataset.dropna(axis = 0, subset=['diag_1','diag_2','diag_3'], inplace = True)
#98351- 1485 = 96866



    





###############################
######## Encoding #############
###############################

#####Merge columns

# Mapping 
dataset['admission_source_id'].replace(['1','2'], 'Admitted because of physician/clinic referral', inplace = True)
dataset['admission_source_id'].replace('7', 'Admitted from emergency room', inplace = True)
# Merging
dataset['admission_source_id'][~dataset['admission_source_id'].isin(['Admitted because of physician/clinic referral',
        'Admitted from emergency room'])] = 'Others'
# check all the unique levels of admission source
dataset['admission_source_id'].unique()

# Mapping
dataset['discharge_disposition_id'].replace('1', 'Discharged to home', inplace = True)
dataset['discharge_disposition_id'][~dataset['discharge_disposition_id'].isin(['Discharged to home'])] = 'Others'
# check again
dataset['discharge_disposition_id'].unique()

# Combine every sub-categories of surgery into 'Surgery'
dataset['medical_specialty'][dataset['medical_specialty'].isin(['Surgery-General',
        'Surgery-Neuro','Surgery-Cardiovascular/Thoracic',
        'Surgery-Colon&Rectal','Surgery-Plastic','Surgery-Thoracic',
        'Surgery-PlasticwithinHeadandNeck','Surgery-Pediatric',
        'Surgery-Maxillofacial','Surgery-Vascular','Surgery-Cardiovascular'])] = 'Surgery'

dataset['medical_specialty'][~dataset['medical_specialty'].isin(['InternalMedicine',
        'Family/GeneralPractice', 'Cardiology', 'Surgery'])] = 'Others'
# Check again
dataset['medical_specialty'].unique()


#####Label Encoding
#medical-specific features
labelencodinglist = ['metformin',
                     'repaglinide','nateglinide','chlorpropamide','glimepiride',
                     'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
                     'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide',
                     'citoglipton','insulin','glyburide-metformin','glipizide-metformin',
                     'glimepiride-pioglitazone','metformin-pioglitazone', 'metformin-rosiglitazone']

labelencoder = LabelEncoder()
for feature in labelencodinglist:
    labelencoder.fit(['No','Down','Steady','Up'])
    dataset[feature] = labelencoder.transform(dataset[feature])


#age
# need to merge them into three groups <30, 30-60, 60-100
dataset['age'].replace(['[0-10)','[10-20)', '[20-30)'],'<30', inplace = True)
dataset['age'].replace(['[30-40)','[40-50)', '[50-60)'],'30-60', inplace = True)
dataset['age'].replace(['[60-70)', '[70-80)','[80-90)', '[90-100)'],'60-100', inplace = True)    


age_enc = dataset['age'].unique()
labelencoder.fit(age_enc)
dataset['age'] = labelencoder.transform(dataset['age'])

#max_glu_serum
max_glu_serum_enc = dataset['max_glu_serum'].unique()
labelencoder.fit(['None','Norm','>200','>300'])
dataset['max_glu_serum'] = labelencoder.transform(dataset['max_glu_serum'])


#A1Cresult
A1Cresult_enc = dataset['A1Cresult'].unique()
labelencoder.fit(['None','Norm','>7','>8'])
dataset['A1Cresult'] = labelencoder.transform(dataset['A1Cresult'])


#### Onehot Encoding
onehotencodinglist = ['race','gender','medical_specialty','change','diabetesMed',
                      'diag_1','diag_2','diag_3', 'admission_source_id', 'discharge_disposition_id'
                      ]
def get_dummies_prefix(feature):
    enc = pd.get_dummies(dataset[feature])
    enc.columns = dataset[feature].unique()
    enc_prefix = enc.add_prefix(feature+'_')
    return enc_prefix


# encoded dataframe for each feature    
race_enc = get_dummies_prefix('race')
# randomly drop one column of the dataframe generated by onehot encoding
race_enc = race_enc.drop(race_enc.sample(1,axis = 1).columns, axis = 1)

gender_enc = get_dummies_prefix('gender')
gender_enc = gender_enc.drop(gender_enc.sample(1,axis = 1).columns, axis = 1)

med_specialty_enc = get_dummies_prefix('medical_specialty')
med_specialty_enc = med_specialty_enc.drop(med_specialty_enc.sample(1,axis = 1).columns, axis = 1)

change_enc = get_dummies_prefix('change')
change_enc = change_enc.drop(change_enc.sample(1,axis = 1).columns, axis = 1)

diabetesMed_enc = get_dummies_prefix('diabetesMed')
diabetesMed_enc = diabetesMed_enc.drop(diabetesMed_enc.sample(1,axis = 1).columns, axis = 1)

admission_enc = get_dummies_prefix('admission_source_id')
admission_enc = admission_enc.drop(admission_enc.sample(1,axis = 1).columns, axis = 1)

discharge_enc = get_dummies_prefix('discharge_disposition_id')
discharge_enc = discharge_enc.drop(discharge_enc.sample(1,axis = 1).columns, axis = 1)


##Diag1,2,3 merging categories and encoding
#Diag1 
data_diag = dataset[['diag_1','diag_2','diag_3']]
#converting Vdd and Edd to 9999,floating number
data_diag['diag_1'] = data_diag['diag_1'].str.replace(r'[A-Z]\d+', '9999')
data_diag['diag_2'] = data_diag['diag_2'].str.replace(r'[A-Z]\d+', '9999')
data_diag['diag_3'] = data_diag['diag_3'].str.replace(r'[A-Z]\d+', '9999')


def create_list (first, last, extra_val = -999):
    if extra_val == -999:
        return list(range(first, last+1))
    else:
        other = list(range(first, last+1))
        print(other)
        other.append(extra_val)
        return other

data_diag.diag_2 = data_diag.diag_2.astype(float)
data_diag.diag_1 = data_diag.diag_1.astype(float)
data_diag.diag_3 = data_diag.diag_3.astype(float)
df = data_diag

list_of_unique_ints = pd.unique(df.values.ravel())

circ = create_list(390, 459, 758)
resp = create_list(460, 519, 786)
dige = create_list(520, 579, 787)
inj  = create_list(800, 999)
musc = create_list(710, 739)
geni = create_list(580, 629, 788)
neo =  create_list(140, 239)

df = df.astype(int)
df = df.replace(circ, 'Circulatory')
df = df.replace(resp, 'Respiratory')
df = df.replace(dige, 'Digestive')
df = df.replace(250, 'Diabetes' c
df = df.replace(inj, 'Injury')
df = df.replace(musc, 'Musculoskeletal')
df = df.replace(geni ,'Genitourinary')
df = df.replace(neo, 'Neoplasms')
df = df.replace(list_of_unique_ints, 'Other')

data_diag_ = pd.get_dummies(df)



# merge all dataframes
new_dataset = pd.concat([dataset,race_enc,gender_enc,med_specialty_enc,change_enc,admission_enc,
                         discharge_enc,diabetesMed_enc, data_diag_], axis = 1)
# remove original feature columns
new_dataset.drop(labels =onehotencodinglist,axis = 1, inplace = True)
new_dataset.to_csv('new_dataset_2.csv')



### plot scatterplot matrix for new_clean data ###
sns.set(style="ticks")
#scatters between numeric features and color by targets
sns.pairplot(new_dataset,palette='husl',hue = 'readmitted', plot_kws={'alpha': 0.5})

                
### plot heatmap of coeffiecient for new data###
sns.heatmap(new_dataset.corr()) 






















