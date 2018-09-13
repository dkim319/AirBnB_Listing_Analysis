# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 20:56:00 2018

@author: dkim3
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost

##########################################################
# Question #3 - Can we use the various attributes of an AirBnB listing to predict price?
##########################################################

# Read in data w/ the distances calculated (the previous code block takes a long time to run)
data_new = pd.read_csv('data_new_ml.csv', encoding = "ISO-8859-1")

# Review the dataset
print(data_new.head(5))

# Remove "unamed:0" column
print('Remove the Unnamed: 0 column')
data_new = data_new.drop('Unnamed: 0', axis = 1)

data_new = data_new[['accommodates','bathrooms'	,'bedrooms','beds','availability_30','availability_60','availability_90','availability_365','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','calculated_host_listings_count'	,'reviews_per_month'	,'name_word_count'	,'summary_word_count'	,'space_word_count'	,'description_word_count'	,'neighborhood_overview_word_count'	,'notes_word_count'	,'transit_word_count'	,'access_word_count'	,'interaction_word_count'	,'house_rules_word_count'	,'host_about_word_count'	,'dist_dtla'	,'dist_hw'	,'dist_sm'	,'dist_bh','neighbourhood_cleansed','property_type','room_type','bed_type','price']]

# need to convert all categorical varaibles into dummy variables
cat_vars = data_new[['neighbourhood_cleansed','property_type','room_type','bed_type']].columns

# this code sript was borrowed from the Categorical Variables notebook 
print('Convert all categorical variables into dummy variables')
#print(cat_vars)

for c in cat_vars:
    # for each cat add dummy var, drop original column
    data_new = pd.concat([data_new.drop(c, axis=1), pd.get_dummies(data_new[c], prefix=c, prefix_sep='_', drop_first=True, dummy_na=True)], axis=1)

# need to impute the columns with missing values
missing_vals = data_new.isnull().mean()
print('The variables with missing values')
print('')
print(missing_vals.sort_values(ascending=0))

print('Impute the review variables with the mean')
print('')

review_vars = data_new[['review_scores_value','review_scores_location','review_scores_checkin','review_scores_communication','review_scores_cleanliness','review_scores_accuracy','review_scores_rating']].columns

for x in review_vars:
    data_new[x] = data_new[x].fillna(data_new[x].mean())

print('Impute the host response rate and reviews per month with 0')
data_new['reviews_per_month'] = data_new['reviews_per_month'].fillna(0)
print('')

print('Impute the beds, bathrooms, bedrooms, host listing count, host total listing count to 1')
print('')

impute_vars = data_new[['beds','bathrooms','bedrooms']].columns

for x in impute_vars:
    data_new[x] = data_new[x].fillna(1)

##################
# Remove 0 price rows
#data_new = data_new[data_new['price'] != 0]

# Take a look at the new dataset
print('The dataset dimensions are now:')
print(data_new.shape)

# Before we start machine learning, let's review the target variable
print('Histogram of Price')
data_new['price'].hist(bins=100)
plt.show()
plt.clf()

data_new[data_new['price'] < 1000]['price'].hist(bins=100)
plt.show()
plt.clf()

print('Finding - Price is heavily skewed and majority of prices are within the 0 to 200 range')
data_new['price'].describe()

print('Finding - Price has large standard deviation, which is caused by the large range of possible values.  Min is 0 and the max is 25,000')
print('')

# Now the dataset is ready, split the data 
features = data_new.drop('price',axis=1)
target = data_new['price']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = .30, random_state=319)

# initalize variables
r2_list_train = []
mae_list_train = []
r2_list_test = []
mae_list_test = []

est_list = []
lr_list = []
depth_list = []
subsample_list = []
colsamplebt_list = []

n_est = 0

# first iteration
#est = [10,20,30,40,50,50,60,70,80,90,100,110,120,130,140,150]
#lr = [0.01,0.5,0.9]
#depth = [5,6,7,8,9,10]
#subsample = [0.5,0.75,1.0]
#colsamplebt = [0.5,0.75,1.0]

# second iteration
#est = [100,150,200,250,300,400,500,600,700,800,900,1000]
#lr = [0.01]
#depth = [7,8,9,10]
#subsample = [0.75,1.0]
#colsamplebt = [0.5]

est = [400]
lr = [0.01]
depth = [9]
subsample = [0.75]
colsamplebt = [0.5]

results = pd.DataFrame()

# perform parameter tuning and use the parameters that produce the best results
for e in est:
    for l in lr:
        for d in depth:
            for s in subsample:
                for c in colsamplebt:
                    xgb = xgboost.XGBRegressor(n_estimators=e, learning_rate=l, gamma=0, subsample=s,
                                               colsample_bytree=c, max_depth=d)
                    
                    xgb.fit(features_train, target_train)
                    pred_test = xgb.predict(features_test)
                    pred_train = xgb.predict(features_train)
                    
                    r2_train = r2_score(target_train, pred_train)
                    mae_train = mean_absolute_error(target_train,pred_train)        
                    
                    r2_test = r2_score(target_test, pred_test)
                    mae_test = mean_absolute_error(target_test,pred_test)
            
                    print (r2_test)
                    print (mae_test)
                    
                    est_list.append(e)
                    lr_list.append(l)   
                    depth_list.append(d)
                    subsample_list.append(s)
                    colsamplebt_list.append(c)
                    
                    r2_list_train.append(r2_train)
                    mae_list_train.append(mae_train)
                    r2_list_test.append(r2_test)
                    mae_list_test.append(mae_test)

results = pd.DataFrame(
        {'est': est_list,
        'lr': lr_list,
        'depth': depth_list,
        'subsample': subsample_list,
        'colsample_bytree': colsamplebt_list,
        'r2_train': r2_list_train,
        'mae_train': mae_list_train,
        'r2_test': r2_list_test,
        'mae_test': mae_list_test
        })

results.to_csv('results.csv')

final_est = e
final_lr = l
final_depth = d
final_subsample = s
final_colsamplebt = c

m1_r2_train = r2_test
m1_mae_train = mae_test
m1_r2_test = r2_test
m1_mae_test = mae_test

print('Xgboost Results:')
print('estimators = ', final_est)
print('learning rate = ', final_lr)
print('depth = ', final_depth)
print('subsample = ', final_subsample)
print('colsample by tree = ', final_colsamplebt)
print('training r2 = ', m1_r2_train)
print('training mae = ', m1_mae_train)
print('testing r2 = ', m1_r2_test)
print('testing mae = ', m1_mae_test)
print('')

# Now after the model has been tuned, use percentile to do feature selection
from sklearn import feature_selection

r2_list_train = []
mae_list_train = []
r2_list_test = []
mae_list_test = []
per_list = []

#percentile = range(1,100)
percentile = [22]

# identify the percentile that will produce the best results 
for per in percentile:
    
    # intilaize SelectFromModel using thresh
    fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile = per)
    feature_model =  fs.fit(features_train,target_train)

    features_train_new = feature_model.transform(features_train)
    features_test_new = feature_model.transform(features_test)

    xgb = xgboost.XGBRegressor(n_estimators=final_est, learning_rate=final_lr, gamma=0, subsample=final_subsample,
                               colsample_bytree=final_colsamplebt, max_depth=final_depth)#,random_state=319)
    
    xgb.fit(features_train_new, target_train)
    pred_test = xgb.predict(features_test_new)
    pred_train = xgb.predict(features_train_new)

    r2_train = r2_score(target_train, pred_train)
    mae_train = mean_absolute_error(target_train,pred_train)        
    
    r2_test = r2_score(target_test, pred_test)
    mae_test = mean_absolute_error(target_test,pred_test)
    
    print (per)
    print (r2_test)
    print (mae_test)
    
    per_list.append(per)
    r2_list_train.append(r2_train)
    mae_list_train.append(mae_train)
    r2_list_test.append(r2_test)
    mae_list_test.append(mae_test)

per_results = pd.DataFrame(
        {'per': per_list,
        'r2_train': r2_list_train,
        'mae_train': mae_list_train,
        'r2_test': r2_list_test,
        'mae_test': mae_list_test
        })

per_results.to_csv('per_results.csv')

m2_percentile = per
m2_r2_train = r2_train
m2_mae_train = mae_train
m2_r2_test = r2_test
m2_mae_test = mae_test

print('Xgboost Results after Feature Selection:')
print('estimators = ', final_est)
print('learning rate = ', final_lr)
print('depth = ', final_depth)
print('subsample = ', final_subsample)
print('colsample by tree = ', final_colsamplebt)
print('percentile = ', m2_percentile)
print('training r2 = ', m2_r2_train)
print('training mae = ', m2_mae_train)
print('testing r2 = ', m2_r2_test)
print('testing mae = ', m2_mae_test)
print('')

