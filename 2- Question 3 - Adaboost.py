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

data_new = data_new[['accommodates'	,'bathrooms'	,'bedrooms'	,'beds'	,'availability_30'	,'availability_60'	,'availability_90'	,'availability_365'	,'number_of_reviews'	,'review_scores_rating'	,'review_scores_accuracy'	,'review_scores_cleanliness'	,'review_scores_checkin'	,'review_scores_communication'	,'review_scores_location'	,'review_scores_value'	,'calculated_host_listings_count'	,'reviews_per_month'	,'name_word_count'	,'summary_word_count'	,'space_word_count'	,'description_word_count'	,'neighborhood_overview_word_count'	,'notes_word_count'	,'transit_word_count'	,'access_word_count'	,'interaction_word_count'	,'house_rules_word_count'	,'host_about_word_count'	,'dist_dtla'	,'dist_hw'	,'dist_sm'	,'dist_bh','neighbourhood_cleansed','property_type','room_type','bed_type','price']]

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
data_new['review_scores_value'] = data_new['review_scores_value'].fillna(data_new['review_scores_value'].mean())
data_new['review_scores_location'] = data_new['review_scores_location'].fillna(data_new['review_scores_location'].mean())
data_new['review_scores_checkin'] = data_new['review_scores_checkin'].fillna(data_new['review_scores_checkin'].mean())
data_new['review_scores_communication'] = data_new['review_scores_communication'].fillna(data_new['review_scores_communication'].mean())
data_new['review_scores_cleanliness'] = data_new['review_scores_cleanliness'].fillna(data_new['review_scores_cleanliness'].mean())
data_new['review_scores_accuracy'] = data_new['review_scores_accuracy'].fillna(data_new['review_scores_accuracy'].mean())
data_new['review_scores_rating'] = data_new['review_scores_rating'].fillna(data_new['review_scores_rating'].mean())

print('Impute the host response rate and reviews per month with 0')
#data_new['host_response_rate'] = data_new['host_response_rate'].fillna(0)
data_new['reviews_per_month'] = data_new['reviews_per_month'].fillna(0)
print('')

print('Impute the beds, bathrooms, bedrooms, host listing count, host total listing count to 1')
print('')
data_new['beds'] = data_new['beds'].fillna(1)
data_new['bathrooms'] = data_new['bathrooms'].fillna(1)
data_new['bedrooms'] = data_new['bedrooms'].fillna(1)


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

print('Finding - Price has large standard deviation, which is caused by the large range of possible values.')
print('Min is 0 and the max is 25,000')

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

n_est = 0
results = pd.DataFrame()

#est = [10,20,30,40,50,50,60,70,80,90,100,110,120,130,140,150]
#lr = [0.01,0.5,0.9]

#est = [100,110,120,130,140,150]
#lr = [0.01,0.02,0.001]

est = [110]
lr = [0.01]

# perform parameter tuning and use the parameters that produce the best results
for e in est:
    for l in lr:
        ada_boost = AdaBoostRegressor(n_estimators = e, learning_rate = l, random_state = 319)
        
        ada_boost.fit(features_train, target_train)
        pred_test = ada_boost.predict(features_test)
        pred_train = ada_boost.predict(features_train)
        
        r2_train = r2_score(target_train, pred_train)
        mae_train = mean_absolute_error(target_train,pred_train)        
        
        r2_test = r2_score(target_test, pred_test)
        mae_test = mean_absolute_error(target_test,pred_test)

        print (r2_test)
        print (mae_test)
        
        est_list.append(e)
        lr_list.append(l)      
        r2_list_train.append(r2_train)
        mae_list_train.append(mae_train)
        r2_list_test.append(r2_test)
        mae_list_test.append(mae_test)

results = pd.DataFrame(
        {'est': est_list,
        'lr': lr_list,
        'r2_train': r2_list_train,
        'mae_train': mae_list_train,
        'r2_test': r2_list_test,
        'mae_test': mae_list_test
        })

results.to_csv('results.csv')

final_n_estimators = e
final_lr = l

m1_r2_train = r2_test
m1_mae_train = mae_test

m1_r2_test = r2_test
m1_mae_test = mae_test

print('Adaboost Results:')
print('n_estimators = ', final_n_estimators)
print('learning rate = ', final_lr)
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

    ada_boost = AdaBoostRegressor(n_estimators = final_n_estimators, learning_rate = final_lr, random_state = 319)
    
    ada_boost.fit(features_train_new, target_train)
    pred_test = ada_boost.predict(features_test_new)
    pred_train = ada_boost.predict(features_train_new)

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


# print out results after feature selection
m2_percentile = per
m2_r2_train = r2_train
m2_mae_train = mae_train
m2_r2_test = r2_test
m2_mae_test = mae_test

print('Adaboost Results after Feature Selection:')
print('n_estimators = ', final_n_estimators)
print('learning rate = ', final_lr)
print('percentile = ', m2_percentile)
print('training r2 = ', m2_r2_train)
print('training mae = ', m2_mae_train)
print('testing r2 = ', m2_r2_test)
print('testing mae = ', m2_mae_test)
print('')