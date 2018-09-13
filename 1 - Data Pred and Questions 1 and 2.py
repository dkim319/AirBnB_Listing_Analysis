# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:31:00 2018

@author: dkim3
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############################################
# Gather
############################################

# Read in the data
data = pd.read_csv('listings.csv.gz', compression='gzip', error_bad_lines=False)

############################################
# Assess/CLean
############################################

# Check the size
print ('Check the shape')
print (data.shape)

# When checking the dataset, it is apparent that are some variables such as ids,
# urls, dates, and hostname that needs to be removed
# There are also duplicate variables such as neighborhood that needs to be removed as well
print (data.head())

data_new = data.drop(['id','listing_url','host_location','scrape_id','last_scraped','thumbnail_url','medium_url','picture_url','xl_picture_url','host_id','host_url','host_name','host_thumbnail_url','host_picture_url','host_neighbourhood','neighbourhood_group_cleansed','host_verifications','street','neighbourhood','state','market','country_code','country','is_location_exact','calendar_last_scraped','jurisdiction_names','host_since','first_review','last_review','smart_location'], axis = 1)
print('Ids, urls, dates, and duplicate variables removed')
print('')

# For the all text variables, it would be interesting to identify the number of buzz words used, the use of caps, etc
# but for simplicity's sake, we will convert all text variables into word counts after punctuation is removed
import re

text_variables = ['name','summary','space','description','neighborhood_overview','notes','transit','access','interaction','house_rules','host_about']

for tcol in text_variables:
    data_new[tcol + '_word_count'] = data_new[tcol].astype(str).apply(lambda x: [len(re.sub(r'[^\w\s]',' ',x).split())][0])
    data_new = data_new.drop(tcol, axis=1)
print('Converted text variables into word count variables')
print('')

# Now delete any variables that have too many missing values
missing_vals = data_new.isnull().mean()

print(missing_vals.sort_values(ascending=0).head(6))
print('')

# Based on the missing %, the top 6 features should be removed
# the weekly and monthly prices are removed since this analysis will focus on the price variable only
data_new = data_new.drop(['host_acceptance_rate','square_feet','license','monthly_price','weekly_price'], axis = 1)
print('The 6 variables with the most missing values were removed')
print('')

# Any variables are removed that would not be needed in predicting price
data_new = data_new.drop(['security_deposit','cleaning_fee','extra_people'], axis = 1)
print('Security_deposit, cleaning_fee, extra_people were removed since they are associated with price and may not be appropriate for machine learning')
print('')

# There are some variables that needs to transformed to make them usable for analysis
# price needs to be converted to a float datatype
data_new['price'] = data_new['price'].replace('[\$,]','', regex=True).astype(float)
print('Price fixed to become a float data type')
print('')

data_new['host_response_rate'] = data_new['host_response_rate'].replace('[\%,]','', regex=True).astype(float)/100.0
print('Host Reponse Rate fixed to become a float data type')
print('')

# Zipcode needs to be cleaned as well
print (data_new['zipcode'].head(20))
print('')

data_new['zipcode'] = data_new['zipcode'].astype(str).apply(lambda x: [re.sub('[^0-9]','', x)][0][:5])
print ('Zipcode variable was cleaned')
print('')

print (data_new['zipcode'].head(20))
print('')

'''
# Since the gps coordinates are available, the coordinates are used to
# calculated the distance for tourist destination or "anchors" such as
# Hollywood, Santa Monica, Beverly Hills, and Downtown LA

import geopy.distance

# Coordinates are pulled from Google Maps
dtla_coords = (34.0407, -118.2468)
hollywood_coords = (34.0928, -118.3287)
santa_monica_coords = (34.0195, -118.4912)
beverly_hills_coords = (34.0736, -118.4004)

data_new['dist_dtla'] = 0.0
data_new['dist_hw'] = 0.0
data_new['dist_sm'] = 0.0
data_new['dist_bh'] = 0.0

for index, row in data_new.iterrows():
    #print(row['latitude'])
    print(index)
    data_new['dist_dtla'][index] = geopy.distance.vincenty((row['latitude'],row['longitude']),dtla_coords).miles
    data_new['dist_hw'][index] = geopy.distance.vincenty((row['latitude'],row['longitude']),hollywood_coords).miles
    data_new['dist_sm'][index] = geopy.distance.vincenty((row['latitude'],row['longitude']),santa_monica_coords).miles
    data_new['dist_bh'][index] = geopy.distance.vincenty((row['latitude'],row['longitude']),beverly_hills_coords).miles

data_new = data_new.drop(['latitude','longitude'], axis = 1)
print ('Distance to tourist destination variables were added')
print

data_new.to_csv('data-new.csv')
'''

# Read in data w/ the distances calculated (the previous code block takes a long time to run)
data_new = pd.read_csv('data-new.csv', encoding = "ISO-8859-1")

# The amenities variable appears to be a list of amenities, which
# needs to be split out into separate amenities using dummy variables
data_new['amenities_split'] = data_new['amenities'].apply(lambda x: x.replace('"',"").replace("{","").replace("}",'').split(','))
data_new = pd.concat((data_new.drop(['amenities','amenities_split'],axis=1), pd.get_dummies(data_new['amenities_split'].apply(pd.Series).stack(),prefix='a', prefix_sep='_').sum(level=0)),axis = 1)
print ('Dummy variables were create for every single amenity with the Amentities variable')
print('')

##########################################################
# Analyze
##########################################################

##########################################################
# Question #1 - What influences AirBnb prices in the Los Angeles, CA
##########################################################

# Review the dataset metrics
print(data_new.describe())
print('')

def print_chart(df, charttype):
    '''
    This function takes in a dataframe and chart type and generates a chart and clears the plt for the next chart
    '''
    if charttype == 'heatmap':
        sns.heatmap(df, annot=True, fmt=".2f")
    else:
        df.plot(kind=charttype)
    plt.show()
    plt.clf()

# First review the correlation between people accommodated, bathrooms, bedrooms, bed & price
# the heatmap shows a strongest correlation with bathrooms, then bed, then accommodates
#heatmap1 =
print('Price Correlation with Accomodates, Bathrooms, Beds')
print_chart(data_new[['accommodates','bathrooms','bedrooms','beds','price']].corr(),'heatmap')

print_chart(data_new.groupby(['bathrooms'])['price'].mean(),'line')

print('Finding - There is strong correlation between price and accomodates, bathrooms, bedrooms, beds.  There is correlation value between price and bathroom is 0.54 and the correlation value between price and bedrooms is 0.45.  When charting mean price by bathrooms, it shows that the more bathrooms results in a higher price for the most part. ')

# Review the correlation between people accomdated, bathrooms, bedrooms, bed & price
# it looks like there is very little correlation between the availability mettrics and price
print('Price Correlation with Availabilty variables')
print_chart(data_new[['availability_30','availability_60','availability_90','availability_365','price']].corr(),'heatmap')

print('Finding - There is slight correlation between price and availability variables.')
print('')

# Compare prices by neighborhood
print('Price Grouped by Neighborhood - Top 10')
print(data_new.groupby(['neighbourhood_cleansed'])['price'].mean().sort_values(ascending=0).head(10))
print(data_new.groupby(['neighbourhood_cleansed'])['price'].mean().sort_values(ascending=0).tail(10))
print('')

# Look further into neighborhood by analyzing the variance from the mean for each neighborhood
# shows that the neighborhood is big factor in terms of price on both ends
nc_agg = data_new.groupby(['neighbourhood_cleansed'])['price'].mean().sort_values(ascending=0)
nc_agg = (nc_agg - data_new['price'].mean())/data_new['price'].mean()

print('Mean Price by Neighborhood - Top 10')
print_chart(nc_agg.head(10),'bar')

print(nc_agg.head(10))
print('')

print('Mean Price by Neighborhood - Bottom 10')
print_chart(nc_agg.tail(10),'bar')

print(nc_agg.tail(10))
print('')

print('Finding - This shows that location is everything.  Neighborhood has a big impact on price. The most expensive neighborhood has a mean price that is 8 times higher than the overall mean price.  The least expensive neighborhoods has a mean price that is 90% less than the overall mean price.  An interesting aspect of the highest priced neighborhoods is that they are on the outskirts of Los Angeles or by the beach')

# Compare price by city
print('Price Grouped by City - Top 10')
print(data_new.groupby(['city'])['price'].mean().sort_values(ascending=0).head(10))

# Like neighborhood, city has a big factor in terms of price
city_agg = data_new.groupby(['neighbourhood_cleansed'])['price'].mean().sort_values(ascending=0)
city_agg = (city_agg - data_new['price'].mean())/data_new['price'].mean()

print('Mean Price by City - Top 10')
print_chart(city_agg.head(10),'bar')

print('Mean Price by City - Bottom 10')
print_chart(city_agg.tail(10),'bar')

print('Finding - The city analysis mirrors the neighborhood analysis since the neighborhoods almost have a one to one mapping with city.')
print('')

# Compare price by zipcode
print('Price Grouped by Zipcode')
print(data_new.groupby(['zipcode'])['price'].mean().sort_values(ascending=0).head(10))
print('Finding - Zipcode analysis results mirrors the neighborhood and city analysis.  It is simply a different cut.')
print('')



# Compare price by property type
print('Price Grouped by Property Type')
print_chart(data_new.groupby(['property_type'])['price'].mean().sort_values(ascending=0),'bar')

print('Finding - The property type also has an impact on price.  Dome house, villa, and castle have the highest average price.  While, nature lodge, dorm, and hut have the lowest average price.')

print (data_new.groupby(['property_type'])['price'].mean().sort_values(ascending=0).head(10))
print('')

print('Price Grouped by Room Type')
print_chart(data_new.groupby(['room_type'])['price'].mean().sort_values(ascending=0),'bar')

print (data_new.groupby(['room_type'])['price'].mean().sort_values(ascending=0))
print('')

print('Finding - Whole house rentals have a higher mean price compared to private and shared room rentals.')

print('Summary Finding - The accomodates, bathrooms, bedrooms, and beds are correlated with price.  The location related variablees such as neighborhood, city, and zipcode have a huge impact on price as well.  Property type and room type impact price as expected.')

##########################################################
# Analyze
##########################################################

##########################################################
# Question #2 - Does revies or distance to tourist destination impact AirBnb Prices in Los Angeles?
##########################################################

print('Price Correlation with Review Variables')
print_chart(data_new[['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price']].corr(),'heatmap')

review_vars = data_new[['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']].columns

for x in review_vars:
    print_chart(data_new.groupby([x])['price'].mean(),'line')

print('Finding - There are no strong correlation or pattern between price and the review related variables.  Based on the location review chart, it can be implied that higher prices are associated with better location reviews.  Based on the value review chart, it can be implied that higher prices are associted with lower value reviews.')
print('')

print('Price Correlation with Distance Variables')
print_chart(data_new[['dist_dtla','dist_hw','dist_sm','dist_bh','price']].corr(),'heatmap')

dist_vars = data_new[['dist_dtla','dist_hw','dist_sm','dist_bh']].columns

for x in dist_vars:
    print_chart(data_new.groupby([x])['price'].mean(),'line')


print('Finding - There is not a strong correlation between the distance variables and price.  Based on the charts, it looks like the higher priced listings are located within 0 to 10 miles within tourist destinations.  Based on the charts, it looks like the price drops for listings that are greater than 30 to 40 miles away.')
print('')

print('Summary Finding - The review and distance varibles are not as impactful; however, there are some patterns that can be identified.  These variables along with the one identified in the first analysis can be used to create a predictive model.')

