#!/usr/bin/env python
# coding: utf-8

# In[200]:


# DSP 461: Final Project 
# Carly Carroll


# In[207]:


##### LOAD LIBRARIES #####

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[202]:


##### LOAD EACH CSV FILE INTO PYTHON #####

cost_of_living = pd.read_csv("cost_of_living.csv")
home_values = pd.read_csv("home_values.csv")
rental_values = pd.read_csv("rental_values.csv")
migration = pd.read_csv("migrationdata.csv")
us_census = pd.read_csv("us_census.csv")
crime = pd.read_csv("crimerate.csv")


# In[203]:


##### RESHAPE NECESSARY DATA #####

# create home yearly average costs 
home_values = pd.melt(
    home_values, 
    id_vars=['RegionName', 'StateName'],  # Include StateName in id_vars
    value_vars=home_values.columns[9:], 
    var_name='Date', 
    value_name='HomeValue'
)
home_values['Year'] = pd.to_datetime(home_values['Date'], errors='coerce').dt.year
home_yearly = home_values.groupby(['RegionName', 'StateName', 'Year']).HomeValue.mean().reset_index()  # Group by RegionName, StateName, and Year
home_yearly = home_yearly.rename(columns={'HomeValue': 'HomeValueYearlyAvg'})

# create rental yearly average costs 
rental_values = pd.melt(
    rental_values, 
    id_vars=['RegionName', 'StateName'],  # Include StateName in id_vars
    value_vars=rental_values.columns[9:], 
    var_name='Date', 
    value_name='RentalValue'
)
rental_values['Year'] = pd.to_datetime(rental_values['Date'], errors='coerce').dt.year
rental_yearly = rental_values.groupby(['RegionName', 'StateName', 'Year']).RentalValue.mean().reset_index()  # Group by RegionName, StateName, and Year
rental_yearly = rental_yearly.rename(columns={'RentalValue': 'RentalValueYearlyAvg'})

#group cost of living dataset by county 
cost_of_living = cost_of_living.groupby(['county', 'state']).mean().reset_index()

# fix capitalization errors in the crime data 
crime['County'] = crime['County'].str.title()
crime['State'] = crime['State'].str.upper()

# change migration and census state data from their full names to their abbreviations 
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
migration['State'] = migration['State'].map(state_abbreviations)
us_census['State'] = us_census['State'].map(state_abbreviations)

# rename necessary columns for easier merges 
cost_of_living = cost_of_living.rename(columns={'county': 'RegionName'})
cost_of_living = cost_of_living.rename(columns={'state': 'StateName'})
us_census = us_census.rename(columns={'County': 'RegionName'})
us_census = us_census.rename(columns={'State': 'StateName'})
migration = migration.rename(columns={'County': 'RegionName'})
migration = migration.rename(columns={'State': 'StateName'})
crime = crime.rename(columns={'County': 'RegionName'})
crime = crime.rename(columns={'State': 'StateName'})

# drop any unnecessary columns 
cost_of_living = cost_of_living.drop(columns = ["case_id", "isMetro"])
us_census = us_census.drop(columns = ["CountyId"])


# In[204]:


##### MERGE ALL DATA INTO ONE DATASET AND REMOVE UNNECESSARY COLUMNS #####

affordability = pd.merge(home_yearly, rental_yearly, on=["RegionName", "StateName", "Year"], how="inner")
affordability = pd.merge(affordability, cost_of_living, on=["RegionName", "StateName"], how="inner")
affordability = pd.merge(affordability, us_census, on=["RegionName", "StateName"], how='inner')
affordability = pd.merge(affordability, migration, on=["RegionName", "StateName"], how="outer")
affordability = pd.merge(affordability, crime, on=["RegionName", "StateName", "INflow", "OUTflow", "NET in", "GROSS out"], how="outer")


# In[205]:


##### CREATE AFFORDABILITY SCORE #####

# calculate TotalLivingCost
affordability['TotalLivingCost'] = (
    affordability['housing_cost'] +
    affordability['food_cost'] +
    affordability['transportation_cost'] +
    affordability['healthcare_cost'] +
    affordability['other_necessities_cost'] +
    affordability['childcare_cost'] +
    affordability['taxes']
)

# calculate the average housing costs
affordability['HousingCostAvg'] = (
    affordability['HomeValueYearlyAvg'] + affordability['RentalValueYearlyAvg']
) / 2

# select relevant columns for scaling
factors_affecting_affordability = affordability[['HousingCostAvg', 'TotalLivingCost', 'median_family_income']]

# normalize the factors
scaler = MinMaxScaler()
factors_normalized = pd.DataFrame(scaler.fit_transform(factors_affecting_affordability), columns=factors_affecting_affordability.columns)

# calculate the affordability score 
alpha, beta, gamma = 0.4, 0.4, 0.2
affordability['AffordabilityScore'] = (
    alpha * (1 - factors_normalized['HousingCostAvg']) +
    beta * (1 - factors_normalized['TotalLivingCost']) +
    gamma * factors_normalized['median_family_income']
)


# In[206]:


##### FINAL AFFORDABILITY DATA #####

# drop any rows with NaN values 
affordability = affordability.dropna()

# save affordability data as csv 
affordability.to_csv('affordability.csv', index=False)


# In[ ]:




