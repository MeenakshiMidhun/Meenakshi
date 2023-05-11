# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:01:22 2023

@author: iamme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.optimize as opt


''' This function is defined to read the datas which was assigned by
calling the function. From the datas Countries and year will be read
as the arguments are mentioned so. column named Country Code was 
dropped and the values are retrieved using loc[] later they are transposed. '''


def read_file(filename, col, value1,years):
    # Reading Data for dataframe
   df = pd.read_csv(filename, skiprows = 4)
   # Grouping data with col value
   df1 = df.groupby(col,group_keys = True)
   #retriving the data with the all the group element
   df1 = df1.get_group(value1)
   #Reseting the index of the dataframe
   df1 = df1.reset_index()
   #Storing the column data in a variable
   a = df1['Country Name']
   # cropping the data from dataframe
   df1 = df1.loc[countries,years]
   #df1 = df1.drop(columns=['Indicator Name', 'Indicator Code'])
   df1.insert(loc=0, column='Country Name', value=a)
   #Dropping the NAN values from dataframe Column wise
   df1= df1.dropna(axis = 0)
   #transposing the index of the dataframe
   df2 = df1.set_index('Country Name').T
   #returning the normal dataframe and transposed dataframe
   return df1, df2

# Assigning variables to take the specific values from the datasets



countries= [29,35,40,81,109,119,202,251] 
years = ["1990", "1995", "2000", "2005", "2010", "2015"]

# Functions are called and assigned the specific datasets to the variables
df_co2emissions_1, df_co2emissions_2 = read_file("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5358347.csv","Indicator Name","CO2 emissions (kt)", years)
print(df_co2emissions_1,df_co2emissions_2)

x = df_co2emissions_1.iloc[:,1:].values
print(x)

def normalizing(value):
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data


#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

x = df_co2emissions_1.iloc[:,1:].values
x

def normalizing(value):
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data


#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

'''
function to find the no of cluster needed by elbow method
the elbow cluster will be used to find the no of clusters which needed to be created
'''
def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss


k = n_cluster(normalized_df,9)
print(k)

'''
Visualization of Elbow method
where we will be picking the sutable no of clusters
'''
plt.figure(figsize=(15,7))
plt.plot(k)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)

'''
Ploting Kmeans clusters
5 cluster has been plotted with the reference of the elbow method
'''

plt.figure(figsize=(10,7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'blue', label = 'Cluster2')



#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')

plt.legend()
# Title of the  plot
plt.title('Clusters of total Co2 Emission of 8 countries for year 1990 to 2015')
plt.xlabel('Countries')
plt.ylabel('Emission Rate')
plt.show()

df_co2emissions_1['lables']=lables
print('dataframe with cluster lables', df_co2emissions_1)
df_co2emissions_1.to_csv("Total data with cluster label.csv")

'''
calling dataFrame functions for all the dataframe which will be used for curve fitting
'''
countries= [29,35,40,81,109,119,202,251] 
years = ["1990", "1995", "2000", "2005", "2010", "2015"]

co2_1, co2_2 = read_file("API_19_DS2_en_csv_v2_4700503.txt","Indicator Name","CO2 emissions from liquid fuel consumption (kt)", years)

co2_2['mean']=co2_2.mean(axis=1)
co2_2['years'] = co2_2.index

print(co2_2)


ax = co2_2.plot(x = 'years', y = 'mean', figsize=(13, 7), title='Mean emission of 8 country ', xlabel='Years', ylabel= 'Mean')

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and emission rate g."""
    t = t - 1999.0
    f = n0 * np.exp(g*t)
    return f

print(type(co2_2["years"].iloc[1]))
co2_2["years"] = pd.to_numeric(co2_2["years"])
print(type(co2_2["years"].iloc[1]))

#calling exponential function
param, covar = opt.curve_fit(exponential, co2_2["years"], co2_2["mean"],p0=(1995,5.78943))

co2_2["fit"] = exponential(co2_2["years"], *param)

co2_2.plot("years", ["mean", "fit"],
           title='Data fitting with n0 * np.exp(g*t)',
           figsize=(13, 7))
plt.show()
print(co2_2)

'''
function for logistic fit which will be used for prediction of emission
before and after the available years
'''
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and emission rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


#fitting logistic fit
param, covar = opt.curve_fit(logistic, co2_2["years"], co2_2["mean"],
                             p0=(3e12, 0.03, 1990.0), maxfev=6000)

sigma = np.sqrt(np.diag(covar))
igma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
co2_2["logistic function fit"] = logistic(co2_2["years"], *param)
co2_2.plot("years", ["mean", "fit"],
           title='Data fitting with f = n0 / (1 + np.exp(-g*(t - t0)))',
           figsize=(7, 7))
plt.show()

#predicting years
year = np.arange(1960, 2010)
print(year)
forecast = logistic(year, *param)
print('forecast=',forecast)

'''
Visualizing  the values of the emission from your 1990 to 2015 with plot
'''
plt.figure()
plt.plot(co2_2["years"], co2_2["mean"], label="emission")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("emission/year")
plt.legend()
plt.title('Prediction of emission from 1990 to 2015')
plt.show()

import err_ranges as err
low, up = err.err_ranges(year, logistic, param, sigma)

plt.figure()
plt.plot(co2_2["years"], co2_2["mean"], label="emission")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("emission")
plt.legend()
plt.show()
