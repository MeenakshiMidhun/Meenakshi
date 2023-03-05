# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:29:34 2023

@author: iamme
"""
#import pandas and matplotlib modules to plot the graph & read the datas
import pandas as pd
import matplotlib.pyplot as plt

#read the data from csv file into Co2Line,Sectors,Co2Pie, which will be used to plot line,bar & Pie chart respectively
Co2Line=pd.read_csv("plt1.csv")
Sectors=pd.read_csv("plt2.csv")
Co2Pie=pd.read_csv("plt3.csv")

#Print the datas in the file as output 
print(Co2Line)
print(Sectors)
print(Co2Pie)
#define a function as linegraph which represents the data inthe form of line plot
def linegraph():
#call the function to plot the line graph and assign values to the arguments ie value in the X axis, Y axis and label for each line
    plt.figure(figsize=(13,10)) # function is used to adjust the figure size
    plt.plot(Co2Line["YEAR"],Co2Line["CANADA_CO2_PER_GDP"],label="Canada")
    plt.plot(Co2Line["YEAR"],Co2Line["CHINA_CO2_PER_GDP"],label="China")
    plt.plot(Co2Line["YEAR"],Co2Line["GERMANY_CO2_PER_GDP"],label="Germany")
    plt.plot(Co2Line["YEAR"],Co2Line["UK_CO2_PER_GDP"],label="UK")
    plt.plot(Co2Line["YEAR"],Co2Line["US_CO2_PER_GDP"],label="US")
    
    plt.xlabel('YEARS')#xlabel function is called to name the x-axis ie "Years"
    plt.ylabel('CO2 PER GDP')#ylabel function is called to name the y-axis ie "CO2 PER GDP"
    plt.legend(loc="upper right")# legend is used to assign the labels of each line in a box ie "Countries"
    plt.title("Co2 Emission(MtCO₂e) per GDP")# title function is  used to give the title of the plot.
    plt.savefig("Line plot.png")# function is used to save the plot into the system or drive
    plt.show()#function is used to display the plot

# define a function as barchart which represents the data inthe form of bar plot
def barchart():
#call the function to plot the bar graph and assign values to the arguments ie value in the X axis, Y axis , label for each bar and the color
    plt.figure(figsize=(13,10)) # function is used to adjust the figure size
    plt.bar(Sectors["YEAR"]+0.00, Sectors["ELECTRICITY/HEAT"], label="Electricity/Heat",width=0.2,color="red")
    plt.bar(Sectors["YEAR"]+0.2, Sectors["TRANSPORTATION"], label="Transportation",width=0.2,color="blue")
    plt.bar(Sectors["YEAR"]+0.4, Sectors["MANUFACTURING/CONSTRUCTIONS"], label="Manufacturing/Constructions",width=0.2,color="green")
    plt.bar(Sectors["YEAR"]+0.6, Sectors["OTHER FUEL COMBUSTION"], label="Other Fuel Cumbustion",width=0.2,color="yellow")
    plt.xlabel('YEARS')#xlabel function is called to name the x-axis ie "Years"
    plt.ylabel('Co2 Emission (MtCO₂e)')#ylabel function is called to name the y-axis ie "Co2 Emmision (MtCO₂e)"
    plt.legend(loc="upper right")# legend is used to assign the labels of each bar in a box ie "Sectors"
    plt.title("Co2 Emission(MtCO₂e) by different sectors in UK")# title function is  used to give the title of the plot.
    plt.savefig("Bar plot.png")# function is used to save the plot into the system or drive
    plt.show()#function is used to display the plot

# define a function as Piechart which represents the data in the form of Pie plot
def piechart():
#call the function to plot the Pie graph and assign values to the arguments 
    plt.figure(figsize=(8,5)) # function is used to adjust the figure size
    plt.pie(Co2Pie["SECTOR"],autopct= '%1.1f%%',startangle=200)
    plt.legend(bbox_to_anchor = (1,1),labels=Co2Pie["COUNTRY"])# legend is used to assign the labels of each pie in a box ie "Countries"
    plt.title("Co2 Emission(MtCO₂e)at the year 2010")# title function is  used to give the title of the plot.
    plt.savefig("Pie plot.png")# function is used to save the plot into the system or drive
    plt.show()#function is used to display the plot
    
        
linegraph()#call the defined function linegraph to display the plot
barchart()#call the defined function bargraph to display the plot
piechart()#call the defined function Piegraph to display the plot