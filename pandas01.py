# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:52 2024

@author: JEMIMA
"""
#Reading files
#import pandas as pd

#country data file
#file = pandas.read_csv("country_data.csv")

#diab data file
#file = pandas.read_csv("diab_data.csv")

#diab data file
#file = pandas.read_csv("housing_data.csv")

#diab data file
#file = pandas.read_csv("insurance_data.csv")

#diab data file
#file = pandas.read_csv("iris.csv")

#print(file)
#print(file.info())
#print(file.describe())

"""
DATA SCIENCE - ETL (Extract, Transform, Load)
Methods:
    .descibe()
    .info()
    
file_path = "C:/Users/JEMIMA/Desktop/Old Files/JEMIMAH STUFFIES/DARA/Computer_training/Day_1/Day_1/iris.csv"
dataframe = pd.read_csv(file_path)


#how to skip first 6 rows when reading csv files
df = pd.read_csv("insurance_data.csv", skiprows=(6))
"""


#Storing Data
#B1 = 30
#B2 = 40
#B3 = 30
#B4 = 49

#print(B1)
#print(B2)

#Using Lists
#age = [30, 40, 30, 49, 22, 35, 22, 46, 29, 25, 39]
#print(age)

#print(age[11])

#print(min(age))
#print(max(age))
#print(len(age))
#print(sum(age))
#average = sum(age)/len(age)
#print(average)

#Storing Text
#C2 = "M"
#C3 = "M"
#C4 = "F"

#Data Storage with Lists
#my_list = [42, -2021, 6.283, "tau", "node"]
#print(my_list)
#print(my_list[:])

#my_list.append("pi")
#print(my_list)

#my_list.insert(1, "pi2")
#print(my_list)

#my_list.remove("pi")
#my_list.remove("pi2")
#my_list.remove("tau")
#print(my_list)
#print(len(my_list))

#View a certain range of items:
#print(my_list[0:3])

#Dictionaries

""""

d = {'key1':'value1', 'key2':'value2'}
freq = [20, 30, 40]
color = ["blue", "green", "yellow"]
my_dc = {"frequency": freq, "color":color}

person = {'name': 'John Doe', 'age': 30, 'address': '123 Main St.'}
#print(person['name'])
#print(person.get('age'))
person['phone'] = '555-555-5555'

#Creating a DataFrame
data = {
     'age' : [30, 40, 30, 49, 22, 35, 22, 46, 29, 25, 39],
     'gender' : ["M", "M", "F", "M", "F", "F", "F", "M", "M", "F", "M"],
     'country' : ["South Africa", "Botswana", "South Africa", "South Africa", "Kenya", "Mozambique", "Lesotho", "Kenya", "Kenya", "Egypt", "Sudan"]
        }

#df = data frame
df = pd.DataFrame(data)

#Displaying the DataFrame
print(df)

#Accessing specific columns
print(df['age'])
print(df['gender'])

#Basic Statistics
print(df['age'].min())
print(df['age'].max())
print(df['age'].mean())

#Filtering data
print(df[df['age'] > 30])

#Slicing data
print(df[1:4])

#Adding a new column
df['new_column'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(df)

#Removing a column
df.drop(columns=['new_column'], inplace=True)
print(df)

"""