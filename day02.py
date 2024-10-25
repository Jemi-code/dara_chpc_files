# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:56:50 2024

@author: JEMIMA
"""

"""
Can only store one value

age = 50

Data storage type: Store multiple values:
    -list
    -dictionary
"""

import pandas as pd

"""
ETL = Extract transform load
df = pd.read_csv("data_02/country_data_index.csv")

Files from a url
has no headers
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")


column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None, names= column_names)
print(df)
"""



"""
RExtract
Reading different file types with pandas
text file with semi-colon
df = pd.read_csv("data_02/Geospatial Data.txt", sep=";")

Excel file
df = pd.read_excel("data_02/residentdoctors.xlsx")

Json file
df = pd.read_json("data_02/student_data.json")
"""

"""
Transform

this has the index column column unnamed
df = pd.read_csv("data_02/country_data_index.csv")

this makes the unnamed index column the defacto index column
df = pd.read_csv("data_02/country_data_index.csv", index_col=0)

skip rows
df = pd.read_csv("data_01/insurance_data.csv", skiprows=5)

column headings
column_names = ["duration", "pulse", "max_pulse", "calories"]
df = pd.read_csv("data_02/patient_data.csv",header=None, names=column_names)

Unique delimiter
df = pd.read_csv("data_02/Geospatial Data.txt", sep=";")
the data points are separate with semi-columns not commas

Iconsistent Data types and Names
df = pd.read_excel("data_02/residentdoctors.xlsx")
there is 30yrs and 30yea so we need to fix this to 30 and 30
df['LOWER_AGE'] = df['AGEDIST'].str.extract('(\d+)-')
we are removing the strings
df['LOWER_AGE'] = df['LOWER_AGE'].astype(int) 
(somehow we can create a column by just working on it)
now we are converting the resulting string into integers

other .str actions are .upper() .lower() .replace() .extract()


Working with Dates
df = pd.read_csv("data_02/time_series_data.csv", index_col=0)
convert the date from strings to date time
df['Date'] = pd.to_datetime(df['Date'])
Use if your date format is in the 'DD-MM-YYYY'
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

to separate the dates
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
"""

#NANs and Wrong Formats
df = pd.read_csv("data_02/patient_data_dates.csv")
#Allows you to see all rows
pd.set_option('display.max_rows', None)
#Remove Index column
df.drop(['Index'], inplace=True, axis=1)

#Replace empty values
#we dont want to delete whole rows of useful data because of Nan, so we replace it with other values
x = df["Calories"].mean()
df["Calories"].fillna(x, inplace=True)
#convert dates to datetime
df['Date'] = pd.to_datetime(df['Date'], format="mixed")

print(df.info())
"""
(to remove entire rows df.dropna(inplace=True))
(to remove rows in the Date column you use df.dropna(subset=['Date'], inplace=True))
#remove empty cells using dropna
#remove the row too
df.dropna(inplace=True)
#now we need to reset the index numbering cus we are deleting rows and it doesnt follow automatically like that
df = df.reset_index(drop=True)

#Wrong data; we're trying to replace 450 with 45
Doesnt always have to be null or NaN, maybe wrong decimal or whatever
df.loc[7, 'Duration'] = 45
#or you can remove that row entirely
#df.drop(7, inplace=True)


#Applying Data Transformations
Now we will look at more involved data transformation methods, namely aggregations, appending, merging, and filtering. We will demonstrate how to:

aggregate data using groupby in pandas
append and merge datasets using different join types
filter and manipulate data to create new variables.

Aggregate
grouped = df.groupby('class')

# Calculate mean, sum, and count for the squared values
mean_squared_values = grouped['sepal_length_sq'].mean()
sum_squared_values = grouped['sepal_length_sq'].sum()
count_squared_values = grouped['sepal_length_sq'].count()

# Display the results
print("Mean of Sepal Length Squared:")
print(mean_squared_values)

print("\nSum of Sepal Length Squared:")
print(sum_squared_values)

print("\nCount of Sepal Length Squared:")
print(count_squared_values)

#Read CSV files into dataframes
df1 = pd.read_csv("data_02/person_split1.csv")
df2 = pd.read_csv("data_02/person_split2.csv")

#concatenate the dataframes
df = pd.concat([df1, df2], ignore_index=True)
print(df)

What happens if you two tables with different column names but are related, i.e relational data. For example in the one csv file we have : "id, Company, Name, Department, Job, Title, Skill" as column names and the other: "id, University". If you are familiar with SQL databases then you should know about this.

So if we want to merge these two datasets together. They are related by the "id" column. We can do that with the following code:

    df1 = pd.read_csv("data_02/person_education.csv")
    df2 = pd.read_csv("data_02/person_work.csv")
    df_merge = pd.merge(df1,df2, on='id')
    print(df_merge) 
    
    using outer merge
    df_merge = pd.merge(df1, df2, on='id', how='outer')
    
filtering data
df = pd.read_csv("data_02/iris.csv")

# Filter data for females (class == 'Iris-versicolor')
iris_versicolor = df[df['class'] == 'Iris-versicolor']


# Calculate the average iris_versicolor_sep_length
avg_iris_versicolor_sep_length = iris_versicolor['sepal_length'].mean()

There is also a better way to label the "class" column since the word "Iris-" is redundant. We can remove it in the following way:

df['class'] = df['class'].str.replace('Iris-', '')

If you have your own custom change you want to do to each value:

# Apply the square to sepal length using a lambda function
df['sepal_length_sq'] = df['sepal_length'].apply(lambda x: x**2)

The .apply(lambda x: x**2) part is used to apply a function to each element in the selected 'sepallength' column. In this case, a lambda function is used. The lambda function takes an input parameter x (each individual sepallength value) and squares it.


Load - export the data after you've cleaned and performed the transformations
CSV
 

df.to_csv("data_02/output/iris_data_cleaned.csv")
If you don't want the Pandas index column you can specify:

df.to_csv("data_02/output/iris_data_cleaned.csv", index=False)

Excel
 

df.to_excel("data_02/output/iris_data_cleaned.xlsx", index=False, sheet_name='Sheet1')

JSON
 

df.to_json("data_02/output/iris_data_cleaned.json", orient='records')
"""

"""
df = pd.read_csv("data_02/iris.csv")

df['class'] = df['class'].str.replace('Iris-', '')

# Apply the square to sepal length using a lambda function
df['sepal_length_sq'] = df['sepal_length'].apply(lambda x: x**2)


df.to_json("data_02/output/iris_data_cleaned.json", orient='records')
"""




