# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:40:01 2024

@author: JEMIMA
"""

"""
Exploratory Data Analysis (EDA)
Creating graphical representations of tdata in order to make it easier to undersatand and analyse
This website data-to-viz.com provides a guide for data visualization

Line plots
Suitable for showing trends over time or over a sequence. 

import matplotlib.pyplot as plt

x_line = [1, 2, 3, 4, 5]
y_line = [2, 4, 6, 8, 10]

plt.plot(x_line, y_line, '-o')
plt.xlabel("x_line")
plt.ylabel("y line")

plt.title('Line Plot')
plt.show() 

Bar plot
Effective for visualising the distribution of categorical variables, count of obsetvations in different groups

import matplotlib.pyplot as plt

x_bar = ['A', 'B', 'C', 'D']
y_bar = [ 1, 2, 3, 4]

plt.bar(x_bar, y_bar)
plt.xlabel("Categories")
plt.ylabel("Values")

plt.title('Bar Plot example')
plt.show()

Scatter plot
Visualise the relationship between two numerical variables

import matplotlib.pyplot as plt

x_scatter = [1, 2, 3, 4, 5]
y_scatter = [2, 4, 6, 8, 10]

plt.scatter(x_scatter, y_scatter)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
plt.show()

Histogram Plot
Provide visual represrmtaion of the distribution of a single variable

import matplotlib.pyplot as plt

x_histogram = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

plt.hist(x_histogram, bins=range(min(x_histogram), max(x_histogram) + 1), edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')
plt.show()

Plotly
Line plot
import plotly.express as px

x_line = [1, 2, 3, 4, 5]
y_line = [2, 4, 6, 8, 10]

fig = px.line(x=x_line, y=y_line, labels={'x' : 'X-axis', 'y': 'Y-axis'}, title='Line Plot')
fig.write_html("plot.html")

#This is used to automatcally open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")

Histogram
import plotly.express as px

x_histogram = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

fig = px.histogram(x=x_histogram, labels={'x': 'Values'}, title='Histogram')
fig.write_html("plot.html")

# This is used to automatically open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")

Bar plot

import plotly.express as px

x_bar = ['A', 'B', 'C', 'D']
y_bar = [1, 2, 3, 4]
fig = px.bar(x=x_bar, y=y_bar, labels={'x': 'Categories', 'y': 'Values'}, title='Bar Plot')
fig.write_html("plot.html")

# This is used to automatically open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")

Scatter plot
import plotly.express as px

x_scatter = [1, 2, 3, 4, 5]
y_scatter = [2, 4, 6, 8, 10]

fig = px.scatter(x=x_scatter, y=y_scatter, labels={'x': 'X-axis', 'y': 'Y-axis'}, title='Scatter Plot')
fig.write_html("plot.html")

# This is used to automatically open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")


Maps

import plotly.express as px

data = px.data.gapminder()

# Create a choropleth world map
fig = px.choropleth(
    data_frame=data,
    locations="iso_alpha",
    color="gdpPercap",
    hover_name="country",
    animation_frame="year",
    title="World Map Choropleth",
    color_continuous_scale=px.colors.sequential.Plasma,
    projection="natural earth"
)

fig.write_html("plot.html")

# This is used to automatically open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")

Combining plots
import plotly.express as px

df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x="year", y="lifeExp", color='country')
fig.write_html("plot.html")

# This is used to automatically open up a browser of your plot
import webbrowser
webbrowser.open("plot.html")
fig.write_html("plot.html")



Numpy

import numpy as np

x = np.arange(-0.1, 0.5, 0.007)

np.arange() is a function in the NumPy library that generates an array of evenly spaced values within a specified interval. 
It's similar to Python's built-in range() function but returns a NumPy array instead of a list. 
You can control the start, stop, and step size of the values.
np.arange([start,] stop, [step,] dtype=None)

print(x)


import numpy as np

x = np.linspace(0, 10.0, 30)

np.linspace(start, stop, num) generates num evenly spaced numbers between start and stop (inclusive).

#np.arange([start,] stop, [step,] dtype=None)
print(x)

convert lists to numpy arrays

 = [1, 2, 3]
aa = np.array(a)
print(a)

b = [[1, 2, 3], [4, 5, 6], [7, 8, -1]]
bb = np.array(b)
print(bb)

#inverse of matrix bb

cc = np.linalg.inv(bb)

matrix multiplication
dd =  np.dot(bb, np.linalg.inv(bb))


Numpy arithmetic

add and subtracting arrays
import numpy as np

p=np.array([1,2,3])
q=np.array([-3,-4,-5])
r=p+q
s=p-q
print(r) 
print(s)

You can check the size (number of elements) in a numpy array like this:

print(p.size)
print(p.shape)
print(p.ndim)
The shape (number of rows and columns) is a list or tuple containing the number of elements along each axis of the array 
(eg. the number of rows and columns in a two-dimensional array) and number of dimensions 
ndim (1D is a row, 2D has rows and columns, 3D has rows, columns and depth, and so on).


The numpy function reshape allows you to rearrange the array so that the elements are arranged in however many rows and columns as you want,
as long as the rows multiplied by the columns stays equal to the total number of elements in the array.

So, to convert p from a 1D to a 2D array, we do this:

import numpy as np

p=np.array([1,2,3])
q=np.array([-3,-4,-5])
r=p+q
s=p-q

p2d=np.reshape(p,[1,3])
print(p2d)

Transpose of an array 
print(p2d.T)

Mulptiply two arrays
f=np.array([1,2,3])
g=np.array([2,2,-3])
print(f*g)

Numpy has it's own maths function, sin, cos, tan, sqrt
plot sin wave and line thingy together
import numpy as np
x= np.arange(0, 10.1, 0.1)
y1 = x*x
y2 = x**2*np.sin(x)

import matplotlib.pyplot as plt

plt.plot(x, y1, 'r*')
plt.plot(x, y2, 'g')

plt.show()

Polynomial curve fit you use polyval
import numpy as np
x = np.arange(0,11)
y=x**2+3*x*np.random.rand(x.size)-1
p = np.polyfit(x,y,2)
xfit=np.arange(0,10.01,0.01)
yfit=np.polyval(p,xfit)
import matplotlib.pyplot as plt
plt.plot(x,y,"g*")
plt.plot(xfit,yfit,"k")
plt.show()

Note that for this example, we used the function rand from the numpy submodule random to introduce some scattter to a polynomial.

One of the optional parameters to the mean and sum functions is axis. 
This allows you to average each column (axis=0) or row (axis=1) and so on for higher dimensional numpy arrays.

import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.mean(a))
print(np.mean(a,axis=0))
print(np.mean(a,axis=1))

There are many more interesting and useful tricks that you can do with numpy arrays; 
here is one that is very powerful but not obvious, which is comparing values elementwise:

k=np.array([4,3,5])
m=np.array([2,3,6])
print(k>m)
print(k<=m)
print(k==m)


Linear Regression

import matplotlib.pyplot as plt

hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

fig = plt.figure()
ax = plt.subplot(111)
ax.set_title("Student Hours vs Test Results")
ax.set_xlabel('Hours (H)')
ax.set_ylabel('Results (%)')
ax.grid()
plt.scatter(hours, results)
plt.show() 



We can clearly see a linear trend showing if the student puts in more hours, they will get better marks however there is some variance. 
So, there must be a correlation between the two data sets. Now how can we derive a relationship from this dataset? That is where linear regression comes in.

So, to describe this relationship we can use the straight-line equation. Remember that m variable is called slope and the c variable is called intercept.

In the machine learning community, the m variable (the slope) is also often called the regression coefficient. The x variable in the equation is the input variable — and y is the output variable.

But in machine learning these x-y value pairs have many alternative names… which can cause some headaches. So here are a few common synonyms that you should know:

input variable (x) – output variable (y)
independent variable (x) – dependent variable (y)
predictor variable (x) – predicted variable (y)
feature (x) – target (y)

In any case let us now work out the straight-line equation in Python. 
For that we will be using our super useful Python library called numpy and its function called polyfit. 
All we need to do is add the following to our code:
    
model = np.polyfit(x, y, 1)
    
What’s the math behind the line fitting. It used the ordinary least squares method (which is often referred to with its short form: OLS). 
It calculates all the errors between all data points and the model. 
Squares each of these error values. Then sum all these squared values. 
Then finds the line where this sum of the squared errors is the smallest possible value.

Now if you print what the model stores you should see:

2.015 x - 3.906

So 2.015 is the slope or regression coefficient and -3.906 is the intercept.

If a student tells you how many hours they studied, you can predict the estimated results of their exam. 
You can do this manually using the equation or using numpy’s method poly1d() as shown below:

predict = np.poly1d(model)
hours_studied = 20
print(predict(hours_studied))
36.38

R-Squared Calculation
There are a few methods to calculate the accuracy of your model namely the R-squared (R2) value. 
I won’t go into the math here, it’s enough if you know that the R-squared value is a number between 0 and 1. 
And the closer it is to 1 the more accurate your linear regression model is.



Unfortunately, R-squared calculation is not implemented in numpy… so we will be doing it by hand using the following code:

import numpy as np
hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]
x = np.asarray(hours)
y = np.asarray(results)
sum_x = np.sum(x)
sum_x_2 = np.sum(np.square(x))
sum_y = np.sum(y)
sum_y_2 = np.sum(np.square(y))
sum_xy = np.sum(x*y)
print(f"sum_x = {sum_x}")
print(f"sum_x_2 = {sum_x_2}")
print(f"sum_y = {sum_y}")
print(f"sum_y_2 = {sum_y_2}")
print(f"sum_xy = {sum_xy}")
n = len(x)
print(f"n = {n}")
top = n*sum_xy - sum_x*sum_y
print(f"top = {top}")
bot_a = np.sqrt(n*sum_x_2 - np.square(sum_x))
bot_b = np.sqrt(n*sum_y_2 - np.square(sum_y))
bot = bot_a*bot_b
print(f"bot = {bot}")
R_2 = np.square(top/bot)
print(f"R_2 = {R_2}")


Output
sum_x = 553
sum_x_2 = 19345
sum_y = 1036
sum_y_2 = 72414
sum_xy = 36814
n = 20
top = 163372
bot = 174378.40331875964
R_2 = 0.8777480188408425


Practically, we should just use a Python library called sklearn that can do this for us but doing the manual approach allows us to learn more about Python. 

import numpy as np
from sklearn.metrics import r2_score

hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

model = np.polyfit(hours, results, 1)
predict = np.poly1d(model)

print(r2_score(results, predict(hours)))

output 
0.8777480188408424


To the plot the regression line over the data we will use Matplotlib. The extra line of code we need to do the regression line is to create a numpy array of values to input into the model. Since our hours range is from 0 to 50 we will use this as our data set upper and lower bonds using the method linspace(). The full code with all the functionality is shown below:

import numpy
import matplotlib.pyplot as plt

hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

mymodel = numpy.poly1d(numpy.polyfit(hours, results, 1))

regline = numpy.linspace(0, 50)

plt.title("Student Hours vs Test Results ")
plt.xlabel("Hours (H)")
plt.ylabel("Results (%)")
plt.grid()
plt.scatter(hours, results)
plt.plot(regline, mymodel(regline))
plt.show()
"""
import numpy
import matplotlib.pyplot as plt

hours = [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50]
results = [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]

mymodel = numpy.poly1d(numpy.polyfit(hours, results, 1))

regline = numpy.linspace(0, 50)

plt.title("Student Hours vs Test Results ")
plt.xlabel("Hours (H)")
plt.ylabel("Results (%)")
plt.grid()
plt.scatter(hours, results)
plt.plot(regline, mymodel(regline))
plt.show()






