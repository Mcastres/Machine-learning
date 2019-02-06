# Linear Regression

This repo contains a Linear regression project that estimate a car price depending on its mileage thanks to a dataset

### Prerequisites

Be sure to have Python >= 3.7.2

### Installing packages

Clone the repository and install all necessary packages by running the following command:

```sh
$ pip3 install -r requirements.txt
```

### Dataset

The repo provide a dataset of cars with their price and mileage present in `Datasets/data.csv`

### Linear regression

The project consist of using linear regression with gradient descent algorithm in order to estimate the price of a car depending on its mileage

![](https://i.imgur.com/306wvA1.png)

The principle is simple, the points you see are the data in your dataset (Not exactly ours) and the line is the linear regression.

The lines connecting these points to the linear regression line are called errors. The goal is to calculate the sum of these errors and to reduce it as much as possible by trying to calculate an estimation line.

![](https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif)


So you will have to find two thetas (m and b) to adjust your line as best as possible to reduce the cost function
