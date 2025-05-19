import os
import numpy as np
import csv
import statsmodels.api as sm
from collections import defaultdict

from info import *


def from_csv(path):
    data = []
    # Load from csv
    filepath = os.path.join(path, 'result.csv')
    with open(filepath, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if there is one
        for row in reader:
            # Ensure data types are converted from string if necessary
            data.append([
                float(row[0]),  # Convert Accessibility to float
                int(row[1]),    # Convert Population to int
                int(row[2]),    # Convert Household Size to int
                float(row[3]),  # Convert Random Impact to float
                float(row[4])   # Convert Critical Impact to float
            ])
    return data

def fit(data):
    data = np.array(data)
    X = data[:, 1:]  # all rows, all columns except the first
    y = data[:, 0]   # all rows, first column

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Create and fit the linear regression model
    model = sm.OLS(y, X).fit()

    return model, X, y

def calculate_y_s(y, X):
    # Calculate y_s: the average y_t for all scenarios that have equal X
    unique_x = defaultdict(list)
    for i, x in enumerate(X):
        unique_x[tuple(x)].append(y[i])

    y_s = np.array([np.mean(unique_x[tuple(x)]) for x in X])

    return y_s

def calculate_custom_metric(y, y_s, y_average):
    # Calculate the custom metric using the provided formula
    numerator = np.sum((y - y_s) ** 2)
    denominator = np.sum((y - y_average) ** 2)
    return numerator / denominator

def r2(y, y_lm, y_average):
    numerator = np.sum((y - y_lm) ** 2)
    denominator = np.sum((y - y_average) ** 2)
    return 1 - (numerator / denominator)

def main():
    # Report
    print(">>> fitting the extracted results...\n")

    path = os.path.dirname(os.path.realpath(__file__))
    data = from_csv(path)

    # Create and fit the linear regression model
    model, X, y = fit(data)

    # Calculate y_average
    y_average = np.mean(y)

    # Calculate y_lm (predicted y)
    y_lm = model.predict(X)

    # Calculate y_s
    y_s = calculate_y_s(y, X)

    # Calculate the custom metric
    custom_metric = calculate_custom_metric(y, y_s, y_average)

    my_r2 = r2(y, y_lm, y_average)

    # Print the coefficients, intercept, and R-squared value
    print('Coefficients: \n', model.params[1:])  # Coefficients for each feature
    print('Intercept: \n', model.params[0])      # Intercept
    print('R^2: \n', model.rsquared)             # R-squared value
    print(model.summary())  # Detailed summary of the regression model
    print('Custom Metric: \n', custom_metric)
    print('R2: \n', my_r2)

    # Report
    print(">>> the extracted results have been fitted successfully.\n")


if __name__ == "__main__":
    main()