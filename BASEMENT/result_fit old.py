import os
import numpy as np
import csv
import statsmodels.api as sm

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

    return model        

def main():
    # Report
    print(">>> fitting the extracted results...\n")

    path = os.path.dirname(os.path.realpath(__file__))
    data = from_csv(path)

    # Create and fit the linear regression model
    model = fit(data)

    # Print the coefficients, intercept, and R-squared value
    print('Coefficients: \n', model.params[1:])  # Coefficients for each feature
    print('Intercept: \n', model.params[0])      # Intercept
    print('R^2: \n', model.rsquared)             # R-squared value
    print(model.summary())  # Detailed summary of the regression model

    # Report
    print(">>> the extracted results have been fitted successfully.\n")


if __name__ == "__main__":
    main()