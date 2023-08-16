# Import necessary libraries
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import required modules from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

 # Read the dataset from a CSV file
dataset = pd.read_csv("d:\dataset.csv")

# Display the first 5 rows of the dataset
print(dataset.head(5))

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], 
                                                    dataset.iloc[:, -1], 
                                                    test_size=0.3, 
                                                    random_state=42)

# Display information about the training data
print(X_train.info())

# Drop the "Name" column from the training and testing data
X_train = X_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]

# Extract the "Manufacturer" from the "Name" column and create a new column for it
make_train = X_train["Name"].str.split(" ", expand=True)
make_test = X_test["Name"].str.split(" ", expand=True)
X_train["Manufacturer"] = make_train[0]
X_test["Manufacturer"] = make_test[0]

# Plot a count of cars based on manufacturers
plt.figure(figsize=(12, 8))
plot = sns.countplot(x='Manufacturer', data=X_train)
plt.xticks(rotation=90)
for p in plot.patches:
    plot.annotate(p.get_height(),
                  (p.get_x() + p.get_width() / 2.0, p.get_height()),
                  ha='center',
                  va='center',
                  xytext=(0, 5),
                  textcoords='offset points')
plt.title("Count of cars based on manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count of cars")

# Drop unnecessary columns and calculate the age of the car
X_train.drop("Name", axis=1, inplace=True)
X_test.drop("Name", axis=1, inplace=True)
X_train.drop("Location", axis=1, inplace=True)
X_test.drop("Location", axis=1, inplace=True)
curr_time = datetime.datetime.now()
X_train['Age'] = X_train['Year'].apply(lambda year: curr_time.year - year)
X_test['Age'] = X_test['Year'].apply(lambda year: curr_time.year - year)

# Preprocess the "Mileage," "Engine," "Power," and "Seats" columns
mileage_train = X_train["Mileage"].str.split(" ", expand=True)
mileage_test = X_test["Mileage"].str.split(" ", expand=True)
X_train["Mileage"] = pd.to_numeric(mileage_train[0], errors='coerce')
X_test["Mileage"] = pd.to_numeric(mileage_test[0], errors='coerce')
X_train["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace=True)
X_test["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace=True)

engine_cc_train = X_train["Engine"].str.split(" ", expand=True)
engine_cc_test = X_test["Engine"].str.split(" ", expand=True)
X_train["Engine"] = pd.to_numeric(engine_cc_train[0], errors='coerce')
X_test["Engine"] = pd.to_numeric(engine_cc_test[0], errors='coerce')
X_train["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace=True)
X_test["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace=True)

power_bhp_train = X_train["Power"].str.split(" ", expand=True)
power_bhp_test = X_test["Power"].str.split(" ", expand=True)
X_train["Power"] = pd.to_numeric(power_bhp_train[0], errors='coerce')
X_test["Power"] = pd.to_numeric(power_bhp_test[0], errors='coerce')
X_train["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace=True)
X_test["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace=True)

X_train["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace=True)
X_test["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace=True)

# Drop the "New_Price" column, and convert categorical variables to dummy variables
X_train.drop(["New_Price"], axis=1, inplace=True)
X_test.drop(["New_Price"], axis=1, inplace=True)
X_train = pd.get_dummies(X_train,
                         columns=["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first=True)

# Ensure that the test set has the same columns as the training set after encoding
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

# Standardize the data using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_scaled, y_train)
y_pred_linear_regression = linear_regression_model.predict(X_test_scaled)

# Calculate and print the R-squared score for Linear Regression
r2_linear_regression = r2_score(y_test, y_pred_linear_regression)
print("R-squared score (Linear Regression):", r2_linear_regression)

# Train a Random Forest Regression model
random_forest_model = RandomForestRegressor(n_estimators=100)
random_forest_model.fit(X_train_scaled, y_train)
y_pred_random_forest = random_forest_model.predict(X_test_scaled)

# Calculate and print the R-squared score for Random Forest Regression
r2_random_forest = r2_score(y_test, y_pred_random_forest)
print("R-squared score (Random Forest Regression):", r2_random_forest)
