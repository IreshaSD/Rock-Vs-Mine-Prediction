# Importing necessary libraries
import numpy as np  # To create numpy arrays
import pandas as pd  # For working with dataframes
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import accuracy_score  # For evaluation

# Loading the dataset into a Pandas DataFrame
sonar_data = pd.read_csv('G:/ML_Projects/Rock Vs Mine Prediction/sonar_data.csv', header=None)
print(sonar_data)  # Display the loaded dataset

sonar_data.head()  # Display the first few rows of the DataFrame

# Number of rows and columns in the DataFrame
sonar_data.shape

sonar_data.describe()  # Statistical measures of the data

sonar_data[60].value_counts  # Count of unique values in the 60th column
sonar_data[60].value_counts()  # Count of unique values in the 60th column

sonar_data.groupby(60).mean()  # Mean value for each column for both Rock and Mines

# Separating data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

print(x)
print(y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)

print(x.shape, X_train.shape, X_test.shape)

print(X_train, Y_train)

model = LogisticRegression()

# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data: ", training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data: ", testing_data_accuracy)

# Input data for prediction
input_data = (0.0094, 0.0333, ... )  # Insert your values for prediction

# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make predictions using the model
prediction = model.predict(input_data_reshaped)
print(prediction)

# Display the prediction
if (prediction[0] == "R"):
    print("The object is a Rock")
else:
    print("The object is a mine")
