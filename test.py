import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris_data = sns.load_dataset('iris')

# Split the data into features and target
X = iris_data[['sepal_length', 'sepal_width', 'petal_length']]
y = iris_data['species']

# Train the logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Iris Species Prediction')
st.write('This app predicts the species of the Iris flower using logistic regression.')

# User input with sliders
st.sidebar.header('User Input')
sepal_length = st.sidebar.slider('Sepal Length', float(X['sepal_length'].min()), float(X['sepal_length'].max()))
sepal_width = st.sidebar.slider('Sepal Width', float(X['sepal_width'].min()), float(X['sepal_width'].max()))
petal_length = st.sidebar.slider('Petal Length', float(X['petal_length'].min()), float(X['petal_length'].max()))

# Predict the species using user input
user_input = [[sepal_length, sepal_width, petal_length]]
prediction = model.predict(user_input)[0]

# Display the predicted species
st.header('Prediction')
st.write('Predicted Species:', prediction)

# Evaluate the model on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the model evaluation metrics
st.header('Model Evaluation')
st.write('Accuracy Score:', accuracy)

# Additional Markdown text
st.header('Additional Information')
st.write('This Streamlit app predicts the species of the Iris flower based on logistic regression.')

# End of the app
