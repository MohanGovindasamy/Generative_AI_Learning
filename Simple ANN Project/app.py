# Step 01: Import required libiraries or Dependencies
# - tensorflow: for building and training the artificial neural network (ANN)
# - Keras: A high-level API built on top of TensorFlow for building and training neural networks.
# - numpy: for numerical computations
# - pandas: for data manipulation and analysis
# - scikit-learn: for preprocessing and evaluation
# - matplotlib: for data visualization
# - tensorboard: for visualizing the training process
# - streamlit: for creating an interactive web application

import numpy as np 
import pandas as pd 
import tensorflow 
import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle 
import streamlit


# Step 02: Load the Trained Model
model = tensorflow.keras.models.load_model('./my_model_keras.keras')

#Step 03: Load Encoders and Scaler
# Load encoded labels for Gender
with open('encode_label_gender.pkl', 'rb') as file:
    encode_label_gender = pickle.load(file)

# Load OneHot Encoded - Geography
with open('oneHot_encode_geography.pkl', 'rb') as file:
    oneHot_encode_geography = pickle.load(file)

# Load standard Scaller
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Step 04: Create Streamlit App UI
streamlit.title("Simple Customer Churn Prediction using Artificial Neural Network (ANN)")

# Step 05: Gather User Input by creating (Streamlit provides interactive elements for user input:)
# creating user inputs in streamlit apps
Credit_Score = streamlit.number_input("Credit Score : ")
Geography = streamlit.selectbox('Geography : ' ,oneHot_encode_geography.categories_[0]) # From dropdown it displays as ['France', 'Germany', 'Spain']
Gender = streamlit.selectbox('Gender : ', encode_label_gender.classes_) # From dropdown it diaplayes as ['Male', 'Female']
Age = streamlit.slider("Age : ", 18, 92) # This will display sideler, user can select Age from the range
Tenure = streamlit.slider("Trenure : ", 0, 10) # This will display sideler, user can select Tenure from the range
Balance = streamlit.number_input("Balance : ")
Num_Of_Products = streamlit.slider("Number of Products : ", 0, 4) # This will display sideler, user can select No.of Products from the range
Has_Credit_Card = streamlit.selectbox("Has Credit Card? : ", [0, 1]) # This will display dropdown, user can select 'has Credit card?' from [0, 1]
Is_Active_Member = streamlit.selectbox("Is Active Number? : ", [0, 1]) # This will display dropdown, user can select 'Is Active Number?' from [0, 1]
Estimated_Salary = streamlit.number_input("Estimated Salary : ")
Exited = streamlit.text_input("Expected Result - Exited? : ")

# - streamlit.selectbox(): Dropdown selection.
# - streamlit.slider(): Slider for numerical selection.
# - streamlit.number_input(): Input box for entering numbers.

# Step 06: Prepare input data for Model
input_data = pd.DataFrame({
    'CreditScore': [Credit_Score], 
    # 'Geography' : [oneHot_encode_geography.transform[columns=]],  # will be process after this need to do one-hot encode
    'Gender': [encode_label_gender.transform([Gender]) [0]], #transform into label encoder
    'Age' : [Age], 
    'Tenure' : [Tenure], 
    'Balance' : [Balance] ,
    'NumOfProducts' : [Num_Of_Products] , 
    'HasCrCard' : [Has_Credit_Card],
    'IsActiveMember' : [Is_Active_Member], 
    'EstimatedSalary' : [Estimated_Salary]
})

# Step 07: Apply One-Hot Encoding to Geography
# - Transforms Geography into One-Hot Encoding format.
geography_encoder = oneHot_encode_geography.transform([[Geography]])
# - Creates a new DataFrame for encoded Geography.
geography_encoder_dataFrame = pd.DataFrame( geography_encoder ,columns= oneHot_encode_geography.get_feature_names_out(['Geography']))

# Step 08: Combine Encoded Data
# Add this one-hot endoer 'Geography' into input data
# - Resets index to avoid merging issues.
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoder_dataFrame], axis=1)


# Step 09: Standard scaler for input data (Feature Scaling)
# input data contains a mix of integer and string column names, but scikit-learn expects all feature names to be strings.
# Convert Column Names to Strings
input_data.columns = input_data.columns.astype(str)

# Applies standardization to all numerical values for model compatibility.
scaler_data = scaler.transform(input_data)

# Step 10: Make Predictions
predection = model.predict(input_data)
predection_score = predection[0][0] # array([[0.72]]) => 0.72 will be output


# Step 11: Display Results in Streamlit
streamlit.write(f'Churn Probability : {predection_score:.2f}') # Shows the churn probability rounded to 2 decimal places.

if predection_score > 0.5:
    streamlit.write("Customer is Likely to be Churn.") # If churn probability > 0.5, the customer is likely to churn.
else:
    streamlit.write("Customer is Not Likely to be Churn.") #Otherwise, they are not likely to churn.

streamlit.write('Entered - Exited values is '+ Exited )

# to Run streamlit enter following code in command prompt
# streamlit run app.py