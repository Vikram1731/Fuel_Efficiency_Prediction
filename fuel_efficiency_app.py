import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

model = load('fuel_efficiency_dumpfile.joblib')
ss= load('Standard_Scaler_dumpfile.joblib')

st.title("Fuel Efficiency Prediction")

st.header("Enter Vehicle Features")
cylinder = st.selectbox("Number of Cylinders",[3,4,5,6,8])
displacement = st.number_input("Enter the displacement of the Vehicle",min_value=0)
horsepower = st.number_input("Enter the Horsepower of the vehicle",min_value=38)
weight = st.number_input("Enter the Weight of the Vehicle(lbs)",min_value=130)
acceleration = st.number_input("Enter the accleration of the Vehicle",min_value=0)
year = st.slider("Enter the year of the Model",min_value=70,max_value=82)

feed_data=pd.DataFrame({'cylinders':[cylinder],'displacement':[displacement],'horsepower':[horsepower],
                         'weight':[weight],'acceleration':[acceleration],
                         'model_year':[year]})

scaled_data = ss.transform(feed_data)

if st.button('Press to get the predicted mileage'):
    prediction = model.predict(scaled_data)
    st.success(f'Predicted Mileage: {prediction[0]:.2f}')
    
 

