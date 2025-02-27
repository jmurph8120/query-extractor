import streamlit as st
import boto3

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Function to call SageMaker endpoint
def call_sagemaker_endpoint(endpoint_name, payload):
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    result = response['Body'].read().decode('utf-8')
    return result

st.title('Sagemaker Model Inference')

# Dropdown for selecting model type
model_options = ['Untrained Model', 'Semi-trained Model', 'Fully trained Model']
selected_model = st.selectbox('Select Model Endpoint:', model_options)

# Set corresponding endpoint based on user selection
if selected_model == 'Untrained Model':
    endpoint_name = 'UntrainedEndpointHere'
elif selected_model == 'Semi-trained Model':
    endpoint_name = 'SemiTrainedEndpointHere'
else:
    endpoint_name = 'FullyTrainedEndpointHere'

# User input
user_input = st.text_area('Enter input data:')

# Predict button
if st.button('Predict'):
    if user_input.strip():
        result = call_sagemaker_endpoint(endpoint_name, user_input)
        st.write('Prediction:', result)
    else:
        st.write("Please enter some input data for prediction.")
