import streamlit as st
import boto3
from prompt_training_clausonnet import untrained

# Initialize the SageMaker runtime client
client = boto3.client('sagemaker-runtime', region_name='us-east-1')

def call_sagemaker_endpoint(endpoint_name, payload):
    """Send input data to the SageMaker endpoint and return the prediction result."""
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=payload
        )
        result = response['Body'].read().decode('utf-8')
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="SageMaker Model Inference", page_icon="🤖", layout="centered")
st.title("🔍 SageMaker Model Inference")
st.markdown("Use this tool to get predictions from different machine learning models hosted on AWS SageMaker.")

# Model selection
model_options = {
    "Untrained Model": "Untrained",
    "Semi-trained Model": "SemiTrainedEndpointHere",
    "Fully trained Model": "FullyTrainedEndpointHere"
}

selected_model = st.selectbox("📌 Select Model Endpoint:", list(model_options.keys()))
endpoint_name = model_options[selected_model]

# User input
user_input = st.text_area("Describe your query", height=150)

# Predict button with improved styling
st.markdown("""
    <style>
    div.stButton > button:first-child { 
        background-color: #4CAF50; 
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

if st.button("🚀 Generate Query"):
    if user_input.strip():
        with st.spinner("Processing your request..."):
            if selected_model == "Untrained Model":
                result = untrained()
            else:
                result = call_sagemaker_endpoint(endpoint_name, user_input)
        
        if "Error:" in result:
            st.error(result)
        else:
            st.success("✅ Query Generation Successful!")
            st.markdown(f"**🔹 Query Result:**")
            st.code(result, language='plaintext')
    else:
        st.warning("⚠️ Please enter some input data for query generation.")
