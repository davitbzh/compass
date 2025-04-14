import os
from dotenv import load_dotenv

import hopsworks
import streamlit as st

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from functions.prompt_engineering import get_reranker, get_context_and_source, get_answer_from_gemini

import config
import warnings

warnings.filterwarnings('ignore')

# Load the .env file
load_dotenv()

st.title("💬 AI assistant")

# Define a global variable for the OpenAI client
openai_client = None


@st.cache_resource()
def connect_to_hopsworks():
    # Initialize Hopsworks feature store connection
    project = hopsworks.login(
        host="c.app.hopsworks.ai",
        project="davitbzh",
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )

    fs = project.get_feature_store()

    # Retrieve the 'documents' feature view
    administrative_protocols_view = fs.get_feature_view(
        name="administrative_protocols",
        version=1)

    return administrative_protocols_view


@st.cache_resource()
def get_models(saved_model_dir):
    # Load the Sentence Transformer
    sentence_transformer = SentenceTransformer(
        config.MODEL_SENTENCE_TRANSFORMER,
    ).to(config.DEVICE)

    reranker = get_reranker(config.RERANKER)

    return sentence_transformer, reranker


def set_openai_client():
    """Initialize and set the OpenAI client globally."""
    global openai_client
    openai_client = OpenAI(
        api_key=""  # os.environ["OPENAI_API_KEY"],
    )


def predict(user_query, sentence_transformer, feature_view, reranker, model):
    st.write('⚙️ Generating Response...')

    administrative_protocols_view = feature_view

    # Retrieve reranked context and source
    reports_and_source = get_context_and_source(user_query=user_query,
                                                sentence_transformer=sentence_transformer,
                                                feature_view=administrative_protocols_view,
                                                reranker=reranker,
                                                year=2024,
                                                k=50)

    reports_company_context = reports_and_source[0]

    # Generate model response
    if model == "Gemini":
        return get_answer_from_gemini(query=user_query, context=reports_company_context, source=reports_and_source[1],
                                      api_key=os.environ["GEMINI_KEY"])
    else:
        return "Unknown model. Please select Gemini."


# Retrieve the feature view and the saved_model_dir
feature_view = connect_to_hopsworks()

# Load and retrieve the sentence_transformer and reranker
sentence_transformer, reranker = get_models(saved_model_dir=None)

# Model selection
model_choice = st.radio(
    "Select the model you want to use:",
    options=["Gemini"],
    index=0,  # Default selection is the first option
    horizontal=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    response = predict(user_query=user_query,
                       sentence_transformer=sentence_transformer,
                       feature_view=feature_view,
                       reranker=reranker,
                       model=model_choice)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
