import os
import requests
import json
import streamlit as st
import uuid
import datetime
import warnings

warnings.filterwarnings('ignore')

st.title("💬 AI assistant")


def predict(user_query):
    st.write('⚙️ Generating Response...')

    url = "https://snurran.hops.works/hopsworks-api/api/project/2179/inference/serving/compass:predict"
    headers = {
        "Authorization": f"ApiKey {os.environ['HOPSWORKS_API_KEY']}",
        "Content-Type": "application/json"
    }
    data = {
        "instances": [{"user_query": user_query}]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

    return response.json()["predictions"]


def generate_feedback_id():
    """Generate a unique ID for feedback tracking."""
    # Combine UUID with timestamp for extra uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = f"{timestamp}-{str(uuid.uuid4())[:8]}"
    return unique_id


def send_feedback(like_status, feedback_text, user_query, assistant_response):
    url = "https://snurran.hops.works/hopsworks-api/api/project/2179/inference/serving/compass:predict"
    headers = {
        "Authorization": f"ApiKey {os.environ['HOPSWORKS_API_KEY']}",
        "Content-Type": "application/json"
    }

    # Generate a unique feedback ID
    feedback_id = generate_feedback_id()

    data = {
        "instances": [{
            "user_feedback": {
                "feedback_id": feedback_id,
                "like": like_status,
                "feedback": feedback_text,
                "user_query": user_query,
                "assistant_response": assistant_response,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    return response.json()


# Initialize chat history and feedback states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_message_index" not in st.session_state:
    st.session_state.current_message_index = -1

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

# Initialize session ID if not already present
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show feedback options for assistant messages that haven't received feedback
        if message["role"] == "assistant" and i not in st.session_state.feedback_given:
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 4])

                with col1:
                    if st.button("👍 Like", key=f"like_{i}"):
                        feedback_text = st.session_state.get(f"feedback_text_{i}", "")
                        # Get the corresponding user query from previous message
                        user_query = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                        assistant_response = message["content"]
                        send_feedback("Yes", feedback_text, user_query, assistant_response)
                        st.session_state.feedback_given[i] = {"status": "liked", "id": generate_feedback_id()}
                        st.rerun()

                with col2:
                    if st.button("👎 Dislike", key=f"dislike_{i}"):
                        feedback_text = st.session_state.get(f"feedback_text_{i}", "")
                        # Get the corresponding user query from previous message
                        user_query = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                        assistant_response = message["content"]
                        send_feedback("No", feedback_text, user_query, assistant_response)
                        st.session_state.feedback_given[i] = {"status": "disliked", "id": generate_feedback_id()}
                        st.rerun()

                with col3:
                    st.text_input("Additional feedback (optional)", key=f"feedback_text_{i}")
        # Show feedback status if feedback has been given
        elif message["role"] == "assistant" and i in st.session_state.feedback_given:
            feedback_status = st.session_state.feedback_given[i]["status"]
            feedback_id = st.session_state.feedback_given[i]["id"]
            if feedback_status == "liked":
                st.success(f"👍 You liked this response (ID: {feedback_id[:8]}...)")
            else:
                st.error(f"👎 You disliked this response (ID: {feedback_id[:8]}...)")

# React to user input
if user_query := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    response = predict(user_query=user_query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

        # Add feedback options for the new response
        message_index = len(st.session_state.messages)
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 4])

            with col1:
                if st.button("👍 Like", key=f"like_{message_index}"):
                    feedback_text = st.session_state.get(f"feedback_text_{message_index}", "")
                    # Get the user query from the previous message
                    user_query = user_query  # This is the current user query
                    assistant_response = response
                    send_feedback("Yes", feedback_text, user_query, assistant_response)
                    st.session_state.feedback_given[message_index] = {"status": "liked", "id": generate_feedback_id()}
                    st.rerun()

            with col2:
                if st.button("👎 Dislike", key=f"dislike_{message_index}"):
                    feedback_text = st.session_state.get(f"feedback_text_{message_index}", "")
                    # Get the user query from the previous message
                    user_query = user_query  # This is the current user query
                    assistant_response = response
                    send_feedback("No", feedback_text, user_query, assistant_response)
                    st.session_state.feedback_given[message_index] = {"status": "disliked",
                                                                      "id": generate_feedback_id()}
                    st.rerun()

            with col3:
                st.text_input("Additional feedback (optional)", key=f"feedback_text_{message_index}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
