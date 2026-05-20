import streamlit as st
from agent import get_agent_response

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Nissan Manufacturing AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS (WHITE UI)
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: white;
}

/* Chat container */
.chat-container {
    padding: 20px;
    border-radius: 12px;
    background-color: #f8f9fa;
    margin-bottom: 10px;
}

/* User message */
.user-message {
    background-color: #e9f5ff;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: black;
    font-size: 16px;
}

/* Assistant message */
.assistant-message {
    background-color: #f1f3f5;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: black;
    font-size: 16px;
}

/* Input box */
.stTextInput input {
    background-color: white !important;
    color: black !important;
}

/* Buttons */
.stButton button {
    background-color: #0d6efd;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 18px;
    font-size: 16px;
}

/* Title */
.main-title {
    color: black;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="main-title">Nissan Manufacturing AI Agent- Enter VIN Number for more details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise AI Assistant powered by Bedrock + Pinecone</div>',
    unsafe_allow_html=True
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                <b>You:</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="assistant-message">
                <b>Assistant:</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# USER INPUT
# ==========================================

user_input = st.chat_input("Ask about manufacturing, inventory, operations...")

# ==========================================
# PROCESS USER INPUT
# ==========================================

if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    st.markdown(
        f"""
        <div class="user-message">
            <b>You:</b><br>
            {user_input}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate response
    with st.spinner("Thinking..."):

        try:

            response = get_agent_response(user_input)

        except Exception as e:

            response = f"Error: {str(e)}"

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Display assistant response
    st.markdown(
        f"""
        <div class="assistant-message">
            <b>Assistant:</b><br>
            {response}
        </div>
        """,
        unsafe_allow_html=True
    )