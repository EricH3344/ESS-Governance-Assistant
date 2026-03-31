import streamlit as st
from llm import answer_question

st.image("https://essaeg.ca/images/team/ESS-logo-placeholder.jpg", width=150)
st.set_page_config(page_title="ESS Governance Assistant", page_icon="⚖️")

st.title("ESS Governance Assistant")
st.markdown("Ask questions about meeting minutes, policies, and bylaws.")
st.markdown("Be specific when asking about meetings or policy documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What Motion was discussed last BoD meeting?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching records..."):
            # Call your existing RAG pipeline
            response = answer_question(prompt)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})