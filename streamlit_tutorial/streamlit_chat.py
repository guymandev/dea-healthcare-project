import streamlit as st

st.title("Simple Chat App")
 
# Initialize messages list in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Chat input form
with st.form("chat_form"):
    msg = st.text_input("Enter a message:")
    send = st.form_submit_button("Send")
 
# Append new message if submitted
if send and msg:
    st.session_state.messages.append(msg)
 
# Display chat history
st.subheader("Chat History")
for i, m in enumerate(st.session_state.messages, 1):
    st.write(f"{i}. {m}")
