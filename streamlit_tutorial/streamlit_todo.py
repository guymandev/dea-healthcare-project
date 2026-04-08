import streamlit as st
 
st.title("To-Do List")
 
# Initialize state
if "tasks" not in st.session_state:
    st.session_state.tasks = []
 
# Form to add tasks
with st.form("task_form"):
    task = st.text_input("New Task")
    add = st.form_submit_button("Add Task")
 
if add and task:
    st.session_state.tasks.append(task)
 
# Display tasks with delete buttons
for i, t in enumerate(st.session_state.tasks):
    col1, col2 = st.columns([4,1])
    col1.write(t)
    if col2.button("X", key=f"del_{i}"):
        st.session_state.tasks.pop(i)
        st.rerun()
