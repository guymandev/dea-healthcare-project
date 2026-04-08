import streamlit as st
import pandas as pd
from PIL import Image
 
# Title
st.title("About Me")
 
# Image
img = Image.open("GuySelfie.jpg")
st.image(img, caption="This is me!", use_column_width=True)
 
# Description
st.write("Hello! I'm a data enthusiast learning Streamlit.")
 
# Skills Table
skills = pd.DataFrame({
    "Skill": ["Python", "Data Analysis", "Machine Learning"],
    "Level": ["Advanced", "Intermediate", "Beginner"]
})
st.table(skills)
 
# Media
st.audio("favorite_song.mp3")
st.video("intro_video.mp4")