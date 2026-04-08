import streamlit as st
import pandas as pd
from PIL import Image

st.title("Hello Streamlit!")
st.write("This is my first Streamlit app.")

st.title("Streamlit Basics")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is plain text.")
st.markdown("**Markdown** _is_ supported!")  # Bold and italic

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35]
}
 
df = pd.DataFrame(data)
 
st.dataframe(df)  # Scrollable, interactive table
st.table(df)      # Static table

# Files are missing for code below

# # Image
# image = Image.open("my_image.jpg")
# st.image(image, caption="Sample Image", use_column_width=True)
 
# # Audio
# st.audio("sample_audio.mp3")
# # Video
# st.video("sample_video.mp4")