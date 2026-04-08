import streamlit as st

 

st.title("Travel Planner")

 

# Sidebar inputs

destination = st.sidebar.selectbox(

    "Choose a destination:",

    ["Paris", "Tokyo", "New York"]

)

date = st.sidebar.date_input("Travel date")

 

# Main layout with columns

col1, col2 = st.columns(2)

 

with col1:

    st.header("Activities")

    if destination == "Paris":

        st.write("Visit Eiffel Tower, Louvre Museum, Seine River Cruise.")

    elif destination == "Tokyo":

        st.write("Explore Shinjuku, Visit Senso-ji Temple, Try sushi.")

    else:

        st.write("See Statue of Liberty, Central Park, Broadway show.")

 

with col2:

    st.header("Weather Forecast")

    st.write("Fetching weather data... (demo placeholder)")

 

# Extra tips in an expander

with st.expander("Travel Tips"):

    st.write("Book tickets early, learn basic local phrases, carry power adapters.")

 

# Output

st.success(f"Trip to {destination} planned for {date}!")