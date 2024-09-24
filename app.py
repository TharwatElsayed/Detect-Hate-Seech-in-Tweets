import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io
import streamlit as st
from streamlit_option_menu import option_menu

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Set Streamlit page title
st.title('Hate Speech and Offensive Language Analysis')

# Create a vertical tab menu in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # Title of the menu
        options=["Home", "About", "Contact"],  # Menu options
        icons=["house", "info", "envelope"],  # Optional icons
        menu_icon="cast",  # Icon for the menu title
        default_index=0,  # Default selected option
        orientation="vertical"  # Set the orientation to vertical
    )

# Display content based on selected tab
if selected == "Home":
    st.title("Home")
    st.write("Welcome to the home page.")
elif selected == "About":
    st.title("About")
    st.write("This is the about page.")
elif selected == "Contact":
    st.title("Contact")
    st.write("This is the contact page.")
