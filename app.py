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
        options=["Home", "Previewing the Dataset","About", "Contact"],  # Menu options
        icons=["house", "info", "envelope"],  # Optional icons
        menu_icon="cast",  # Icon for the menu title
        default_index=0,  # Default selected option
        orientation="vertical"  # Set the orientation to vertical
    )

# Display content based on selected tab
if selected == "Home":
    st.title("Introduction")
    st.write("""This dataset contains data related to hate speech and offensive language. 
    Davidson introduced a dataset of tweets categorized using a crowdsourced hate speech vocabulary. 
    These tweets were classified into three categories: hate speech, offensive language, and neither. 
    The dataset, consisting of 24,802 labeled tweets, includes columns for the number of CrowdFlower coders, 
    the count of hate speech and offensive language identifications, and a class label indicating 
    the majority opinion: 0 for hate speech, 1 for offensive language, and 2 for neither.\n
    The dataset published in:\n
    Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May). Automated hate speech 
    detection and the problem of offensive language. In Proceedings of the international 
    AAAI conference on web and social media (Vol. 11, No. 1, pp. 512-515).
    The Dataset can be downloaded from:
    
    https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
    
    https://github.com/t-davidson/hate-speech-and-offensive-language
    
    """)

elif selected == "Previewing the Dataset":
elif selected == "About":
    st.title("About")
    st.write("This is the about page.")
elif selected == "Contact":
    st.title("Contact")
    st.write("This is the contact page.")
