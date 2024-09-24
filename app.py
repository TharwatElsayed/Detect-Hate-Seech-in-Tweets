import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Set Streamlit page title
st.title('Hate Speech and Offensive Language Analysis')

# Horizontal line separator
st.markdown("---")

st.subheader('Introduction')
st.write("""This dataset contains data related to hate speech and offensive language. 
Davidson introduced a dataset of tweets categorized using a crowdsourced hate speech vocabulary. 
These tweets were classified into three categories: hate speech, offensive language, and neither. 
The dataset, consisting of 24,802 labeled tweets, includes columns for the number of CrowdFlower coders, 
the count of hate speech and offensive language identifications, and a class label indicating 
the majority opinion: 0 for hate speech, 1 for offensive language, and 2 for neither.\n
The dataset published in:\n
Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May). Automated hate speech detection and the problem of offensive language. In Proceedings of the international AAAI conference on web and social media (Vol. 11, No. 1, pp. 512-515).\n

The Dataset can be downloaded from:\n
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset\n
https://github.com/t-davidson/hate-speech-and-offensive-language
""")

# Horizontal line separator
st.markdown("---")

st.subheader('Loading and Previewing the Dataset')

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dataset Information", "Dataset Description", "Dataset Overview", "Missing values"])

# Tab 1: Dataset Brief Information
with tab1:
    st.subheader('Dataset Information')

    # Capture the df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    # Display the info in Streamlit
    st.text(s)

# Tab 2: Dataset Columns Description
with tab2:
    st.subheader('Dataset Columns Description')
    st.write(df.describe(include='all'))

# Tab 3: Dataset Overview
with tab3:
    st.subheader('Dataset Overview (Before Preprocessing)')
    st.write(df.head(10))

# Tab 4: Dataset Overview
with tab4:
    # Check for missing data
    st.subheader("Missing values in each column:")
    st.write(df.isnull().sum())
   

# Horizontal line separator
st.markdown("---")

st.subheader('Understanding Class Distribution')

# Create a radio button for chart selection
chart_type = st.radio("Choose Chart Type", ('Bar Chart', 'Pie Chart'))

# Display the selected chart
if chart_type == 'Bar Chart':
    st.subheader('Distribution of Classes (Bar Chart)')
    plt.figure(figsize=(8, 6))
    plt.hist(df['class'], bins=3, edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes')
    plt.xticks([0, 1, 2], ['Hate Speech', 'Offensive Language', 'Neither'])
    st.pyplot(plt)

elif chart_type == 'Pie Chart':
    st.subheader('Proportion of Classes (Pie Chart)')
    labels = 'Hate Speech', 'Offensive Language', 'Neither'
    sizes = df['class'].value_counts()
    explode = (0, 0.1, 0)  # only "explode" the 2nd slice
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribution of Classes')
    st.pyplot(plt)

# Horizontal line separator
st.markdown("---")

st.subheader('Text Preprocessing')

# Create tabs
tab1, tab2 = st.tabs(["Tweets Before Preprocessing", "Tweets After Stemming"])

# Tab 1: Dataset Brief Information
with tab1:
    st.subheader('Tweets (Before Preprocessing)')
    st.write(df.tweet)
 
# Tab 2: Dataset Columns Description
with tab2:
    st.subheader('Tweets (After Stemming)')
    st.write(pd.read_csv('stemmed_tweets.csv'))
# Horizontal line separator
st.markdown("---")

