import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import io
import streamlit as st
from streamlit_option_menu import option_menu

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Set Streamlit page title
#st.title('Hate Speech and Offensive Language Analysis')

# Create a vertical tab menu in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Hate Speech and Offensive Language Analysis",  # Title of the menu
        options=["Home", "Tweets Dataset", "Tweets Classes", "Tweets Preprocessing", "Model Selection", "About", "Contact"],  # Menu options
        icons=["house","","","","","info","envelope"],  # Optional icons
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
    # Horizontal line separator
    st.markdown("---")

elif selected == "Tweets Dataset":
    st.title("Loading and Previewing the Dataset")
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

elif selected == "Tweets Classes":
    st.title("Understanding Class Distribution")
    # Create tabs
    tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])

    # Tab 1: Dataset Brief Information
    with tab1:
        st.subheader('Distribution of Classes (Bar Chart)')
        plt.figure(figsize=(8, 6))
        plt.hist(df['class'], bins=3, edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Distribution of Classes')
        plt.xticks([0, 1, 2], ['Hate Speech', 'Offensive Language', 'Neither'])
        st.pyplot(plt)
        plt.clf()  # Clear the figure after using it to prevent overlap
 
    # Tab 2: Dataset Columns Description
    with tab2:
        st.subheader('Proportion of Classes (Pie Chart)')
        labels = ['Hate Speech', 'Offensive Language', 'Neither']
        sizes = df['class'].value_counts().reindex([0, 1, 2])  # Ensure correct order
        explode = (0, 0.1, 0)  # only "explode" the 2nd slice
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Distribution of Classes')
        st.pyplot(fig1)  # Pass the figure object instead of plt
        plt.clf()  # Clear the figure after using it

    # Horizontal line separator
    st.markdown("---")

elif selected == "Tweets Preprocessing":
    st.title("Dataset Preprocessing")

    st.write("""
    We used slight pre-processing to normalize the tweets content by:
    A) Delete the characters outlined here (— : , ;	! ?).
    B) Normalize hashtags into words, thus ’refugeesnotwelcome’ becomes ’refugees not welcome’.
       This is due to the fact that such hashtags are frequently employed when creating phrases. 
    C) We separate such hashtags using a dictionary-based lookup.
    D) To eliminate word inflections, use lowercase to remove capital letters and stemming to overcome the problem of several forms of words.
    E) Encode the tweets into integers and pad each tweet to the max length of 100 words.

    """)
    # Horizontal line separator
    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tweets Before Preprocessing", "Cleaned Tweets", "Stemmed Tweets", "Tokenized Tweets"])

    # Tab 1: Dataset Brief Information
    with tab1:
        st.subheader('Tweets Before Preprocessing')
        st.write(df.tweet)
        # Horizontal line separator
        st.markdown("---")

    # Tab 2: Dataset Columns Description
    with tab2:
        st.subheader('Tweets After Cleaning')
        st.write(pd.read_csv('cleaned_tweets.csv'))
        # Horizontal line separator
        st.markdown("---")

    # Tab 3: Dataset Columns Description
    with tab3:
        st.subheader('Tweets After Stemming')
        st.write(pd.read_csv('stemmed_tweets.csv'))
        # Horizontal line separator
        st.markdown("---")

    # Tab 3: Dataset Columns Description
    with tab4:
        st.subheader('Tweets After Tokenization')
        st.write(pd.read_csv('Tokenized_Padded_tweets.csv'))
        # Horizontal line separator
        st.markdown("---")

elif selected == "Model Selection":
    st.title("Model Selection")
    st.write("""
    (Classifier training and testing): Ten-fold cross-validation was used to train 
    and test all the six classifiers (logistic regression, decision tree, random forest, 
    naive Bayes, k-nearest neighbors, and support vector machines). We utilized 
    traditional machine learning methods provided by the Scikit-learn Python module 
    for classification. The Logistic Regression class uses L2 regularization with 
    a regularization parameter C equals 0.01. The hyper parameter used value of maximum depth 
    in decision trees and random forest equals 2. The hyper parameter used value of k in 
    k-nearest neighbors is 5, this means that the algorithm will consider the class or value of 
    the 5 nearest neighbors, when making predictions. In naive Bayes there are no specific default 
    values for this algorithm, as it does not require tuning hyper parameters. The hyper parameter 
    used value of C in SVM is 1.0.""")
    # Horizontal line separator
    st.markdown("---")
    tab1, tab2 = st.tabs(["Classification Results", "222"])
    # Tab 3: Dataset Columns Description
    with tab1:
        st.subheader('Table I. Classification Results')
        # Define the data for the table
        data = {
        'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                      'Naive Bayes', 'K-Nearest Neighbor', 'SVM - SVC'],
        'Precision': ['0.83 ± 0.04', '0.77 ± 0.06', '0.77 ± 0.06', '0.71 ± 0.07', '0.79 ± 0.05', '0.78 ± 0.05'],
        'Recall': ['0.96 ± 0.02', '1.00 ± 0.01', '1.00 ± 0.01', '0.96 ± 0.02', '0.90 ± 0.03', '1.00 ± 0.01'],
        'F1-Score': ['0.88 ± 0.02', '0.87 ± 0.03', '0.87 ± 0.03', '0.81 ± 0.04', '0.84 ± 0.04', '0.87 ± 0.03']
        }

        # Convert the data to a pandas DataFrame
        df_results = pd.DataFrame(data)

        # Display the table in Streamlit
        st.table(df_results)
        # Horizontal line separator
        st.markdown("---")

    # Tab 3: Dataset Columns Description
    with tab2:
        st.subheader('Tweets After Tokenization')
        # Data for the table
        data = {
            'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                          'Naive Bayes', 'K-Nearest Neighbor', 'SVM - SVC'],
            'Precision': [0.83, 0.77, 0.77, 0.71, 0.79, 0.78],
            'Recall': [0.96, 1.00, 1.00, 0.96, 0.90, 1.00],
            'F1-Score': [0.88, 0.87, 0.87, 0.81, 0.84, 0.87]
        }

        # Convert the data to a pandas DataFrame (renaming it df_fig)
        df_fig = pd.DataFrame(data)

        # Create a grouped bar chart using Plotly
        fig = go.Figure()

        # Add Precision bars
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['Precision'], name='Precision'))

        # Add Recall bars
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['Recall'], name='Recall'))

        # Add F1-Score bars
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['F1-Score'], name='F1-Score'))

        # Update layout for grouped bars
        fig.update_layout(
            title='Classification Results',
            xaxis_title='Algorithm',
            yaxis_title='Score',
            barmode='group',  # Group the bars side by side
            xaxis_tickangle=-45
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("---")
 
elif selected == "About":
    st.title("About")
    st.write("This is the about page.")
    # Horizontal line separator
    st.markdown("---")

elif selected == "Contact":
    st.title("Contact")
    st.write("This is the contact page.")
    # Horizontal line separator
    st.markdown("---")
