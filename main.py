import streamlit as st
import pandas as pd
import wine_home
import wine_models
import wine_plots
# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title='Red Wine Classification',
                   page_icon='random',
                   layout='wide',
                   initial_sidebar_state='auto'
                   )


# Loading the dataset.
@st.cache_data()
def load_data():
    df = pd.read_csv("winequality-red.csv")
    return df

wine_df = load_data()

st.title('Red Wine Classification App')
pages_dict = {"Home": wine_home,
              "Classification": wine_models,
              "Plots": wine_plots}

st.sidebar.title('Navigation')
user_choice = st.sidebar.radio('Go To', tuple(pages_dict.keys()))
selected_page = pages_dict[user_choice]
selected_page.app(wine_df)