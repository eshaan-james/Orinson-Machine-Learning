import streamlit as st

def app(wine_df):
    st.write("Welcome to the Red Wine Classification App!This interactive platform leverages machine learning models like Decision Tree, KNN, and Random Forest to predict the quality of red wine based on its physicochemical properties. Explore the dataset, train models, visualize results, and make predictions—all in one place!
    Navigate through the app to uncover insights and understand what makes a wine exceptional. Cheers to data science! 🍷")
    with st.expander('View Database'):
        st.dataframe(wine_df)
    st.subheader('Columns description')
    col_name, col_dtype, col_display = st.columns(3)
    if col_name.checkbox('Show all column names'):
        st.table(wine_df.columns)
    if col_dtype.checkbox('View column datatype'):
        st.write(list(wine_df.dtypes))
    if col_display.checkbox('View Column data'):
        col = st.selectbox('Select columns', wine_df.columns)
        st.table(wine_df[col])
