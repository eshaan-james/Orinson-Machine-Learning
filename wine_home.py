import streamlit as st

def app(wine_df):
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