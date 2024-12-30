import numpy as np
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 


@st.cache_data()
def data_split(wine_df):
    X = wine_df.iloc[:, :-1]
    y = wine_df.quality

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    return X_train, X_test, y_train, y_test

def d_tree_pred(wine_df, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol ):
    X_train, X_test, y_train, y_test = data_split(wine_df)

    dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree_clf.fit(X_train, y_train)

    y_train_pred = dtree_clf.predict(X_train)
    y_test_pred = dtree_clf.predict(X_test)

    # Classifify wine quality using the 'predict()' function.
    prediction = dtree_clf.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    prediction = prediction[0]
    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)
    return prediction, score

def knn_pred(wine_df, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol ):
    X_train, X_test, y_train, y_test = data_split(wine_df)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    y_train_pred = knn_clf.predict(X_train)
    y_test_pred = knn_clf.predict(X_test)

    # Classifify wine quality using the 'predict()' function.
    prediction = knn_clf.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    prediction = prediction[0]
    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)
    return prediction, score

def rfc_pred(wine_df, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol ):
    X_train, X_test, y_train, y_test = data_split(wine_df)

    rfc_clf = RandomForestClassifier(random_state= 42)
    params = {
    'max_depth': range(10 , 60 , 10),
    'n_estimators': range(25 , 100 , 25)
    }
    rfc_clf = GridSearchCV(rfc_clf,param_grid= params,cv= 5,n_jobs= -1,verbose=1)
    rfc_clf.fit(X_train, y_train)

    y_train_pred = rfc_clf.predict(X_train)
    y_test_pred = rfc_clf.predict(X_test)

    # Classifify wine quality using the 'predict()' function.
    prediction = rfc_clf.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    prediction = prediction[0]
    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)
    return prediction, score

def app(wine_df):
    st.markdown(
        "<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier, K nearest Classifier and Random Forest Classifier</b> for Wine Quality Classification.",
        unsafe_allow_html=True)
    
    st.subheader("Select Values:")

    fixed_acidity = st.slider('Select Fixed Acidity values', float(wine_df['fixed acidity'].min()), float(wine_df['fixed acidity'].max()), 0.1)

    volatile_acidity = st.slider('Select Volatile Acidity Value', float(wine_df['volatile acidity'].min()),float(wine_df['volatile acidity'].max()), 0.1)

    citric_acid = st.slider('Select Citric Acid Value', float(wine_df['citric acid'].min()), float(wine_df['citric acid'].max()),  )

    residual_sugar = st.slider('Select Residual Sugar Value', float(wine_df['residual sugar'].min()), float(wine_df['residual sugar'].max()), 0.1)

    chlorides = st.slider('Select Chlorides Value', float(wine_df['chlorides'].min()), float(wine_df['chlorides'].max()), 0.1)
    
    free_sulfur_dioxide = st.slider('Select Free Sulfur Dioxide Value', float(wine_df['free sulfur dioxide'].min()), float(wine_df['free sulfur dioxide'].max()), 0.1)

    total_sulfur_dioxide = st.slider('Select Total Sulfur Dioxide Value', float(wine_df['total sulfur dioxide'].min()), float(wine_df['total sulfur dioxide'].max()), 0.1)

    density = st.slider('Select density Value', float(wine_df['density'].min()), float(wine_df['density'].max()), 0.1)

    pH = st.slider('Select pH Value', float(wine_df['pH'].min()), float(wine_df['pH'].max()), 0.1)

    sulphates = st.slider('Select Sulphates Value', float(wine_df['sulphates'].min()), float(wine_df['sulphates'].max()), 0.1)

    alcohol = st.slider('Select Alcohol Value', float(wine_df['alcohol'].min()), float(wine_df['alcohol'].max()), 0.1)

    st.subheader("Model Selection")

    predictor = st.selectbox("Select the Decision Tree Classifier", ('Decision Tree Classifier', 'K Nearest Neighbors', 'Random Forest Classifier'))

    if predictor == 'Decision Tree Classifier':
        prediction, score = d_tree_pred(wine_df, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
        if st.button("Predict"):
           
           st.write(f"Quality of wine: {prediction}")
           st.write(f"The accuracy score of Decision Tree is {score}%""")

    elif predictor == 'K Nearest Neighbors':
        prediction, score = knn_pred(wine_df, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
        if st.button("Predict"):
           
           st.write(f"Quality of wine: {prediction}")
           st.write(f"The accuracy score of KNN model is {score}%""")
    elif predictor == 'Random Forest Classifier':
        prediction, score = rfc_pred(wine_df, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
        if st.button("Predict"):
           
           st.write(f"Quality of wine: {prediction}")
           st.write(f"The accuracy score of RFC model is {score}%""")
