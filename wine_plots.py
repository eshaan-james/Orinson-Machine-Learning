import warnings
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import streamlit as st
import wine_models
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 



def app(wine_df):
    warnings.filterwarnings('ignore')
    st.title("Visualise the Wine Quality Prediction Web app ")

    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(wine_df.iloc[:, 1:].corr(),annot=True)  
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        st.pyplot()

    st.subheader("Predictor Selection")
    plot_select = st.selectbox("Select the Classifier to Visualise the Wine Quality Prediction:",
                               ('Decision Tree Classifier', "Random Forest Classifier", 'K Nearest Neighbor Classifier'))

    if plot_select == 'Decision Tree Classifier':
        X_train, X_test, y_train, y_test = wine_models.data_split(wine_df)

        dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        dtree_clf.fit(X_train, y_train)
        y_train_pred = dtree_clf.predict(X_train)
        y_test_pred = dtree_clf.predict(X_test)

        if st.checkbox("Plot confusion matrix"):
            cm = confusion_matrix(y_test, y_test)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            st.pyplot(fig)

        if st.checkbox("Plot Decision Tree"):
            # Export decision tree in dot format and store in 'dot_data' variable.
            feature_columns = X_train.columns

            dot_data = tree.export_graphviz(decision_tree=dtree_clf, max_depth=3, out_file=None, filled=True,
                                            rounded=True,
                                            feature_names=feature_columns, class_names=["3", '4', '5', '6', '7', '8'])
            # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module.
            st.graphviz_chart(dot_data)

    if plot_select == 'Random Forest Classifier':
        X_train, X_test, y_train, y_test = wine_models.data_split(wine_df)

        # Initialize the Random Forest Classifier with the specified best parameters
        rfc_clf = RandomForestClassifier(max_depth=10, n_estimators=75, random_state=42)
        rfc_clf.fit(X_train, y_train)

        # Predictions
        y_train_pred = rfc_clf.predict(X_train)
        y_test_pred = rfc_clf.predict(X_test)

        # Confusion Matrix Plot
        if st.checkbox("Plot confusion matrix"):
            cm = confusion_matrix(y_test, y_test_pred)  # Fixed to use y_test_pred
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            st.pyplot(fig)

        # Plot Decision Tree from the Random Forest
        if st.checkbox("Plot a Tree from Random Forest"):
            # Select one tree from the Random Forest
            tree_index = st.slider("Select a tree index (0-74)", min_value=0, max_value=74, value=0)
            selected_tree = rfc_clf.estimators_[tree_index]

            # Export decision tree in dot format and store in 'dot_data' variable
            feature_columns = X_train.columns  # Assumes X_train is a DataFrame with column names
            dot_data = tree.export_graphviz(
                decision_tree=selected_tree,
                max_depth=10,
                out_file=None,
                filled=True,
                rounded=True,
                feature_names=feature_columns,
                class_names=["3", "4", "5", "6", "7", "8"]
            )

            # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module
            st.graphviz_chart(dot_data)


    if plot_select == 'K Nearest Neighbor Classifier':
        X = wine_df.iloc[:, :-1].values  # Convert DataFrame to NumPy array
        y = wine_df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train, y_train)

        y_train_pred = knn_clf.predict(X_train)
        y_test_pred = knn_clf.predict(X_test)

        if st.checkbox("Plot confusion matrix"):
            cm = confusion_matrix(y_test, y_test_pred)  # Fixed y_test, y_test_pred
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            st.pyplot(fig)
