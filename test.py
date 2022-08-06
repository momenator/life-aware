import streamlit as st
import pandas as pd
from io import StringIO
import time
from autofeat import AutoFeatRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import streamlit.components.v1 as components

# Helper functions


def get_formula(model, threshold=0.0001, pre=4):
    """Get formula from the model"""
    formula = "{} ".format(round(model.prediction_model_.intercept_, pre))
    for c, v in zip(model.good_cols_, model.prediction_model_.coef_):
        if v >= threshold:
            formula = formula + "+" + str(round(v, pre)) + "*"+c
    return formula


def get_score(model, X, y):
    """Get the model score"""
    return round(model.score(X, y), 3)


def handle_dates(df):
    """Handle dates in the dataframe"""
    date_fields = []
    for index, value in df.dtypes.items():
        if value == "datetime64[ns]":
            df["{} relative".format(index)] = df["{}".format(index)].apply(
                lambda x: (pd.Timestamp.now() - x).days)
            date_fields.append(index)

    df.drop(date_fields, axis=1, inplace=True)

    return date_fields


# Main file
st.image("https://i.imgur.com/Ej1CCGd.png")
uploaded_file = st.sidebar.file_uploader("Upload a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    if 'csv' in uploaded_file.name:
        dataframe = pd.read_csv(uploaded_file)
    elif 'xlsx' in uploaded_file.name:
        dataframe = pd.read_excel(uploaded_file)

    st.write("Your file")
    st.write(dataframe.head())

    feat_cols = st.sidebar.multiselect(
        'select feature columns',
        dataframe.columns
    )

    targ_col = st.sidebar.selectbox(
        'select target column',
        dataframe.columns)

    if targ_col and targ_col in feat_cols:
        st.error('target column and feature columns cannot overlap!')
    else:
        # Precessing

        # Handling dates
        df_copy = dataframe.copy()
        date_fields = handle_dates(dataframe)

        y = dataframe[targ_col]
        dataframe = dataframe.drop([targ_col], axis=1)

        feat_cols = list(dataframe.columns)
        X = dataframe[feat_cols]

        complexity = st.sidebar.slider(
            'model complexity (1 = simplest, 4 = very complex)', 1, 4, 2)

        # run your ML model here
        if st.sidebar.button('Run Formula Extractor'):

            model = AutoFeatRegressor(verbose=0, feateng_steps=complexity)
            df_transform = model.fit_transform(X, y)

            formula = get_formula(model)
            score = get_score(model, X, y)

            st.subheader("Your Results", anchor=None)

            st.write("Transformed dataframe")
            st.write(df_transform.head())

            result = formula
            st.write('Confidence: %s' % score)
            # display results here
            txt = st.latex(result)

        if st.sidebar.button('Run Tree Regressor'):
            # the tree stuff here
            regressor = DecisionTreeRegressor(max_depth=5)
            regressor.fit(X, y)

            reg_score = get_score(regressor, X, y)

            st.write(f'Regressor score {reg_score}')

            # tree viz - mockup
            st.write("Tree Visualisation")
            st.image(
                "https://mljar.com/blog/visualize-decision-tree/output_31_0.svg")
