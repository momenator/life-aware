import streamlit as st
import pandas as pd
from io import StringIO
import time


uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    if 'csv' in uploaded_file.name:
        dataframe = pd.read_csv(uploaded_file)
    elif 'xlsx' in uploaded_file.name:
        dataframe = pd.read_excel(uploaded_file)

    st.write(dataframe)

    feat_cols = st.multiselect(
        'select feature columns',
        dataframe.columns
    )

    st.write('Features:', feat_cols)

    targ_col = st.selectbox(
        'select target column',
        dataframe.columns)

    if targ_col and targ_col in feat_cols:
        st.error('target column and feature columns cannot overlap!')
    else:
        # run your ML model here
        if st.button('Run'):
            result = None

            # display results here
            txt = st.text_area('extracted rules', 'your rules here...')

            st.write('result: %s' % result)
