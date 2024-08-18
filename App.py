import streamlit as st
import pandas as pd

st.title("Web-based Data Mining and Analysis Application")

# Upload Data
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, sep='\t')
    df['Label'] = range(1, len(df) + 1)
    st.write("Data Loaded:")
    st.dataframe(df)

