import pandas as pd
import streamlit as st

from MyUtils.Firebase import upload_file_to_firestore
from MyUtils.HideStDefaults import hideNavBar

hideNavBar()


st.title("Search Your Dataset")
file = st.file_uploader("Upload Your Dataset", type=["csv"])

if file:
    with st.spinner(text="Upload in progress..."):
        upload_url = upload_file_to_firestore(file)
    if upload_url:
        st.success("File Uploaded Successfully")
        df = pd.read_csv(upload_url, index_col=None, encoding_errors='replace')
        st.dataframe(df.head(5))
    else:
        st.error("File Upload Failed")
