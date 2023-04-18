import pandas as pd
import streamlit as st

from MyUtils.Firebase import get_firestore_files



def selectDataset():

    files = get_firestore_files("")
    search_files = {}
    for doc in files:
        # print(f'{doc.id} => {doc.to_dict()}')
        search_files[doc.to_dict()['file_name']] = doc.to_dict()['file_url']
    file = st.selectbox('Select your Dataset',
                        search_files.keys())

    if file:
        st.write('You selected `%s`' % file)
        df = pd.read_csv(search_files[file], index_col=None, encoding_errors='replace')
        #st.dataframe(df.head(5))
        return df
def selectDataset_with_msg(select_msg):

    files = get_firestore_files("")
    search_files = {}
    for doc in files:
        # print(f'{doc.id} => {doc.to_dict()}')
        search_files[doc.to_dict()['file_name']] = doc.to_dict()['file_url']
    file = st.selectbox(select_msg,
                        search_files.keys())

    if file:
        st.write('You selected `%s`' % file)
        df = pd.read_csv(search_files[file], index_col=None, encoding_errors='replace')
        #st.dataframe(df.head(5))
        return df
