import streamlit as st

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset

hideNavBar()

st.title("Search Your Dataset")

selectDataset()
