import streamlit as st


def displayMetrics(df):
    selectedYear = st.selectbox("Select Year", df["Year"].unique()[::-1])
    df_present = df[df["Year"] == selectedYear]
    df_prev = df[df["Year"] == (selectedYear - 1)]
    st.subheader(f"Year: {selectedYear}")
    col1, col2, col3, col4, col5 = st.columns(5)
    df_sum = df_present.sum()
    df_sum_prev = df_prev.sum()
    with col1:
        st.metric(
            label="Product 1 Sales",
            value=df_sum.values[5],
            delta=int(df_sum.values[5] - df_sum_prev.values[5])
        )
    with col2:
        st.metric(
            label="Product 2 Sales",
            value=df_sum.values[6],
            delta=int(df_sum.values[6] - df_sum_prev.values[6])
        )
    with col3:
        st.metric(
            label="Product 3 Sales",
            value=df_sum.values[7],
            delta=int(df_sum.values[7] - df_sum_prev.values[7])
        )
    with col4:
        st.metric(
            label="Product 4 Sales",
            value=df_sum.values[8],
            delta=int(df_sum.values[8] - df_sum_prev.values[8])
        )
    with col5:
        st.metric(
            label="Product 5 Sales",
            value=df_sum.values[9],
            delta=int(df_sum.values[9] - df_sum_prev.values[9])
        )
