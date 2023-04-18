import streamlit as st


def hideNavBar():
    st.set_page_config(initial_sidebar_state="collapsed", page_title="Metriverse", layout="wide", menu_items=None)

    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="collapsedControl"] {visibility: hidden;}
    div[data-testid="metric-container"] {
   background-color: rgba(216, 198, 188, 1);
   border: 2px solid rgba(32, 48, 84, 1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   overflow-wrap: break-word;
   text-align: center
}

div.plot-container.plotly {
   border: 2px solid rgba(32, 48, 84, 1);
   border-radius: 4px;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
}
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] {
   grid-template-columns: auto !important;
   }
    </style> """, unsafe_allow_html=True)
