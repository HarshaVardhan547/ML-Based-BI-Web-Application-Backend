import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import plotly.express as px
from lime import lime_tabular


from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset_with_msg

hideNavBar()        
#df=pd.read_csv('train_data.csv')
#df_test=pd.read_csv('test data.csv')

st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(216, 198, 188, 1);
   border: 2px solid rgba(32, 48, 84, 1);
   border-radius: 5px;
}
</style>
"""
, unsafe_allow_html=True)

col31, col32 = st.columns(2,gap="small")
with col31: 
    df_train = selectDataset_with_msg("Select your Training dataset")
with col32:
    df_test = selectDataset_with_msg("Select your Test dataset")

col33, col45, col34, col40 = st.columns(4,gap="small")
with col33:
    chosen_target_X_continous = st.multiselect(label="Choose Continuous Independant  variable", options=df_train.columns)
with col45:
    chosen_target_X_categorical = st.multiselect(label="Choose Categorical Independant  variable", options=df_train.columns)
with col34:
    chosen_target_Y = st.selectbox(label="Choose Dependant  variable",
                                   options=(df_train.columns).insert(0, "Choose an option"))
with col40:
    if chosen_target_Y != 'Choose an option':
        customer_id = st.selectbox(label="Choose Customer ID",options=(df_test.CustomerID.unique()))

if chosen_target_Y != 'Choose an option':
    #input  parameters
    #input  parameters from user
    chosen_target_Y=['Churn Value']
    chosen_target_X_continous=['Tenure Months','Monthly Charges']
    chosen_target_X_categorical=['Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security',
                                'Online Backup','Device Protection']

    #Define the ensemble model
    model_churn=VotingClassifier(estimators=[('LR', LogisticRegression(max_iter=1000)), ('RF', RandomForestClassifier()), ('GB', GradientBoostingClassifier())],
                            voting='soft')
    #Preparing the independant variables after one hot encoding
    X_train=pd.concat([df_train[chosen_target_X_continous],df_train[chosen_target_X_categorical]],axis=1)
    X_train=pd.get_dummies(X_train, columns =chosen_target_X_categorical,drop_first='True')

    #Dependant variable
    Y_train=df_train[chosen_target_Y]

    #Training the model
    model_churn.fit(X_train.values,Y_train.values.ravel())

    #Model prediction
    #Preparing the independant variables after one hot encoding
    X_test=pd.concat([df_test[chosen_target_X_continous],df_test[chosen_target_X_categorical]],axis=1)
    X_test=pd.get_dummies(X_test, columns=chosen_target_X_categorical,drop_first='True')
    churn_probability_predictions=model_churn.predict_proba(X_test.values)
    churn_probability_predictions=churn_probability_predictions[:,1]

    df_churn=pd.concat((df_test[['CustomerID','City']],X_test),axis=1)
    df_churn['churn_probability']=np.nan
    df_churn['churn_probability']=churn_probability_predictions



    #Filtering  dataframe based on customer ID
    
    condition = (df_churn['CustomerID'] == customer_id) 
    df_customer=(df_churn[condition]).copy().reset_index(drop=True)

    #tile 1 - Tenure
    tenure=(df_customer['Tenure Months'].values)[0]
    #tile 2 - City
    city=(df_customer['City'].values)[0]
    #tile 3 - Annual Revenue
    annual_revenue=round(((df_customer['Monthly Charges'].values)[0])*12,2)
    

    col4, col5, col6, = st.columns(3,gap="large")

    with col4:
        st.metric(
        label="Customer Tenure (Months)",
        value=tenure,
        
    )

    with col5:
        st.metric(
        label="Customer City",
        value=city,
        
    )

    with col6:
        st.metric(
        label="Annual Revenue from Customer",
        value=annual_revenue,
        
    )
        
    
    col7, col8, col9, = st.columns(3,gap="small")

    with col7:
        #Churn Probability
        churn_prob=(df_customer['churn_probability'].values)[0]
        no_churn_prob=(1-churn_prob)

        column=['Churn','Will Not Churn']
        prob=[churn_prob,no_churn_prob]

        prob_dict={'Outcome':column,'Probability':prob}
        df_prob=pd.DataFrame(prob_dict)

        fig_prob= px.pie(df_prob,
                            names='Outcome', values='Probability',
                            template="plotly", title="Customer Churn Risk",width=400, height=400,color='Outcome',color_discrete_sequence=px.colors.sequential.RdBu)
        fig_prob.update_layout( xaxis_title="Factors", yaxis_title="Percentage Contribution to Churn Probability",xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),title_x=0.3)
        st.plotly_chart(fig_prob) 

    with col9:
        #LIME
        categorical_features = [i for i, col in enumerate(X_train)
                                if ((X_train.iloc[[0],[i]].values)[0][0]) <= 1] 

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            categorical_features=categorical_features,
            mode='classification'
        )

        exp = explainer.explain_instance(
            data_row=(df_customer[X_test.columns]).values[0], 
            predict_fn=model_churn.predict_proba,num_features=5
        )

        df_weights= pd.DataFrame(exp.as_list())
        df_weights= pd.DataFrame(exp.as_list())
        df_weights.rename( columns={0:'Factor',1:'Influence'}, inplace=True)

        #Visual representaion of factors
        fig= px.bar(df_weights,
                            x='Factor', y='Influence',
                            template="plotly", title="Impact on Customer Churn",width=400, height=400,color_discrete_sequence=['#B2182B'])
        fig.update_layout( xaxis_title="Factors", yaxis_title="Percentage Contribution to Churn Probability",xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),title_x=0.2)
        st.plotly_chart(fig) 

    with col8:
        #Creating rows for upcoming months
        df_forecast=df_customer.copy()
        df_forecast = pd.concat([df_forecast]*6, ignore_index=True).copy()
        
        #Updating the tenure months column
        for i in (df_forecast.index):
            df_forecast.loc[[i],["Tenure Months"]]=(df_forecast.loc[[i],["Tenure Months"]]) + i

        #Predicting churn forecast
        df_forecast['churn_pobability']=np.nan
        df_forecast['churn_pobability']=model_churn.predict_proba(df_forecast[X_test.columns].values)[:,1]

        #Forecast graph
        fig_tenure_churn = px.line(df_forecast,
                            x=df_forecast.index, y="churn_pobability",
                            template="plotly", title="Churn Risk Forecast",width=400, height=400,
                            color_discrete_sequence=['#B2182B'])
        
        fig_tenure_churn.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),title_x=0.3,xaxis_title="Time to Churn(Months)",yaxis_title="Churn Probability")                 
        st.plotly_chart(fig_tenure_churn) 
        








############################################################################
