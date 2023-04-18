
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import numpy as np
#import hydralit_components as hc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from MyUtils.HideStDefaults import hideNavBar
from MyUtils.searchAndSelectFile import selectDataset_with_msg
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
#from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from MyUtils.HideStDefaults import hideNavBar
from MyUtils.Metrics import displayMetrics
from MyUtils.searchAndSelectFile import selectDataset
from sklearn.model_selection import cross_validate

hideNavBar()


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

#df_train=pd.read_csv('training_data_insights.csv')
#df_test=pd.read_csv('test_data_insights.csv')

col33, col45, col34 = st.columns(3,gap="small")
with col33:
    chosen_target_X_continous = st.multiselect(label="Choose Continuous Independant  variable", options=df_train.columns)
with col45:
    chosen_target_X_categorical = st.multiselect(label="Choose Categorical Independant  variable", options=df_train.columns)
with col34:
    chosen_target_Y = st.selectbox(label="Choose Dependant  variable",
                                   options=(df_train.columns).insert(0, "Choose an option"))

    

if chosen_target_Y != 'Choose an option':
    #Model training
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

    df_churn=pd.concat((df_test[['CustomerID']],X_test),axis=1)
    df_churn['churn_probability']=np.nan
    df_churn['churn_probability']=churn_probability_predictions
    
    #Metrics
    metrics = cross_validate(
    estimator=model_churn, X=X_train.values, y=Y_train.values.ravel(), cv=5, 
    scoring=['accuracy','precision', 'recall','f1'])
    #tile A
    accuracy=round((metrics['test_accuracy'].mean())*100,2)
    #tile B
    precision=round((metrics['test_precision'].mean())*100,2)
    #tile 3
    recall=round((metrics['test_recall'].mean())*100,2)
    #tile 4
    f1=round((metrics['test_f1'].mean())*100,2)
    col54, col55, col56, col57 = st.columns(4,gap="small") 

    with col54:
        st.metric(
        label="Accuracy",
        value=accuracy,
    )

    with col55:
        st.metric(
        label="Precision",
        value=precision,   
    )
        
    with col56:
        st.metric(
             label="Recall",
             value=recall,
    )
    with col57:
        st.metric(
             label="F1",
             value=f1,
    )     

    
    
    
    #Tile  1 - Total  active customers
    total_active_customers=len(df_test.index)
    #Tile 2 - Customers at the risk of churn
    df_risky_customers=(df_churn[df_churn['churn_probability']>0.50])
    total_risky_customers=len(df_risky_customers.index)
    #Tile 3  Overall loss
    overall_annual_loss=round(((df_risky_customers['Monthly Charges'].sum())*12),2)
    col4, col5, col6, = st.columns(3,gap="small")

    with col4:
        st.metric(
        label="Total Active Customers",
        value=total_active_customers,
        
    )

    with col5:
        st.metric(
        label="Potential Customer Churn",
        value=total_risky_customers,
        
    )

    with col6:
        st.metric(
        label="Potential Revenue Loss",
        value=overall_annual_loss,
        
    )
    #Filtering customers with churn rate higher than 0.5
    df_pareto = df_churn[df_churn['churn_probability'] > 0.5]

    #Sorting the dataaccording to monthly charges
    df_pareto=(df_pareto.sort_values(by='Monthly Charges' , ascending=False).reset_index(drop=True)).copy()
    #Calculating Prospective annual revenue loss
    df_pareto['Annual Revenue Loss']= df_pareto['Monthly Charges']*12

    #Cumulative values
    cumulative_sum=df_pareto['Annual Revenue Loss'].cumsum()
    total = df_pareto['Annual Revenue Loss'].sum()
    percentage = cumulative_sum / total * 100

    #Visual representaion of revenue loss with customer id
    trace1 = go.Bar(
        x=df_pareto['CustomerID'],
        y=df_pareto['Annual Revenue Loss'],
        marker=dict(
            color='rgb(178,24,43)'
                ),
        name='Potential Annual Loss'
    )
    trace2 = go.Scatter(
        x=df_pareto['CustomerID'],
        y=percentage,
        marker=dict(
        color='rgb(255,255,0)'
                ),
        name='Cumulative Percentage',
        yaxis='y2'

    )

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(trace1)
    fig_pareto.add_trace(trace2,secondary_y=True)
    fig_pareto['layout'].update(height = 450, width = 1300, title = "Pareto analysis of Annual Loss",xaxis=dict(tickangle=-90,showgrid=False),template="plotly_white", yaxis=dict(showgrid=False),title_x=0.4)
    st.plotly_chart(fig_pareto)

    #Graph 1 -  Overall feature importance
    col21, col22 = st.columns(2,gap="small")
    with col21:

        fi = permutation_importance(model_churn,X_train.values,Y_train.values.ravel(), n_repeats=1,random_state=0)

        r_list=list(fi.importances_mean)
        X_list=list(X_train.columns)


        fi=[X_list,r_list]
        df_fi=pd.DataFrame (fi).transpose()
        df_fi.columns = ['Feature', 'Importance']
        df_fi=(df_fi.sort_values(by='Importance',ascending=False).reset_index(drop=True)).copy()

        fig_fi = px.bar(df_fi,
                            y="Importance", x="Feature",
                            template="plotly_white", title="Reasons for Churn",labels={"Feature":"Feature","Importance":"Importance"},width=500, height=300,
                            color_discrete_sequence=['#B2182B'], orientation='v')
        fig_fi.update_layout( xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),title_x=0.4)
        st.plotly_chart(fig_fi)

    with col22:
            #Top & bottom 3 influencing parameters 
        first_influencing_parameter=(df_fi.iloc[[0],[0]].values)[0][0]
        second_influencing_parameter=(df_fi.iloc[[1],[0]].values)[0][0]
        third_influencing_parameter=(df_fi.iloc[[2],[0]].values)[0][0]

        least_influencing_parameter=(df_fi.iloc[[-1],[0]].values)[0][0]
        secondleast_influencing_parameter=(df_fi.iloc[[-2],[0]].values)[0][0]
        thirdleast_influencing_parameter=(df_fi.iloc[[-3],[0]].values)[0][0]  
        col28, col29 = st.columns(2,gap="small")
        with col28:
            
            st.subheader(":green[Top Churn Triggers]")
            #hc.info_card(title='Some heading GOOD', content=first_influencing_parameter, sentiment='good')
            st.text(first_influencing_parameter)
            st.text(second_influencing_parameter)
            st.text(third_influencing_parameter)
            
        with col29:
            st.subheader(":red[Bottom Churn Triggers]")
            st.text(least_influencing_parameter)
            st.text(secondleast_influencing_parameter)
            st.text(thirdleast_influencing_parameter)
            
            

    col25, col26 = st.columns(2,gap="small")
    with col25:
        col23,col24=st.columns(2,gap='small')
        #st.text("")
        #st.text("")
        #st.text("")
        #st.text("")
        #st.text("")
        with col23:
            query_variable=st.selectbox(label="Group Category", options=df_train.columns.insert(0, "Choose an option"))
            if query_variable!="Choose an option":
                with col24:
                    query_variable_value=st.selectbox(label="Category Value", 
                                                    options=df_train[query_variable].unique())
                #Querying the dataset
                if query_variable==df_train.columns[0]:
                    df_queried=df_train.copy()
                else:    
                    condition = (df_train[query_variable] == query_variable_value) 
                    df_queried=(df_train[condition]).copy().reset_index(drop=True)

                    #loading the input data into the variables
                    #Preparing the independant variables after one hot encoding
                    X_q=pd.concat([df_queried[chosen_target_X_continous],df_queried[chosen_target_X_categorical]],axis=1)
                    X_q=pd.get_dummies(X_q, columns =chosen_target_X_categorical,drop_first='True')
                    Y_q=df_queried[chosen_target_Y]
                    model_churn.fit(X_q.values,Y_q.values.ravel())

                    fi_q = permutation_importance(model_churn,X_q.values,Y_q.values.ravel(), n_repeats=1,random_state=0)

                    r_list_q=list(fi_q.importances_mean)
                    X_list_q=list(X_q.columns)


                    fi_q=[X_list_q,r_list_q]
                    df_fi_q=pd.DataFrame (fi_q).transpose()
                    df_fi_q.columns = ['Feature', 'Importance']
                    df_fi_q=(df_fi_q.sort_values(by='Importance',ascending=False).reset_index(drop=True)).copy()

                    fig_fi_q = px.bar(df_fi_q,
                                        y="Importance", x="Feature",
                                        template="plotly_white", title="Reasons for Chrun - Group Specific",labels={"Feature":"Feature","Importance":"Importance"},width=500, height=300,
                                        color_discrete_sequence=['#B2182B'], orientation='v')
                    fig_fi_q.update_layout(title_x=0.2, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                    st.plotly_chart(fig_fi_q)

                    with col26:
                            #Top & bottom 3 influencing parameters 
                        first_influencing_parameter=(df_fi_q.iloc[[0],[0]].values)[0][0]
                        second_influencing_parameter=(df_fi_q.iloc[[1],[0]].values)[0][0]
                        third_influencing_parameter=(df_fi_q.iloc[[2],[0]].values)[0][0]

                        least_influencing_parameter=(df_fi_q.iloc[[-1],[0]].values)[0][0]
                        secondleast_influencing_parameter=(df_fi_q.iloc[[-2],[0]].values)[0][0]
                        thirdleast_influencing_parameter=(df_fi_q.iloc[[-3],[0]].values)[0][0]  
                        col35, col36 = st.columns(2,gap="small")
                        with col35:
                            
                            st.subheader(":green[Top Churn Triggers - Group specific]")
                            st.text(first_influencing_parameter)
                            st.text(second_influencing_parameter)
                            st.text(third_influencing_parameter)
                            
                        with col36:
                            st.subheader(":red[Bottom Churn Triggers - Group Specific]")
                            st.text(least_influencing_parameter)
                            st.text(secondleast_influencing_parameter)
                            st.text(thirdleast_influencing_parameter)
            









    
   
