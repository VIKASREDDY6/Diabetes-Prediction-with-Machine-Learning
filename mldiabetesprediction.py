import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def main():
    feat_list=[]
    df=pd.read_csv('diabetes.csv')
    from sklearn.model_selection import train_test_split
    features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    pred_class=['Outcome']
    X=df[features].values
    y=df[pred_class].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=10)
    from sklearn.impute import SimpleImputer
    fill_vals=SimpleImputer(missing_values=0,strategy='mean')
    X_train=fill_vals.fit_transform(X_train)
    X_test=fill_vals.fit_transform(X_test)
    st.title('Diabetes Prediction using Machine Learning.')
    st.subheader("Enter your data below.")
    preg=st.slider("Number of Pregnancies:",0,20)
    feat_list.append(preg)
    glu=st.slider("Glucose level:",40,200)
    feat_list.append(glu)
    bp=st.slider("Blood Pressure",20,125)
    feat_list.append(bp)
    sk=st.slider("Skin Thickness:",10,90)
    feat_list.append(sk)
    ins=st.slider("Insulin level:",20,900)
    feat_list.append(ins)
    bmi=st.slider("BMI index:",10,70)
    feat_list.append(bmi)
    st.info('Multiply your Diabetes Pedigree Function with 1000')
    dpf=st.slider("Diabetes Pedigree Function:",0,2500)
    feat_list.append(dpf/1000)
    age=st.slider("Age:",10,110)
    feat_list.append(age)
    feat_dict={}
    for i in range(0,len(features)):
        feat_dict[features[i]]=feat_list[i]
    st.text("Your Data is:")
    st.write(feat_dict)
    #model selection
    selected_model=st.selectbox("Select ML Model:",['SVM','DecisionTree','RandomForest','NaiveBayes'])
    #SVM
    if selected_model=='SVM':
        svmmodel=svm.SVC(kernel='linear')
        svmmodel.fit(X_train,y_train.ravel())
        if st.button("Get Prediction"):
            op=svmmodel.predict([feat_list])
            if op[0]==0:
                st.info("Hurray!You dont have Diabetes.")
            else:
                st.warning("Sorry.You have Diabetes.")
            #Metrics
            st.subheader("Model Metrics.")
            pred_test_svm=svmmodel.predict(X_test)
            st.write("Accuracy: ",metrics.accuracy_score(y_test,pred_test_svm))
            st.write("Recall: ",metrics.recall_score(y_test,pred_test_svm))
            st.write("Precision: ",metrics.precision_score(y_test,pred_test_svm))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_test_svm))
            st.subheader("Classification Report:")
            st.info(metrics.classification_report(y_test,pred_test_svm))
                
    #DecisionTree
    if selected_model=='DecisionTree':
        dtmodel=tree.DecisionTreeClassifier()
        dtmodel.fit(X_train,y_train.ravel())
        if st.button("Get Prediction"):
            op=dtmodel.predict([feat_list])
            if op[0]==0:
                st.info("Hurray!You dont have Diabetes.")
            else:
                st.warning("Sorry.You have Diabetes.")
                
            #Metrics
            st.subheader("Model Metrics.")
            pred_test_dt=dtmodel.predict(X_test)
            st.write("Accuracy: ",metrics.accuracy_score(y_test,pred_test_dt))
            st.write("Recall: ",metrics.recall_score(y_test,pred_test_dt))
            st.write("Precision: ",metrics.precision_score(y_test,pred_test_dt))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_test_dt))
            st.subheader("Classification Report:")
            st.info(metrics.classification_report(y_test,pred_test_dt))
                
    #RandomForest
    if selected_model=='RandomForest':
        rand_for_model=RandomForestClassifier(random_state=10)
        rand_for_model.fit(X_train,y_train.ravel())
        if st.button("Get Prediction"):
            op=rand_for_model.predict([feat_list])
            if op[0]==0:
                st.info("Hurray!You dont have Diabetes.")
            else:
                st.warning("Sorry.You have Diabetes.")
                
            #Metrics
            st.subheader("Model Metrics.")
            pred_test_rf=rand_for_model.predict(X_test)
            st.write("Accuracy: ",metrics.accuracy_score(y_test,pred_test_rf))
            st.write("Recall: ",metrics.recall_score(y_test,pred_test_rf))
            st.write("Precision: ",metrics.precision_score(y_test,pred_test_rf))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_test_rf))
            st.subheader("Classification Report:")
            st.info(metrics.classification_report(y_test,pred_test_rf))
                
    #NaiveBayes
    if selected_model=='NaiveBayes':
        nbmodel=GaussianNB()
        nbmodel.fit(X_train,y_train.ravel())
        if st.button("Get Prediction"):
            op=nbmodel.predict([feat_list])
            if op[0]==0:
                st.info("Hurray!You dont have Diabetes.")
            else:
                st.warning("Sorry.You have Diabetes.")
                
            #Metrics
            st.subheader("Model Metrics.")
            pred_test_nb=nbmodel.predict(X_test)
            st.write("Accuracy: ",metrics.accuracy_score(y_test,pred_test_nb))
            st.write("Recall: ",metrics.recall_score(y_test,pred_test_nb))
            st.write("Precision: ",metrics.precision_score(y_test,pred_test_nb))
            st.subheader("Confusion matrix(Actual VS Predicted):")
            st.write(metrics.confusion_matrix(y_test,pred_test_nb))
            st.subheader("Classification Report:")
            st.info(metrics.classification_report(y_test,pred_test_nb))
            
    if st.button("See who created this!"):
        st.info("Name: K. Vikas Reddy")
        st.info("College: SASTRA Deemed to be University")
        st.info("Gmail: reddyvikas995@gmail.com")
        
    st.warning("Please report any bugs and suggestions if any.")
        
    if st.checkbox("About this Project"):
        st.write("Diabetes mellitus (DM), commonly known as diabetes, is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period of time.")
        st.write("Symptoms often include frequent urination, increased thirst, and increased appetite. If left untreated, diabetes can cause many complications.")
        st.write("Diabetes is very common disorder in the world. So, identifying its existence in the body in its early stages is very important to reduce its growing effect.")
        st.write("This Project uses Machine Learning Techniques to predict probable diabetic condition using your required medical data")
        st.subheader("Note:")
        st.write("Please note that the prediction accuracy of all the techniques used is less than 80%.It is because of the constraint in the available data(Not huge amount of data is  available) and the prediction is based only on the available data.")
        st.write("For same data, all models may not give same prediction.")
        st.write("Classification Report's columns are misalligned. Precision, Recall, F-1 Score and Support are four columns.")
                
    
if __name__=='__main__':
    main()
