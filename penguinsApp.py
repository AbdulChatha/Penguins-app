from distutils.command.upload import upload
import streamlit as st
import pandas as pd
import pickle
st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!
""")
st.sidebar.header('User input features')
# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)
# uploadfile=st.sidebar.file_uploader('Upload input CSV file ',type=['csv'])
# if uploadfile is not None:
#     input_df=pd.read_csv(uploadfile)
# else:
def user_input():
    island=st.sidebar.selectbox('island',('Biscoe','Dream','Torgersen'))
    sex=st.sidebar.selectbox('Gender',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data={"island":island,
    'sex':sex,
    'bill_length_mm':bill_length_mm,
    'bill_depth_mm':bill_depth_mm,
    'flipper_length_mm':flipper_length_mm,
    'body_mass_g':body_mass_g}
    input_df=pd.DataFrame(data,index=[0])
    return input_df
input_df= user_input()
st.subheader('input features')
st.write(input_df)
Dataset=pd.read_csv('penguins_cleaned.csv')
x=Dataset.drop(['species'],axis=1)
x=pd.concat([x,input_df],axis=0)
x=pd.get_dummies(x)
target=Dataset.species
st.write('# Target Classes')
st.write(target.unique())
input=x.iloc[-1,:]
clf=pickle.load(open('penguins_clf.pkl','rb'))
prediction=clf.predict([input.T])
st.subheader('Prediction')
st.write(prediction)
prediction_proba=clf.predict_proba([input.T])
st.subheader('Prediction Probability')
st.write(prediction_proba)