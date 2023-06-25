import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
x_train,x_test,y_train,y_test=train_test_split(X,y)
model = LinearRegression()
model.fit(x_train,y_train)


string = "Startup’s Profit Prediction"
st.set_page_config(page_title=string, page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(string, anchor=None)

from PIL import Image
image = Image.open('startup.jpg')
st.image(image)

rds = st.number_input('Insert a R&D Spend')
st.write('The current number is ', rds)

ads = st.number_input('Insert a Administration Spend')
st.write('The current number is ', ads)

mks = st.number_input('Insert a Marketing Spend')
st.write('The current number is ', mks)


y_pred = model.predict([[rds, ads, mks, 1]])

if st.button('PREDICT'):
    st.success(y_pred)
else:
    st.write('Fill in important details!')

