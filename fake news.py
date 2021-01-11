import pandas as pd
import numpy as np
import joblib 
import seaborn as sns
import streamlit as st


# loading joblib files
naive = joblib.load("naive.joblib")
vectorizer = joblib.load("vectoriszer.joblib")


st.balloons()
st.title("Fake News Detection")

news_text = st.text_area("paste the news here:", height=300)
submit_button = st.button("CHECK")

if submit_button:
    if news_text:
        vect = vectorizer.transform([news_text])
        prediction = naive.predict(vect)
        st.warning(prediction)
    else:
        st.warning("Enter a text")