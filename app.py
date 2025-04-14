#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open("rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Enter the **title** and **text** of a news article below to check if it's real or fake.")

title_input = st.text_input("News Title")
text_input = st.text_area("News Content")

# Prediction
if st.button("Predict"):
    if title_input and text_input:
        user_input = title_input + " " + text_input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        if prediction == 1:
            st.success("✅ This appears to be REAL news.")
        else:
            st.error("❌ This appears to be FAKE news.")
    else:
        st.warning("Please enter both a title and content.")


# In[ ]:




