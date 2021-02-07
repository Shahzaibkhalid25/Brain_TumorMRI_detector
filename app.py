#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification


# In[2]:


st.title("Brain Tumor MRI Classification")
st.header("Upload a brain MRI Image for image classification as tumor or no-tumor")
st.text("A prototype AI model created by Shahzaib Khalid (19MECSN06)")


# In[3]:


uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")


# In[4]:

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded MRI.', use_column_width=True)
  st.write("")
  st.write("Classifying...")
  label = teachable_machine_classification(image, 'keras_model.h5')
  if label == 0:
   st.write("The MRI scan has a brain tumor")
  else:
   st.write("The MRI scan is healthy")

