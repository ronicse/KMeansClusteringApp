import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("\U0001F308 KMeans Clustering App")
st.subheader("Data Science with Dr. MS Rahman")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
use_example = st.sidebar.checkbox("Use example dataset")

# Load Data
if use_example:
  df = sns.load_dataset('iris')
  df = df.dropna()
  st.success("Loaded sample dataset: 'IRIS'")
else:
  uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
  else:
    st.warning("Please upload a CSV file or use the example dataset.")
    st.stop()

# Show Dataset
st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Data Preprocessing")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
  st.error("Need at least two numeric columns for clustering.")
  st.stop()

features = st.multiselect("Select feature columns for clustering",numeric_cols, default=numeric_cols)
if len(features) == 0:
  st.write("Please select at least one feature.")
  st.stop()

# Drop missing values
df = df[features].dropna()

# Elbow method
st.subheader("Find Optimal Number of Clusters (Elbow Method)")
max_k = st.slider("Maximum number of clusters to test", min_value=2, max_value=10, step=1)









  
