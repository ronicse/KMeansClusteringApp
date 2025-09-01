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
max_k = st.slider("Maximum number of clusters to test", min_value=2, max_value=10, value=10, step=1)
wcss = []

for k in range(1, max_k+1):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(df)
  wcss.append(kmeans.inertia_)


fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, max_k+1), wcss, marker='o')
ax_elbow.set_xlabel('Number of Clusters (k)')
ax_elbow.set_ylabel('WCSS (Inertia)')
ax_elbow.set_title("Elbow Method For Optimal k")
st.pyplot(fig_elbow)

st.subheader("KMeans Model Training")
n_clusters = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3, step=1)
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(df)
labels = model.labels_

df_clustered = df.copy()
df_clustered['Cluster'] = labels

st.success("KMeans clustering complete !")

# Show Dataset
st.subheader("Final Dataset Preview")
st.write(df_clustered.head())









  
