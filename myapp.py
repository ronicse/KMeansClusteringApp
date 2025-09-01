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
st.subheader("Dataset Preview After Training")
st.write(df_clustered.head())

st.subheader("Cluster Centers (original scale)")
st.write(pd.DataFrame(model.cluster_centers_, columns=features))

if len(features) >= 2:
  st.subheader("Cluster Visualization")
  plt.figure()
  scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap='viridis')
  plt.xlabel(features[0])
  plt.ylabel(features[1])

  elements = scatter.legend_elements()
  handles = elements[0]
  labels_list = elements[1]
  plt.legend(handles, labels_list, title='Clusters')
  st.pyplot(plt)
  
else:
  st.info("Select at least two features to view scatter plot of clusters")
  

st.subheader("Predict Cluster for New Input")

input_data = {}
valid_input = True

for feature in features:
  user_val = st.text_input(f"Enter {feature} (numeric value)")
  try:
    if user_val.strip()=="":
      valid_input = False
    else:
      input_data[feature] = float(user_val)
  except ValueError:
    valid_input = False


if.button("Predict Cluster"):
  if valid_input:
    input_df = pd.DataFrame([input_data])[features]
    cluster_pred = model.predict(input_df)
    st.success(f"The new input belongs to Cluster: {cluster_pred}")
  else:
    st.error("Please enter valid numeric values for all features before predicting.")





  
