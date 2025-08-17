import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("KMeans Clustering App")
st.subheader("Data Science with Dr. MS Rahman")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
st.sidebar.checkbox("Use example dataset")
