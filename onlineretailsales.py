import pandas as pd
data = pd.read_csv("OnlineRetail.csv", encoding='latin1')
print(data.head())

print(data.info())
print(data.columns)
print(data.describe)
print('The null values:', data.isnull().sum())
df_null = round(100*(data.isnull().sum())/len(data), 2)
print (df_null)
data['CustomerID'] = data['CustomerID'].astype(str)
print(data.info())
data.drop(['StockCode'], axis=1, inplace=True)
print(data.columns)
data['Amount'] = data['Quantity']*data['UnitPrice']
rfm_m= data.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
print(rfm_m.head())

# Grouping by Country and calculating total sales
sales_by_country = data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
print('The countries selling the most are:')
print(sales_by_country.head())

# Grouping by Description and calculating total quantity sold
sales_by_product = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print('The products selling the most are:')
print(sales_by_product.head())

# Grouping by Description and calculating the total quantity sold for each product
frequently_sold_products = data.groupby('Description')['Quantity'].count().sort_values(ascending=False)

# Display the top 10 most frequently sold products
print(frequently_sold_products.head(10))

# Grouping by Description and calculating the total quantity sold for each product
frequently_sold_products = data.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)

# Display the top 10 most frequently sold products
print(frequently_sold_products.head(10))

#customer most frequent
CF = data.groupby('CustomerID')['InvoiceNo'].count()
CF = CF.reset_index()
CF.columns = ['CustomerID', 'Frequency']
print(CF.head())

m_r = pd.merge(rfm_m, CF, on='CustomerID', how='inner')
print(m_r.head())

# Convert the 'InvoiceDate' column to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')

print (data['InvoiceDate'])


# Find the maximum date in the dataset
max_date = max(data['InvoiceDate'])
print(max_date)

min_date =min(data['InvoiceDate'])
print(min_date)

# Calculate the date 30 days before the maximum date
last_30_days_start = max_date - pd.DateOffset(days=30)
print(f'Last 30 days start date: {last_30_days_start}')

# Filter the data to include only transactions from the last 30 days
last_30_days_data = data[(data['InvoiceDate'] >= last_30_days_start) & (data['InvoiceDate'] <= max_date)]
'''
# Group by 'Description' and calculate the total quantity sold for each product
totalsales = last_30_days_data.groupby('')['total_sales'].sum().sort_values(ascending=False)
# Print the total sales for the top 10 products
print(totalsales.head(10))
'''
if 'Amount' not in data.columns:
    data['Amount'] = data['Quantity'] * data['UnitPrice']

# Calculate the total sales for the last 30 days for all products combined
total_sales_last_30_days = last_30_days_data['Amount'].sum()

# Print the total sales for the last 30 days
print(f'Total sales for the last 30 days: £{total_sales_last_30_days:.2f}')

data['Diff'] = max_date - data['InvoiceDate']

lt= data.groupby('CustomerID')['Diff'].min()
lt= lt.reset_index()
lt['Diff'] = lt['Diff'].dt.days

da_t =pd.merge(m_r, lt, on='CustomerID', how='inner')
da_t.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
print(da_t.head())

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Assuming the data is already loaded and preprocessed up to this point

# Ensure l_t2 contains only numeric columns
l_t2 = da_t[['Amount', 'Frequency', 'Recency']].copy()

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
l_t2_scaled = scaler.fit_transform(l_t2)

# Initialize empty lists to store the results of the KMeans clustering and silhouette scores
inertia = []
silhouette_scores = []

# Range of k values to try
k_range = range(2, 11)

# Perform KMeans clustering for each value of k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(l_t2_scaled)
    
    # Append the inertia (sum of squared distances of samples to their closest cluster center)
    inertia.append(kmeans.inertia_)
    
    # Calculate the silhouette score
    silhouette_scores.append(silhouette_score(l_t2_scaled, kmeans.labels_))

# Plot the inertia and silhouette scores to determine the optimal k
plt.figure(figsize=(12, 6))

# Plotting the inertia
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs Number of clusters')

# Plotting the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of clusters')

plt.tight_layout()
plt.show()

# Based on the plots, choose the optimal k
# Let's assume optimal_k based on the visualization of inertia and silhouette score
optimal_k = 3  # Adjust this based on the visualization

# Perform KMeans clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(l_t2_scaled)

# Add the cluster labels to the original dataframe
da_t['Cluster'] = kmeans.labels_

# Print the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print('Cluster centers:')
print(pd.DataFrame(cluster_centers, columns=['Amount', 'Frequency', 'Recency']))

# Visualize the clusters (you can choose any two dimensions for plotting)
plt.figure(figsize=(8, 6))
plt.scatter(da_t['Frequency'], da_t['Amount'], c=da_t['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Amount')
plt.title('Visualization of Clusters')
plt.colorbar(label='Cluster')
plt.show()

# Explore the characteristics of each cluster
cluster_summary = da_t.groupby('Cluster').agg({
    'Amount': 'mean',
    'Frequency': 'mean',
    'Recency': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).reset_index()

print('Cluster summary:')
print(cluster_summary)

