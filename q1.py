import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# read dataset into dataframe
wines = pd.read_csv('wineQualityReds(1).csv')

# drop wine from the dataframe
wines.drop('Wine', axis=1, inplace=True)

# extract quality and save it as a separate variable
quality = wines['quality']

# drop quality from the dataframe
wines.drop('quality', axis=1, inplace=True)

# print the dataframe and quality
print(wines)
print(quality)

# normalize all columns of the dataframe using MinMaxScaler
myScaler = MinMaxScaler()
myScaler.fit(wines)
wineDataNorm = pd.DataFrame(myScaler.transform(wines), columns=wines.columns)

# print the normalized dataframe
print(wineDataNorm)

# create range of k values from 1:21 for k-means clustering; iterate on the k values and store the inertias
ks = range(1, 21)
inertia = []
for k in ks:
    model = KMeans(n_clusters=k, random_state=2022)
    model.fit(wineDataNorm)
    inertia.append(model.inertia_)

# plot the chart of inertia vs number of clusters
plt.plot(ks, inertia, "-o")
plt.xlabel("Number of Clusters, k")
plt.ylabel("Inertia")
plt.xticks(ks)
plt.show()

# cluster the wines into k=6 clusters and assign cluster numbers to each wine
model = KMeans(n_clusters=6, random_state=2022)
model.fit(wineDataNorm)
labels = model.predict(wineDataNorm)
wineDataNorm["cluster Label"] = pd.Series(labels)
print(wineDataNorm)
print(plt.hist(labels))

# add quality back to the dataframe
wineDataNorm["quality"] = quality

# print a crosstab of cluster number vs. quality
print(pd.crosstab(wineDataNorm["quality"], wineDataNorm["cluster Label"], values=None, rownames=None, colnames=None))
