#!/usr/bin/env python
# coding: utf-8

# # IBM Data Science Capstone Project

# ## Introduction

# Hong Kong is a paradise for food lovers. It is famously known to be the culinary capital of Asia offering a wide variety of world's delicious food. In recent years, coffee culture has been brewing a storm and growing in popularity in Hong Kong. Hanging out in cafes became a popular trend among the younger generation. According to a market search, revenue in the coffee segment amounts to US$ 1,352 million in 2020 and the market is expected to grow annually by 8.1%. 
# 
# With consumers’ growing appreciation for coffee, more and more investors are motivated to open cafes in Hong Kong. In the brick-and-mortar retail world, it’s said that the three most important decisions you’ll make are location, location, and location. So putting the cafe in the proper location might be the single most important thing to do at startup. By using data science methods and machine learning techniques such as clustering, this project aims to identify the best location for running a cafe in Hong Kong.

# ## Business Problem

# The main idea behind the project is to help investors to analyse the optimal location for opening cafes in Hong Kong. However, opening a cafe in Hong Kong can be challenging due to high rent. Also, most business districts are now being awash with coffee shops. Starting a cafe business in such an area could be very competitive and won’t be much profitable.Therefore, it is very important to find out the best possible neighborhood for opening a cafe.

# ## Data Source

# We need the following data sources to extract the required information:
# 
# 1. List of the districts of Hong Kong downloaded from: https://www.rvd.gov.hk/doc/tc/hkpr20/Appendix_TC.xlsx
# 
# 2. Geo-coordinates of the districts generated via Geocoder API
# 
# 3. Top Venues of districts data collected using Foursquare API

# ## Import libraries

# In[1]:


# library to handle data in a vectorized manner
import numpy as np

# For data manipulation and analysis(Dataframe)
import pandas as pd

# Display full dataframe without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Extract data from Excel file
get_ipython().system('pip install xlrd')

# install and import geocoder
# !pip install geocoder
# import geocoder

# convert an address into latitude and longitude values
get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="hk_explorer")

# library to handle JSON files
import json

# library to handle requests (Foursquare API to return results)
import requests

# tranform JSON file into a pandas dataframe
from pandas.io.json import json_normalize

# map rendering library
get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
get_ipython().system('pip install folium==0.5.0')
import folium

# matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
import matplotlib.colors as colors

#import k-means from clustering
from sklearn.cluster import KMeans

# from sklearn import datasets
# from sklearn import metrics
# from scipy.spatial.distance import cdist

print("")
print("-------------Libraries imported!------------------")


# ## Import Dataset

# Hong Kong is divided into 18 Districts which are further divided into 127 sub-districts. The available dataset in the Government website is in Excel format. 
# 
# First, we import the Excel file **Appendix_TC.xlsx** that contains the 18 districts and the neighborhoods that exist in each district.

# In[2]:


# Import dataset into Pandas Dataframe

hkn = pd.read_excel("https://www.rvd.gov.hk/doc/tc/hkpr20/Appendix_TC.xlsx", header = 4)
hkn.head()


# In[3]:


hkn.shape


# ## Data Cleaning

# In[4]:


# Remove unrelated columns

del hkn ["區 域 \nArea"]
del hkn["地 區 內 的 分 區 名 稱"]
del hkn["小 規 劃 統 計 區 \nTertiary Planning Units"]
hkn.head()


# In[5]:


# Simplify the column name

hkn.rename(columns={"地 區 \nDistrict" : "District", 
                   "Names of Sub-districts\nwithin District Boundaries":"Neighborhood" } ,
          inplace = True)

hkn.head()


# In[6]:


#Replace bogus \n with spacing from data

hkn = hkn.replace('\n',' ', regex=True)
hkn.head()
hkn.tail(15)


# In[7]:


# to remove the empty rows (do not contain any data)
# check all rows if it is null (True = null, False = not null), store the result in df2

removed_elements = []
df2 = pd.isnull(hkn)
df2


# In[8]:


# remove the rows if both District and Neighborhood are null

for n in range(len(hkn)) : 
    if df2.at[n,'District'] and df2.at[n, 'Neighborhood'] :
        removed_elements.append(n)
        
hkn.drop(removed_elements, axis = 0, inplace = True)

hkn


# In[9]:


# Remove irrelevant multiple consecutive rows from line 28 to 35

hkn.drop([28, 30, 31, 32, 33, 34, 35], axis = 0, inplace = True)


# In[10]:


# Remove Chinese characters in the dataframe

import string

printable = set(string.printable)
hkn['District'] = hkn['District'].apply(lambda row: ''.join(filter(lambda x: x in printable, row)))


# In[11]:


hkn


# In[12]:


# to split the Neighborhood
#hkn = hkn.assign(Neighborhood=hkn['Neighborhood'].str.split(',')).explode('Neighborhood')


# The other method to split the Neighborhood

hkn = (hkn.set_index(hkn.columns.drop('Neighborhood',1).tolist())
 .Neighborhood.str.split(',', expand=True)
 .stack()
 .reset_index()
 .rename(columns={0:'Neighborhood'})
 .loc[:, hkn.columns]
)


# In[13]:


hkn.head(20)


# ## Add Geo-coordinates to each district

# Before we want to get the top venues data from Foursquare, we need to get the geo-coordinates of each district

# In[14]:


# Get the coordinates of each Neighborhood
hkn["Coordinates"] = hkn["Neighborhood"].apply(geolocator.geocode)

# Use apply lambda function to get the latitude and longitude and store it in respective columns
hkn["Latitude"] = hkn["Coordinates"].apply(lambda x: x.latitude if x!= None else None)
hkn["Longitude"] = hkn["Coordinates"].apply(lambda x: x.longitude if x!= None else None)
hkn


# In[15]:


# Check if the dataframe contains any missing values
hkn.isnull().values.any()


# In[16]:


hkn.isnull().sum()


# In[17]:


hkn.at[124, "Neighborhood"] = "Tung Chung"
hkn.at[125, "Neighborhood"] = "Discovery Bay"


# In[18]:


hkn.tail()


# As we can see that the latitude and the longitude of some of the neighborhoods are obviously mistakenly located via Geocoder, we are going to add a new column called address.

# In[19]:


# In case the district is not found via Geocoder, we can concatenate the neighborhood with the area to form a detailed address.
hkn["Address"] = hkn["Neighborhood"] + "," + "Hong Kong, China"


# In[20]:


hkn.head()


# In[21]:


# Check if the dataframe contains any missing values
hkn.isnull().values.any()


# In[22]:


# Get the coordinates of each district
hkn["Coordinates"] = hkn["Address"].apply(geolocator.geocode)

# Use apply lambda function to get the latitude and longitude and store it in respective columns
hkn["Latitude"] = hkn["Coordinates"].apply(lambda x: x.latitude if x!= None else None)
hkn["Longitude"] = hkn["Coordinates"].apply(lambda x: x.longitude if x!= None else None)
hkn


# In[23]:


# As we get all the coordinates of each neighborhood correctly, we may remove "Coordinates" and "Address" column now
del hkn["Coordinates"]
del hkn["Address"]


# In[24]:


hkn


# ## Visualize neighborhoods on map 

# We now visualize the center locations of each district. The map of Hong Kong is created with districts superimposed on top.

# In[25]:


# The geodata for Hong Kong

address = "Hong Kong, China"

geolocator = Nominatim(user_agent="hk_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Hong Kong are {}, {}.'.format(latitude, longitude))


# In[26]:


# create map of Hong Kong using latitude and longitude values
map_hongkong = folium.Map(location=[latitude, longitude], tiles = 'cartodbpositron', zoom_start=10)

# Folium supports displaying different tiles in the same map
tiles = ['cartodbpositron', 'openstreetmap']
for tile in tiles:
    folium.TileLayer(tile).add_to(map_hongkong)

# Add the layer control button at the top right hand side
folium.LayerControl().add_to(map_hongkong)

# add markers to map
for lat, lng, neighborhoods in zip(hkn['Latitude'], hkn['Longitude'], hkn['Neighborhood']):
    label = '{}'.format(neighborhoods)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_hongkong)  
    
map_hongkong


# ## Exploring Top Venues for each neighborhood with Foursquare API

# #### Define Foursquare Credentials and Version

# In[27]:


CLIENT_ID = 'I0AVIIA5NFB1R5PB1FYFVRGM1NZ42CLK15IVDFFYKUJX1WTZ' # your Foursquare ID
CLIENT_SECRET = '1G4MRZXUS04SOYWMFTCZJP5KC5PGEDBSMWLFORMZD3D4UTUM' # your Foursquare Secret
VERSION = '20200522' # Date of Today

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# Let's examine the first neighorhood in the dataframe

# In[28]:


hkn.loc[0, "Neighborhood"]


# In[29]:


neighborhood_latitude = hkn.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = hkn.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = hkn.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# Getting the top 100 venues that are in I. Kennedy Town (First district) within a radius of 1000 meters. This will be obtained from Foursquare.
# 

# In[30]:


radius = 1000
LIMIT = 100

# Define the corresponding URL to get the venues data from Foursquare API
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url


# In[31]:


# Send the GET Request to Foursquare and the results will be returned in JSON format

results = requests.get(url).json()
results


# In[32]:


# define a function that extracts the category of the venue

def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# Now we are ready to tranform JSON file into a pandas dataframe

# In[33]:


venues = results['response']['groups'][0]['items']

nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# In[34]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## Explore other neighborhoods

# In[35]:


# Let's create a function to repeat the same process to all the neighborhoods

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'N_Latitude', 
                  'N_Longitude', 
                  'Venue', 
                  'V_Latitude', 
                  'V_Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[37]:


hk_venues = getNearbyVenues(names=hkn['Neighborhood'],
                            latitudes=hkn['Latitude'],
                            longitudes=hkn['Longitude']
                           )


# In[38]:


print(hk_venues.shape)
hk_venues.head()


# In[39]:


# Check if the dataframe contains any missing values
hk_venues.isnull().values.any()


# Now, checking how many venues were collected for other districts as well.

# In[40]:


hk_venues.groupby('Neighborhood').count()


# Checking how many distinct venue categories we have

# In[41]:


print('There are {} uniques categories.'.format(len(hk_venues['Venue Category'].unique())))


# ## Analyzing the Districts

# For that, we use one hot encoding.

# In[42]:


# one hot encoding
hk_onehot = pd.get_dummies(hk_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
hk_onehot['Neighborhood'] = hk_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [hk_onehot.columns[-1]] + list(hk_onehot.columns[:-1])
hk_onehot = hk_onehot[fixed_columns]

hk_onehot.head()


# In[43]:


hk_onehot.shape


# Grouping rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[44]:


hk_grouped = hk_onehot.groupby('Neighborhood').mean().reset_index()
hk_grouped


# In[45]:


hk_grouped.shape


# The size of grouped dataframe is different from the neighborhood dataframe. Let's find out it.
# 

# In[46]:


missing_neighborhood = [i for i in hkn['Neighborhood'].unique() if i not in hk_grouped['Neighborhood'].unique()]

missing_neighborhood


# The result shows that there are four places missing in the grouped dataframe. As far as we know, Stonecutters Island is a military port, while Tai Lam Chung is country park where famous for the Tai Lam Chung Reservoir. Pat Heung and Luk Keng are rural areas without business activities. Therefore, it is a good idea to exclude it from the dataset.

# In[47]:


hkn = hkn[hkn['Neighborhood'] != 'Stonecutters Island']
hkn = hkn[hkn['Neighborhood'] != 'Tai Lam Chung']
hkn = hkn[hkn['Neighborhood'] != 'Pat Heung']
hkn = hkn[hkn['Neighborhood'] != 'Luk Keng']


# In[48]:


hkn


# In[49]:


# Print each neighborhood along with the top 5 most common venues.

num_top_venues = 5

for hood in hk_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = hk_grouped[hk_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# Putting it into dataframe

# In[50]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[51]:


# Create the new dataframe and display the top 10 venues for each neighborhood.

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = hk_grouped['Neighborhood']

for ind in np.arange(hk_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(hk_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Using Machine Learning for Clustering Neighborhoods

# Let's find the optimal number of k for running k-means clustering by using the Elbow method. The Elbow method is a very popular technique and the idea is to run k-means clustering for a range of clusters k (let’s say from 1 to 10) and for each value, we are calculating the sum of squared distances from each point to its assigned center(distortions). When the distortions are plotted and the plot looks like an arm then the “elbow”(the point of inflection on the curve) is the best value of k.

# ### Using Elbow Method to find the optimal k

# In[52]:


# Running K-Means with a range of k

hk_grouped_clustering = hk_grouped.drop('Neighborhood', 1)

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(hk_grouped_clustering)
    distortions.append(kmeanModel.inertia_)

    
# Plotting the distortions of K-Means    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# ### Using silhouette score to find the optimal k

# In[53]:


def plot(x, y, xlabel, ylabel):
    plt.figure(figsize=(20,10))
    plt.plot(np.arange(2, x), y, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(2, x))
    plt.show()
    
    
max_range = 8

hk_grouped_clustering = hk_grouped.drop('Neighborhood', 1)

from sklearn.metrics import silhouette_samples, silhouette_score

indices = []
scores = []

for kclusters in range(2, max_range) :
    
    # Run k-means clustering
    kmc = hk_grouped_clustering
    kmeans = KMeans(n_clusters = kclusters, init = 'k-means++', random_state = 0).fit_predict(kmc)
    
    # Gets the score for the clustering operation performed
    score = silhouette_score(kmc, kmeans)
    
    # Appending the index and score to the respective lists
    indices.append(kclusters)
    scores.append(score)


plot(max_range, scores, "No. of clusters", "Silhouette Score")


# Based on this graph, we can see that the optimal number of clusters is 6.

# ### Run k-means to cluster the neighborhood into 6 clusters.

# In[54]:


# set number of clusters
kclusters = 6

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(hk_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Creating a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood
# 
# 

# In[55]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[56]:


hk_merged = hkn

hk_merged = hk_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

hk_merged


# In[57]:


# Check if the dataframe contains any missing values
hk_merged.isnull().values.any()


# In[58]:


hk_merged.isnull().sum()


# In[59]:


hk_merged.drop([45, 93, 99], axis = 0, inplace = True)


# In[60]:


hk_merged["Cluster Labels"] = hk_merged["Cluster Labels"].astype(int)


# Now, visualizing the clusters

# In[61]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]


# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(hk_merged['Latitude'], hk_merged['Longitude'], hk_merged['Neighborhood'], hk_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster+1), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Examining Clusters

# ### Cluster 1

# In[62]:


hk_merged.loc[hk_merged['Cluster Labels'] == 0, hk_merged.columns[[1] + list(range(5, hk_merged.shape[1]))]]


# ### Cluster 2

# In[63]:


hk_merged.loc[hk_merged['Cluster Labels'] == 1, hk_merged.columns[[1] + list(range(5, hk_merged.shape[1]))]]


# ### Cluster 3

# In[64]:


hk_merged.loc[hk_merged['Cluster Labels'] == 2, hk_merged.columns[[1] + list(range(5, hk_merged.shape[1]))]]


# ### Cluster 4

# In[65]:


hk_merged.loc[hk_merged['Cluster Labels'] == 3, hk_merged.columns[[1] + list(range(5, hk_merged.shape[1]))]]


# ### Cluster 5

# In[66]:


hk_merged.loc[hk_merged['Cluster Labels'] == 4, hk_merged.columns[[1] + list(range(5, hk_merged.shape[1]))]]


# ### Cluster 6

# In[67]:


hk_merged.loc[hk_merged['Cluster Labels'] == 5, hk_merged.columns[[1] + list(range(5, hk_merged.shape[1]))]]


# ## Conclusion

# By looking at the cluster data, we can see that cluster 2 is the one that we are the most interested in. The majority of the most common venues are food and restaurant. To find the optimal location for running a coffee shop, we can conclude that the best location is indicated in cluster 2. 
# 
# The rest of the clutsters shows their local specialties in the districts: 
# 
# Cluster 1: Trail and scenic lookout
# 
# Cluster 3: Tunnel
# 
# Cluster 4: Harbor / Marina
# 
# Cluster 5: Electronics store
# 
# Cluster 6: Supermarket
# 
# We will look into the details of the cluster 2 to see which neighborhood is the best location for running a coffee shop. The details as well as the conclusion will be included in the report.

# In[ ]:




