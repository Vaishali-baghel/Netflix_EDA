#!/usr/bin/env python
# coding: utf-8

# ** Import Libaries and Data **

# In[1]:


import pandas as pd  # linear algebra operation
import numpy as np # used for data preparation 
import plotly.express as px # used for visualization
from textblob import TextBlob # used for sentiment analysis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import plotly.graph_objs as go


# In[2]:


df = pd.read_csv("netflix_titles (1).csv")


# Checking number of rows and columns in data

# In[3]:


df.tail()


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.isnull().sum().sort_values(ascending=False)


# In[9]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[10]:


df['director'].value_counts().head(10)


# # Movies Vs TVShows

# In[11]:


go.Figure(data=[go.Pie(labels = df.type.value_counts(normalize=True).index,
                       values = df.type.value_counts(normalize = True).values,
                       hole = 0.5,title = "Movies Vs TVShows")])


# In[12]:


df.type.value_counts()


# # Which content is available in the most amount on netflix 

# In[13]:


Rating = df.groupby(['rating']).size().reset_index(name = 'counts')
Rating


# In[15]:


sns.barplot(x=df.rating.value_counts(), y=df.rating.value_counts().index, data =df,
           orient ="h")
plt.show()


# In[16]:


df.country.value_counts().head(10)


# # Year Wise Counts

# In[31]:


plt.figure(figsize=(12,10))
ax = sns.countplot(y='release_year',data = df, order=df.release_year.value_counts().index[0:15])


# Highest Releases in 2018 followed by 2017 and 2019

# In[17]:


df['director'].value_counts().head(10)


# In[19]:


df.listed_in.value_counts().head(10)


# In[22]:


plt.figure(figsize=(12,10))
ax = sns.countplot(y='listed_in', data = df, order=df.listed_in.value_counts().index[0:15])


# # Handling Missing Values

# In[4]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[25]:


df.isnull().sum().sort_values(ascending=False)


# # Droping rows for small percentages of null

# In[5]:


df.dropna(subset=['rating','duration'],axis=0, inplace=True)


# In[6]:


df.shape


# In[15]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[8]:


df.dropna(subset=['date_added'],axis=0, inplace=True)


# # Replacing The Null Values 

# #Replacing missing values in country with "unknown"

# In[9]:


df["country"].replace(np.nan, "Unknown", inplace=True)


# In[10]:


df.country.value_counts().head(10)


# In[11]:


df.cast.value_counts().head(10)


# In[12]:


## Replacing missing values in cast with "NO Cast"


# In[13]:


df["cast"].replace(np.nan, "No Cast", inplace=True)


# In[14]:


df['director']=df['director'].fillna('No Director Specified')


# In[33]:


df['title'].head(10)


# In[23]:


# filtered_cast_shows = ["cast"]
# cast_shows = df[df.cast != 'No Cast'].set_index('title').cast.str.split(',', expand=True).stack().reset_index(level=1, drop=True)
# plt.figure(figsize=(13,7))
# plt.title('Top 10 Actor Movies based on the number of titles')
# sns.countplot(y = filtered_cast_shows, order = cast_shows.value_counts().index[:11],palette='pastel')
# plt.show()

cast_shows = df['cast'].str.split(',',expand = True).stack()
cast_shows = cast_shows.to_frame()
cast_shows.columns = ['Actors']
actors = cast_shows.groupby(['Actors']).size().reset_index(name = "Total_Count")
actors = actors[actors.Actors != 'No Cast'].sort_values(by = ['Total_Count'], ascending = False)
top_10_Actors = actors.head(10)
top_10_Actors = top_10_Actors.sort_values(by = ['Total_Count'])
barChart2= px.bar(top_10_Actors, x = 'Total_Count', y = 'Actors', title = "top 10 Actors on Netflix")
barChart2.show()


# In[31]:


df.head(2)


# In[20]:


movies_df = df.loc[(df['type']=='Movie')]
movies_df.head(2)


# In[21]:


shows_df = df.loc[(df['type']=='TV Show')]
shows_df.head(2)


# In[22]:


movies_df.duration = movies_df.duration.apply(lambda x:x.replace("min","")if "min" in x else x)
movies_df.head(2)


# In[23]:


movies_df.info()


# In[25]:


movies_df.loc[:,["duration"]] = movies_df.loc[:,["duration"]].apply(lambda x: x.astype('int64', errors='ignore'))
movies_df.describe()


# In[26]:


# shortest Movie
shortest_movie = movies_df.loc[(movies_df['duration']==np.min(movies_df.duration))]
shortest_movie


# In[27]:


# Longest Movie
longest_movie = movies_df.loc[(movies_df['duration']==np.max(movies_df.duration))]
longest_movie


# In[29]:


longest_movies = movies_df.loc[(movies_df['duration']>=200)]
longest_movies


# In[33]:


shows_df.duration = shows_df.duration.apply(lambda x: x.replace('Season','')if 'Season' in x else x)
shows_df.head(2)


# In[34]:


shows_df.duration = shows_df.duration.apply(lambda x:x.replace('s','')if 's' in x else x) 
shows_df.head(3)


# In[36]:


shows_df.info()


# In[38]:


shows_df.loc[:,['duration']]=shows_df.loc[:,["duration"]].apply(lambda x: x.astype('int64', errors='ignore'))
shows_df.describe()


# In[39]:


shows_df.duration.value_counts()


# In[41]:


# show with highest no. of seasons
longest_shows = shows_df.loc[(shows_df['duration']>13)]
longest_shows


# In[42]:


longest_shows.rating.value_counts()


# In[44]:


netflix_date = df[['date_added']].dropna()
netflix_date['year'] = netflix_date["date_added"].apply(lambda x: x.split(', ')[0])
netflix_date['month'] = netflix_date['date_added


# In[ ]:





# In[ ]:





# In[18]:


directors_list = pd.DataFrame()
print(directors_list)


# In[32]:


directors_list = df['director'].str.split(',', expand = True).stack()
directors_list.head(5)


# In[11]:


directors_list = directors_list.to_frame()
print(directors_list)


# In[12]:


directors_list.columns = ['Directors']
print(directors_list)


# In[13]:


directors = directors_list.groupby(['Directors']).size().reset_index(name = 'Total_count')
print(directors)


# In[14]:


directors = (directors[directors.Directors != 'Director not specified']).sort_values(by = ['Total_count'], ascending = False)
directors


# In[15]:


top10directors = directors.head(10)
top10directors


# In[16]:


top5directors = top5directors.sort_values(by = ['Total_count'])
barChart = px.bar(top5directors, x = 'Total_count', y = 'Directors', title = 'Top 5 Directors on Netflix',)
barChart.show()


# In[17]:


df.info()


# # Analyzing the top 10 Actors on netflix

# In[18]:



cast_df = df['cast'].str.split(',',expand = True).stack()
cast_df = cast_df.to_frame()
cast_df.columns = ['Actors']
actors = cast_df.groupby(['Actors']).size().reset_index(name = "Total_Count")
actors = actors[actors.Actors != 'No cast Specified'].sort_values(by = ['Total_Count'], ascending = False)
top_10_Actors = actors.head(10)
top_10_Actors = top_10_Actors.sort_values(by = ['Total_Count'])
barChart2= px.bar(top_10_Actors, x = 'Total_Count', y = 'Actors', title = "top 10 Actors on Netflix")
barChart2.show()


# In[19]:


df.info()


# # Analyzing the content produced on Netflix on years

# In[20]:


df1 = df[['type', 'release_year']]
df1 = df1.rename(columns = {'release_year': 'Release Year', 'type': 'Type'})
df2 = df1.groupby(['Release Year', 'Type']).size().reset_index(name = "Total Count") 
df2


# In[21]:


df2 = df2.sort_values("Release Year", ascending = False)
graph = px.line(df2, x = 'Release Year', y = 'Total Count', color = 'Type', title = 'Trend of content Produced on Netflix Every Year')
graph.show()


# In[22]:


df2['Release Year'] = pd.to_numeric(df2['Release Year'], errors='coerce')


# In[23]:


df2 =df2[df2['Release Year'] >= 2000]
graph = px.line(df2, x = 'Release Year', y = 'Total Count', color = 'Type', title = 'Trend of content Produced on Netflix Every Year')
graph.show()


# In[24]:


df2.info()


# # Sentiment Analysis of Netflix Content

# In[25]:


df3 = df[['release_year', 'description']]
df3 = df3.rename(columns = {'release_year': 'Release Year', 'description': 'Description'})
for index, row in df3.iterrows():
    d = row['Description']
    testimonial = TextBlob(d)
    p = testimonial.sentiment.polarity
    if p==0:
        sent = 'Neutral'
    elif p>0:
        sent = 'Positive'
    else:
        sent = 'Negative'
    df3.loc[[index, 2], 'Sentiment']=sent
df3 = df3.groupby(['Release Year', 'Sentiment']).size().reset_index(name = "Total Count")    
df3['Release Year'] = pd.to_numeric(df3['Release Year'], errors='coerce')
df3 = df3[df3["Release Year"]> 2005]    
barGraph = px.bar(df3, x = 'Release Year', y = 'Total Count', color = 'Sentiment', title = "Sentiment Analysis of content on Netflix ")
barGraph.show()


# In[26]:


df.describe()


# In[27]:


df.info()


# In[ ]:




