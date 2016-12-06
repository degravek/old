---
layout:     notebook
title:      Twitter Mining UFC 202&#58; Diaz vs. McGregor
author:     Kyle DeGrave
tags: 		  jupyter workflows template
#subtitle:   Predicting Survival Rates on Titanic
category:   project1

#notebookfilename: project_ufc
#visualworkflow: true
---


# Introduction
The Ultimate Fighting Championchip (UFC) is a premier mixed martial arts (MMA) organization founded in 1993, and headquartered in the United States. The organization contains over 500 of the worlds top fighters who continually pit themselves against one another for a shot at winning the championchip belt in their respective weight class. To date, the UFC has held over 300 televised events, taking place about once every month on average at various locations around the world.

On March 6, 2016 at UFC 196, UFC Featherweight Champion Conor McGregor faced off against Nate Diaz in what was to become a strong cadidate for fight of the year. McGregor was the heavy favorite going into the bout, with Diaz taking the fight on only eleven days notice as a replacement for another injured fighter. In a shocking upset, Diaz finished McGregor by submission in the very first round, taking the MMA community by storm. On August 20, 2016 at UFC 202, a heavilyanticipated rematch between the two fighters took place in Las Vegas, Nevada. McGregor went in again as a 2 to 1 fan favorite, and again, the fight did not disappoint. The bout went all five rounds, with McGregor winning by majority decision.

In this analysis, I try my hand for the first time at Twitter mining using an application programming interface (API) written in Python (visit this link for a description of the API). The API was used to stream tweets by MMA fans around the world in the 15 minutes leading up to the Diaz McGregor rematch. In total, over 16,000 tweets were collected for the following analysis. By digging through these Tweets, there are several questions I hope to address:

From which countries/locations are the tweets originating?
What does the distribution of languages spoken by these users look like?
What what was the variation in tweet rate (tweets per second) over the 15 minute time span?
Which fighter was mentioned most often?
We begin the analysis by looping over the JSON data file line by line, and storing the tweets in a list. The JSON file contains a very large amount of useful information, along with many other string characters and expressions that we are not interested in analyzing. A dataframe is created, and we fill it by parsing the data to isolate tweet content, language, location, time zone, and time-of-tweet. Many columns contain several missing entries of type "None", and so these are replaced with NaN's.


```python
from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud, STOPWORDS
from matplotlib import animation
from pygeocoder import Geocoder
import seaborn as sns
import pandas as pd
import numpy as np
import json
import re

import matplotlib.pyplot as mp
%matplotlib inline

data_path = '/Users/degravek/Downloads/ufc.json'

tweets_data = []
tweets_file = open(data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

df = pd.DataFrame()
df['Text'] = list(map(lambda tweet: tweet.get('text', None), tweets_data))
df['Language'] = list(map(lambda tweet: tweet.get('lang', None), tweets_data))
df['Location'] = list(map(lambda tweet: tweet.get('user', {}).get('location'), tweets_data))
df['Tzone'] = list(map(lambda tweet: tweet.get('user', {}).get('time_zone'), tweets_data))
df['Tstamp'] = list(map(lambda tweet: tweet.get('timestamp_ms', None), tweets_data))
```

# Looking at Languages
Once the data are in dataframe format, it is straight forward to examine the various languages spoken by the users. The first thing I note is that the language column is composed of entries like "en", "es", "pt", etc., which are short abbreviations for the different languages. For clarity, we can make a Python dictionary which maps these abbreviations to the full "English", "Spanish", "Portuguese", etc., languages. We can then invoke the value_counts() command to find the number of users speaking each language, and plot the values as a bar chart.


```python
country_map = {'en':'English', 'und':'Undetermined', 'pt':'Portuguese', 'es':'Spanish', 'ja':'Japanese', 'fr':'French', 'pl':'Polish', 'ru':'Russian', 'de':'Danish', 'ar':'Arabic'}

df['Language'] = list(df['Language'].map(country_map))
```


```python
sns.countplot(y=df['Language'], color='deepskyblue')
mp.xlabel('Number of People')
mp.ylabel('Language')
```




    <matplotlib.text.Text at 0x1a84ddc50>




![png](output_4_1.png)


English is far and away the most common language spoken by these paticular users. Portuguese was the second most common language, which is not so surprising considering the very large Brazilian MMA fanbase that closely follows UFC events. It's nice to see that the viewership is quite geographically extensive, reaching many foreign countries located in several different time zones. It's worth noting that the Diaz McGregor fight began aroung 9:45 pm (Mountain time) on a Saturday night. This means that users in western Europe were watching at around 5:45 am on Sunday morning. That's some serious dedicaiton!

# Tweets by Location
We can get a better feel for which countries these users were tweeting from by looking at their actual geographic location. Lucky for us, many Twitter users have made public the city, state, and/or country in which they live, and these tags are included in the tweets. It's fair to say that the location information, in its raw form, is very messy. For example, a subset of raw entries looks something like this:


```python
0                         UK/Ireland
1                        El Salvador
2                                18x
3                      United States
4                Gloucester, England
5                           Miami,FL
6                       milwaukee,wi
7     Colorado via McKees Rocks, PA 
8                         Everywhere
9     every mirror you're staring at
10                  Philadelphia, PA
11                       Philippines
12                 Moreno Valley, CA
13            My conscience : Me too
```


      File "<ipython-input-54-991dd8709329>", line 1
        0                         UK/Ireland
                                   ^
    SyntaxError: invalid syntax



I still have no idea why someone would enter their location as "18x", or as "My conscience : Me too", but there we are. Many locations are entered, in order, as (city, country), (city, state), (state, country), only city, only state, or only country. Ideally, I would like to quickly scan through the various locations and extract the corresponding latitude and longitude locations wherever a (city, country) or (city, state) combination is present. Luckily, there is an amazing Python library called geopy which is capable of extracting the coordinates from this kind of messy data. I first split the location column by comma, and keep only those entries where the number of separate strings is equal to 2. The idea here is that any location that is correctly formatted as (city, country) or (city, state) will only have two entries. Then, for each remaining entry in the dataframe, we loop over location, extract the coordinates when possible, and place them in a dataframe called df_world.


```python
df['nLocation'] = df.loc[df.Location.notnull(), 'Location'].apply(lambda x: x.split(','))

new = []
for i in range(0,df.shape[0]):
    try:
        tmp = len(df.nLocation[i])
        new.append(tmp)
    except:
        tmp = np.nan
        new.append(tmp)

df['tmp'] = new
df['city_country'] = np.where(df['tmp']==2,df['Location'], np.nan)

df = df.dropna(subset=['city_country']).reset_index(drop=True)

Lat = []
Lon = []
for i in range(0,df.shape[0]):
    try:
        results = Geocoder.geocode(df.city_country[i])
        tmp_lat = results.coordinates[0]
        tmp_lon = results.coordinates[1]
        Lat.append(tmp_lat)
        Lon.append(tmp_lon)
    except:
        tmp_lat = np.nan
        tmp_lon = np.nan
        Lat.append(tmp_lat)
        Lon.append(tmp_lon)

df_world = df.copy()
df_world['lat'] = Lat
df_world['lon'] = Lon
df_world = df_world.dropna(subset=['lat']).reset_index(drop=True)
```

This process leaves us with 3,576 (1,690 unique) correctly formatted (city, country) pairs, along with their corresponding latitude, longitude positions. We can now use the Basemap package to visualize the spatial distribution of these points.


```python
mp.figure(figsize=(12,9))
eq_map = Basemap(projection='robin', resolution = 'h', area_thresh = 1000.0, lat_0=0, lon_0=0)
eq_map.drawcoastlines()
eq_map.drawcountries()
eq_map.fillcontinents(color = 'gray')
eq_map.drawmapboundary()
eq_map.drawmeridians(np.arange(0, 360, 30))
eq_map.drawparallels(np.arange(-90, 90, 30))

for lon, lat in zip(df_world.lon, df_world.lat):
    x,y = eq_map(lon, lat)
    eq_map.plot(x, y, 'oy', markersize=2)
```


![png](output_10_0.png)


We see that the bulk of the users are located in the United States, with other heavy concentrations throughout Latin America, South America, and Europe. Let's now zoom in on the United States and immediate surrounding regions.


```python
mp.figure(figsize=(12,9))
eq_map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45, lon_0=-95, resolution='h', area_thresh=10000)
eq_map.drawcoastlines()
eq_map.drawcountries()
eq_map.fillcontinents(color = 'gray')
eq_map.drawmapboundary()
eq_map.drawstates()

for lon, lat in zip(df_world.lon, df_world.lat):
    x,y = eq_map(lon, lat)
    eq_map.plot(x, y, 'oy', markersize=4)
```


![png](output_12_0.png)


I'm glad to see my home state of Michigan representing! In fact, several tweets came from a small town called Gladstone, located about ten miles away from my hometown of Escanaba. There were tweets sent from every single U.S. state including Alaska and Hawaii (not shown here). Remember, there were close to 12,500 other users with badly-formatted locations which are not plotted in these figures, but this is still interesting to see. Remember, all of this was happening in only a 15 minute time interval!

# Tweet Frequency
Okay, now let's take a quick look at tweet frequency. For this we will need to look at the time stamp for each tweet. The time stamp available in the dataframe is initially in string format, so we first convert it to an integer. The values are given in milliseconds, so a division by 1,000 will put them in seconds. I then subtract the smallest time value from the entire column, and sort in ascending order. As there are several tweets per second, the plot may be a bit noisy looking. To smooth it out a little, we can bin the tweets in ten second intervals. The figure below shows the resulting tweet frequency as a function of time leading up to the main event (hence the negative time values). Time zero represents the start of the fight.


```python
df['nTstamp'] = df['Tstamp'].astype(int)
df['nTstamp'] = df['nTstamp']/1000
df['nTstamp'] = df['nTstamp'] - df['nTstamp'].min()

df = df.sort_values(['nTstamp'], ascending=True).reset_index(drop=True)

L = []
trange =np.arange(0,900,10)
for i in trange:
    tmp = df.loc[(df.nTstamp>=[i]) & (df.nTstamp<[i+10]),'Tstamp']
    num = len(tmp)
    L.append(num)

x = trange + 5
mp.plot(x-900, L)
mp.xlabel('Time [seconds]')
mp.ylabel('Number of Tweets per 10 Second Interval')
mp.ylim([0,350])
```




    (0, 350)




![png](output_14_1.png)


The tweet rate is quite high for all times shown here (about 18 tweets per second on average), peaking about six minutes before the main event with tweets being sent at a rate of about 32 per second. We see that there are several distinct peaks visible, corresponding to exciting points during the co-main event between Anothony Johnson and Glover Teixeira. The highest peak corresponds to the point in time immediately after Johnson knocked out Teixeira, only 13 seconds into the fight.

# Common Tweet Terms
Lastly, let's take a look at tweet content. A simple and effective way of quickly looking at common words used by users in their tweets is to use a word cloud. There is a nice package in Python aptly called wordcloud that we will use. I was able to find a well-written article by Sebastian Raschka illustrating the use of this library which can be found [here](http://sebastianraschka.com/Articles/2014_twitter_wordcloud.html).

To create the word cloud, we first join all tweets into a single string. We don't want our word cloud to contain things like retweet abbreviations (RT's), links, and twitter handles, so we can eliminate those. The word cloud is then generated, showing only the top 100 most commonly used terms.


```python
words = ' '.join(df['Text'])
notags = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])

wordcloud = WordCloud(stopwords=STOPWORDS,max_words=100, background_color='black').generate(notags)
mp.figure(figsize=(12,9))
mp.imshow(wordcloud)
mp.axis('off')
```




    (-0.5, 399.5, 199.5, -0.5)




![png](output_16_1.png)


As one may expect, some of the most commonly used terms were "UFC" and "UFC 202", with the names of Diaz, McGregor, and Johnson also very prevalent.

Let's do one final simple comparison were we try and identify the number of supportive tweets sent for fighters Diaz and McGregor. We can assume that a tweet sent by a user in support of a particular fighter may contain only the mention of that fighter's name, Twitter handle, or hashtag, and no others. So let's identify the number of tweets sent where "Nate Diaz", "NateDiaz209", "#NateDiaz", and "#TeamDiaz" were included, with no mention of McGregor. We will then look for instances where McGregor is mentioned, with no reference to Diaz.


```python
diaz_tweets = df[(df.Text.str.contains('NateDiaz209|#NateDiaz|#TeamDiaz')) & (~df.Text.str.contains('TheNotoriousMMA|#ConorMcGregor|#TeamMcGregor'))].shape[0]
mcgr_tweets = df[(df.Text.str.contains('TheNotoriousMMA|#ConorMcGregor|#TeamMcGregor')) & (~df.Text.str.contains('NateDiaz209|#NateDiaz|#TeamDiaz'))].shape[0]

x = [0,1]
y = [2717,2834]
ypos = [0,1]
objects = ('Diaz','McGregor')
mp.barh(x, y, align='center', color='deepskyblue')
mp.yticks(ypos,objects)
mp.xlabel('Number of Tweets Containing Fighter Name')
```




    293



Before separating the fighters, Diaz and Mcgregor were mentioned almost the same number of times (2717 for Diaz, 2834 for McGregor). After separation, I was able to clearly identify 217 and 372 instances of Diaz and McGregor support, respectively. Interestingly, these values account for 37% and 63% of the 589 tweets, which is almost identical to the 35/65 split identified in UFC's own pre-fight poll of fight fans around the world.

# A Video of Worldwide Tweets
As a last aside, I thought it would be interesting to create a short video showing the locations of our Twitter users as they sent out tweets in near real time. To do this, we invoke the animation library from matplotlib to animate a series of Basemap images like the ones shown in the earlier section Looking at Languages. The script is a simple function which loops through the latitude and longitude positions of each user (where available), and plots a single point on our map. These images are collected by animation.FuncAnimation(), and saved in mp4 format. For brevity, I loop over only the first 500 points.


```python
mp.figure(figsize=(12,9))
eq_map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=-130)
eq_map.drawcoastlines()
eq_map.drawcountries()
eq_map.fillcontinents(color = 'gray')
eq_map.drawmapboundary()
eq_map.drawmeridians(np.arange(0, 360, 30))
eq_map.drawparallels(np.arange(-90, 90, 30))

x,y = eq_map(0, 0)
point = eq_map.plot(x, y, 'oy', markersize=12)[0]

def init():
    point.set_data([], [])
    return point,

# Animate the figures
def animate(i):
    print(i)
    lon = df_world.lon[i]
    lat = df_world.lat[i]
    x, y = eq_map(lon, lat)
    point.set_data(x, y)
    return point,

anim = animation.FuncAnimation(mp.gcf(), animate, frames=df_world.shape[0], init_func=init, interval=56, blit=True)

mywriter = animation.FFMpegWriter()
anim.save('mymovie.mp4', writer=mywriter)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-125-42353fa99403> in <module>()
          1 mp.figure(figsize=(12,9))
    ----> 2 eq_map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=-130)
          3 eq_map.drawcoastlines()
          4 eq_map.drawcountries()
          5 eq_map.fillcontinents(color = 'gray')


    /Users/degravek/anaconda/lib/python3.5/site-packages/mpl_toolkits/basemap/__init__.py in __init__(self, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, llcrnrx, llcrnry, urcrnrx, urcrnry, width, height, projection, resolution, area_thresh, rsphere, ellps, lat_ts, lat_1, lat_2, lat_0, lon_0, lon_1, lon_2, o_lon_p, o_lat_p, k_0, no_rot, suppress_ticks, satellite_height, boundinglat, fix_aspect, anchor, celestial, round, epsg, ax)
       1085             # replace coastsegs with line segments (instead of polygons)
       1086             self.coastsegs, types =\
    -> 1087             self._readboundarydata('gshhs',as_polygons=False)
       1088         # create geos Polygon structures for land areas.
       1089         # currently only used in is_land method.


    /Users/degravek/anaconda/lib/python3.5/site-packages/mpl_toolkits/basemap/__init__.py in _readboundarydata(self, name, as_polygons)
       1287                     poly = Shape(b)
       1288                     poly = poly.union(polyE)
    -> 1289                     if not poly.is_valid(): poly=poly.fix()
       1290                     b = poly.boundary
       1291                     b2 = b.copy()


    KeyboardInterrupt: 



    <matplotlib.figure.Figure at 0x1b04992b0>



```python
%%HTML
<video width="320" height="240" controls>
  <source src="http://www.sample-videos.com/video/mp4/720/big_buck_bunny_720p_1mb.mp4" type="video/mp4">
</video>
```


<video width="320" height="240" controls>
  <source src="http://www.sample-videos.com/video/mp4/720/big_buck_bunny_720p_1mb.mp4" type="video/mp4">
</video>


The animation plots approximately five points every second, which is still actually about 3 - 6 times slower than the real-time rates we found in the Tweet Frequency section earlier. It really makes me wonder what kind of tweet rates were seen during, for example, the Olympic opening ceremony a few weeks ago. It must have been off the charts!

# Concluding Remarks
In this project we examined the tweeting habits of fight fans around the world during UFC 202: Diaz vs. McGregor. Over 16,000 tweets were collected using an application programming interface (API) in the 15 minutes leading up to the main event. We found that the majority of tweeters were based in the United States, but significant fanbases are also prevalent in Latin America, South America, and Europe, along with a healthy smattering throughout the Middle East, India, the Philippines, and Australia. This is refreshing to see, as I am also a fight fan, and I know the UFC has been pushing its global brand very hard over the last decade or so. These users are far and away primarily English speakers, with other less-represented languages reflecting the countries mentioned above (i.e. Spanish, Portuguese, etc.).

In terms of tweet frequency, rates averaged about 18 tweets per second, maxing out at nearly twice this value. Rates varied significantly over the short time that the API was collecting tweets, and showed several prominent peaks, the largest of which ocurred 6 minutes before the start of the main event, and corresponding to a stunning knockout finish by Anothony Johnson. Of the two main event fighters, McGregor was mentioned (in isolation) almost twice as often as Diaz, closely reflecting UFC fight fan poll numbers aquired before the bout.


```python

```
