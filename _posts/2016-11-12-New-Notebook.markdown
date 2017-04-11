---
layout:     notebook
title:      Predicting Bad Loans in the Fannie Mae Data Set
author:     Kyle DeGrave
tags: 		  jupyter workflows template
subtitle:   Analyzing Single Family Loan Performance Data
category:   project1

notebookfilename: project_loans
#visualworkflow: true
---


# Turning Online Reviews into Actionable Insights


# Table of Contents
1. [Introduction](#introduction)

# Introduction
Back in December, I was lucky enough to have been accepeted into the Insight Data Science program. Insight is an intensive 7-week program designed to help individuals in academia transition into data science roles.


## Loading the Data
Below, we start by reading the review data into a Pandas dataframe, and drop any rows containing missing values.


```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from collections import Counter
from wordcloud import WordCloud
from string import punctuation
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import random
import string
import nltk
import re

import matplotlib.pyplot as mp
%matplotlib inline

from plotly.offline import download_plotlyjs
from plotly.offline import init_notebook_mode
from plotly.offline import plot, iplot
import cufflinks as cf

cf.go_offline()

# Read data from csv file
df = pd.read_csv('/Users/degravek/Insight/project/podium_code/reviews10000.csv', header=0)
df.rename(columns={'Rating': 'rating', 'Review Text': 'text', 'Location Id': 'location',
                    'Publish Date': 'date', 'Industry': 'industry'}, inplace=True)

# For speed purposes, we can cut the dataframe down
#df = df[:100]

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
```
