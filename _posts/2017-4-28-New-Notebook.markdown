---
layout:     notebook
title:      Detecting Clickbait Headlines Using Machine Learning and NLP 
author:     Kyle DeGrave
tags: 		  jupyter workflows template
#subtitle:  Analyzing Single Family Loan Performance Data
category:   project1

notebookfilename: project_clickbait_nn
#visualworkflow: true
---


# Introduction
In today's age of information, we are constantly bombarded with advertisements and news headlines of various kinds on virtually every web page we visit. With so many people connected to the internet, websites and news outlets are in continuous competition for viewers, and so they are pushed to produce ever more interesting, catchy, and provacative article headlines, regardless of their veracity. In recent years, this has led to an influx of "news" headlines that are designed essentially entirely to capture the attention of new viewers, rather than to convey any real, practical journalistic information. These catchy headlines and their associated content are referred to as "clickbait". Sites like BuzzFeed and Upworthy are well known for hosting many of these kinds of articles.

In a formal sense, clickbait is somewhat hard to define. The Oxford English dictionary defines it as *"on the internet, content whose main purpose is to attract attention and encourage visitors to click on a link to a particular web page"*. In a general sense, clickbait headlines are those which satisfy two main criteria:

1. They exploit the so-called "curiosity gap" by not explaining the full article contents
2. They provide misleading information about the article contents

In other words, these headlines contain text which leaves the reader curious about what the article contents might be, or they contain text about topics not really covered in the article itself. Examples of clickbait headline include things like

* I left my husband and daughter home alone, and you'll never believe what happened!
* 19 Tweets anyone addicted to diet coke will totally relate to
* What these pilots do with dogs is the most brilliant thing I've ever seen
* She picks this object off the ground, but watch what happens when it starts to move!

We must be sure to make a distinction here between clickbait and fake news, which has also been talked about more and more as of late. The distinction comes from the fact that fake news actively tries to get the reader to believe things that are untrue. Clickbait, on the other hand, usually just contains "junk" news with no real journalistic integrity, and is not constructed to get the reader to believe false claims. 

Since clickbait is becomming increasingly common, and is genereally considered a nuisance to internet users, let's see if we can use machine learning and natural language processing to identify these headlines automatically. This sort of study has been carried out before by, for example, [Chakraborty et al.](http://cse.iitkgp.ac.in/~abhijnan/papers/chakraborty_clickbait_asonam16.pdf), and is covered in [this informative article](https://www.linkedin.com/pulse/identifying-clickbaits-using-machine-learning-abhishek-thakur).

The dataset used in the Chakraborty et al. study is publicly available, and can be found [here](https://github.com/bhargaviparanjape/clickbait/tree/master/dataset). The data contain 16,000 clickbait headlines from BuzzFeed, Upworthy, ViralNova, Thatscoop, Scoopwhoop and ViralStories, along with 16,000 non-clickbait headlines from WikiNews, New York Times, The Guardian, and The Hindu. Let's import some useful Python libraries, and read in the dataset. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from string import punctuation
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re

import matplotlib.pyplot as mp
%matplotlib inline

file = '/Users/degravek/Downloads/clickbait_data.txt'
with open(file, encoding="utf-8") as f_in:
    lines = [line.rstrip() for line in f_in]
    ybait = list([line for line in lines if line])

df_ycb = pd.DataFrame(ybait, columns=['text'])
df_ycb['clickbait'] = 1

file = '/Users/degravek/Downloads/non_clickbait_data.txt'
with open(file, encoding="utf-8") as f_in:
    lines = [line.rstrip() for line in f_in]
    nbait = list([line for line in lines if line])

df_ncb = pd.DataFrame(nbait, columns=['text'])
df_ncb['clickbait'] = 0

df = df_ycb.append(df_ncb, ignore_index=True).reset_index(drop=True)
```

# Feature Engineering
Now that the data is loaded, one option is to move ahead using only the headline text to identify clickbait headlines. It turns out that this actually gives very good results. However, it is also possible to engineer some new features from the available data. These new features include things like:

* Number of words in the headline
* Number of stop words in the headline
* Number of contractions in the headline
* Frequency of various parts of speech in the headline
* Is the headline a question?
* Ratio of stop words to total number of words
* Ratio of contractions to total number of words

To produce these new features, we'll first need to define a few functions. The function process_text carries out the task of processing the raw headlines. It makes every headline lowercase, removes any punctuation and extra white space, and replaces any numerical values with the word "number". Next, the function cnt_stop_words tokenizes (splits) each headline into it's individual words, and counts the number of those words that appear in the NLTK corpus of pre-defined stop words.

It is often the case that clickbait headlines are stated in the form of a question. For example, something like "Can you spot the amazing thing in this photo?". Since the particular dataset we're using doesn't actually contain punctuation like question marks, we can try to determine if the text is stated as a question based on the first word in the headline. Questions typically start with words like "who", "what", "when", "can", etc. To this end, we can define a list of question_words, and a function to check if any of these words starts a headline.

Clickbait headlines also tend to contain more informal writing than non-clickbait headlines. As such, they may contain many more contractions, occurrences of slang, etc. Below, we can make a list of possible contractions to look for, as well as a function to check for them.

Lastly, a function is defined to check the part-of-speech (i.e., noun, verb, adjective, etc.) of each word in a headline. It's possible that non-clickbait headlines contain, for example, more nouns than do clickbait headlines. This part-of-speech tagging is carried out using methods from the NLTK package.


```python
question_words = ['who', 'whos', 'whose', 'what', 'whats', 'whatre', 'when', 'whenre', 'whens', 'couldnt',
        'where', 'wheres', 'whered', 'why', 'whys', 'can', 'cant', 'could', 'will', 'would', 'is',
        'isnt', 'should', 'shouldnt', 'you', 'your', 'youre', 'youll', 'youd', 'here', 'heres',
        'how', 'hows', 'howd', 'this', 'are', 'arent', 'which', 'does', 'doesnt']

contractions = ['tis', 'aint', 'amnt', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',
                'didnt', 'doesnt', 'dont', 'gonna', 'gotta', 'hadnt', 'hadntve', 'hasnt',
                'havent', 'hed', 'hednt', 'hedve', 'hell', 'hes', 'hesnt', 'howd', 'howll',
                'hows', 'id', 'idnt', 'idntve', 'idve', 'ill', 'im', 'ive', 'ivent', 'isnt',
                'itd', 'itdnt', 'itdntve', 'itdve', 'itll', 'its', 'itsnt', 'mightnt',
                'mightve', 'mustnt', 'mustntve', 'mustve', 'neednt', 'oclock', 'ol', 'oughtnt',
                'shant', 'shed', 'shednt', 'shedntve', 'shedve', 'shell', 'shes', 'shouldve',
                'shouldnt', 'shouldntve', 'somebodydve', 'somebodydntve', 'somebodys',
                'someoned', 'someonednt', 'someonedntve', 'someonedve', 'someonell', 'someones',
                'somethingd', 'somethingdnt', 'somethingdntve', 'somethingdve', 'somethingll',
                'somethings', 'thatll', 'thats', 'thatd', 'thered', 'therednt', 'theredntve',
                'theredve', 'therere', 'theres', 'theyd', 'theydnt', 'theydntve', 'theydve',
                'theydvent', 'theyll', 'theyontve', 'theyre', 'theyve', 'theyvent', 'wasnt',
                'wed', 'wedve', 'wednt', 'wedntve', 'well', 'wontve', 'were', 'weve', 'werent',
                'whatd', 'whatll', 'whatre', 'whats', 'whatve', 'whens', 'whered', 'wheres',
                'whereve', 'whod', 'whodve', 'wholl', 'whore', 'whos', 'whove', 'whyd', 'whyre',
                'whys', 'wont', 'wontve', 'wouldve', 'wouldnt', 'wouldntve', 'yall', 'yalldve',
                'yalldntve', 'yallll', 'yallont', 'yallllve', 'yallre', 'yallllvent', 'yaint',
                'youd', 'youdve', 'youll', 'youre', 'yourent', 'youve', 'youvent']

def process_text(text):
    result = text.replace('/', '').replace('\n', '')
    result = re.sub(r'[1-9]+', 'number', result)
    result = re.sub(r'(\w)(\1{2,})', r'\1', result)
    result = re.sub(r'(?x)\b(?=\w*\d)\w+\s*', '', result)
    result = ''.join(t for t in result if t not in punctuation)
    result = re.sub(r' +', ' ', result).lower().strip()
    return result

stop = stopwords.words('english')
def cnt_stop_words(text):
    s = text.split()
    num = len([word for word in s if word in stop])
    return num

def num_contract(text):
    s = text.split()
    num = len([word for word in s if word in contractions])
    return num

def question_word(text):
    s = text.split()
    if s[0] in question_words:
        return 1
    else:
        return 0

def part_of_speech(text):
    s = text.split()
    nonstop = [word for word in s if word not in stop]
    pos = [part[1] for part in nltk.pos_tag(nonstop)]
    pos = ' '.join(pos)
    return pos
```

With our functions defined, let's apply them to the data.


```python
df['text']     = df['text'].apply(process_text)
df['question'] = df['text'].apply(question_word)

df['num_words']       = df['text'].apply(lambda x: len(x.split()))
df['part_speech']     = df['text'].apply(part_of_speech)
df['num_contract']    = df['text'].apply(num_contract)
df['num_stop_words']  = df['text'].apply(cnt_stop_words)
df['stop_word_ratio'] = df['num_stop_words']/df['num_words']
df['contract_ratio']  = df['num_contract']/df['num_words']
```

Here's what the data look like.


```python
                                                text  clickbait  question  \
0                                 should i get bings          1         1   
1      which tv female friend group do you belong in          1         1   
2  the new star wars the force awakens trailer is...          1         0   
3  this vine of new york on celebrity big brother...          1         1   
4  a couple did a stunning photo shoot with their...          1         0   

   num_words                   part_speech  num_contract  num_stop_words  \
0          4                        VB NNS             0               2   
1          9               NN NN NN NN VBD             0               4   
2         14  JJ NN NNS VBP NNS VBP JJ NNS             0               6   
3         12       JJ JJ NN NN JJ NN NN NN             0               4   
4         18  NN VBG NN NN NN VBG JJ NN NN             0               9   

   stop_word_ratio  contract_ratio  
0         0.500000             0.0  
1         0.444444             0.0  
2         0.428571             0.0  
3         0.333333             0.0  
4         0.500000             0.0  
```

Let's plot some of these columns.


```python
figure, axes = mp.subplots(nrows=2, ncols=2, figsize=(12,12))

plot = df.groupby('question')['clickbait'].value_counts().unstack().plot.bar(ax=axes[0,0], rot=0)
plot.set_xlabel('Headline Question')
plot.set_ylabel('Number of Headlines')

plot = df.groupby('num_words')['clickbait'].value_counts().unstack().plot.bar(ax=axes[0,1], rot=0)
plot.set_xlabel('Number of Words')
plot.set_ylabel('Number of Headlines')

plot = df.groupby('num_stop_words')['clickbait'].value_counts().unstack().plot.bar(ax=axes[1,0], rot=0)
plot.set_xlabel('Number of Stop Words')
plot.set_ylabel('Number of Headlines')

plot = df.groupby('num_contract')['clickbait'].value_counts().unstack().plot.bar(ax=axes[1,1], rot=0)
plot.set_xlabel('Number of Contractions')
plot.set_ylabel('Number of Headlines')
```




    <matplotlib.text.Text at 0x26de21908>




![png](output_9_1.png)


Well, there definitely do appear to be differences between the clickbait and non-clickbait headlines in terms of number of words, stop words, and contractions. As I suspected, clickbait headlines are also much more likely to be phrased as a question.

Let's check for collinearity in the numerical data.


```python
figure, axes = mp.subplots(figsize=(10,8))
sns.heatmap(df.drop(['text','clickbait'], axis=1).corr(), annot=True, vmax=1, linewidths=.5, cmap='Reds')
mp.xticks(rotation=45)
```




    (array([ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5]),
     <a list of 6 Text xticklabel objects>)




![png](output_11_1.png)


There does appear to be some collinearity present, especially between num_words vs. num_stop_words, between num_stop_words vs. stop_word_ratio, and between num_contract vs. contract_ratio. This makes sense. This collinearity could potentially jeopardize the accuracy of our classifier, though, and to this end, we can go ahead and drop num_stop_words and num_contract from our dataset.


```python
df.drop(['num_stop_words','num_contract'], axis=1, inplace=True)
```

Now we can split our data into training and test sets and get to classifying! We'll put 80% of the data into the training set, and 20% into the test set.


```python
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
```

To convert the headline text into the numerical data necessary for classification, we can use Scikit-Learn's TfidfVectorizer. The tf-idf is short for term frequency-inverse document frequency. In essence, tf-idf counts the number of occurrences of the words in each headline and weights these frequencies by their total number of occurrences across all headlines. The idea is that commonly occurring words like "the" and "can" appear so frequently that they are very likely to be unimportant in distinguishing between the two classes.


```python
tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                           analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),
                           use_idf=1, smooth_idf=1, sublinear_tf=1)

X_train_text = tfidf.fit_transform(df_train['text'])
X_test_text  = tfidf.transform(df_test['text'])
```

Next, we can use CountVectorizer to count the number of part-of-speech occurrences in each headline. These counts are then scaled using StandardScaler.


```python
cvec = CountVectorizer()

X_train_pos = cvec.fit_transform(df_train['part_speech'])
X_test_pos  = cvec.transform(df_test['part_speech'])

sc = StandardScaler(with_mean=False)
X_train_pos_sc = sc.fit_transform(X_train_pos)
X_test_pos_sc  = sc.transform(X_test_pos)
```

    /Users/degravek/anaconda/lib/python3.5/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, _DataConversionWarning)


We can then isolate the engineered features and scale their values so that they heave a mean of zero and unit standard deviation. This is necessary for many classifiers in order to obtain good results.


```python
X_train_val = df_train.drop(['clickbait','text','part_speech'], axis=1).values
X_test_val  = df_test.drop(['clickbait','text','part_speech'], axis=1).values

sc = StandardScaler()
X_train_val_sc = sc.fit(X_train_val).transform(X_train_val)
X_test_val_sc  = sc.transform(X_test_val)

y_train = df_train['clickbait'].values
y_test  = df_test['clickbait'].values
```

Lastly, we can combine the new tf-idf vectors with the scaled engineered features and store them as sparse arrays. This helps to save memory as the tf-idf vectors are extremely large, but are composed mostly of zeros.


```python
from scipy import sparse

X_train = sparse.hstack([X_train_val_sc, X_train_text, X_train_pos_sc])#.tocsr()
X_test  = sparse.hstack([X_test_val_sc, X_test_text, X_test_pos_sc])#.tocsr()
```

It turns out that logistic regression tends to give the best results for this classification problem, so we'll use it here. Below, we can use GridSearchCV to find the best regularization parameter for the job.


```python
param_grid = [{'C': np.linspace(90,100,20)}]

grid_cv = GridSearchCV(LogisticRegression(), param_grid, scoring='accuracy', cv=5, verbose=1)
grid_cv.fit(X_train, y_train)

print(grid_cv.best_params_)
print(grid_cv.best_score_)

Best parameter: 'C': 93.684210526315795
Best accuracry:  0.9755859375
```

Five-fold cross-validation suggests that we obtain a classification accuracy of 97.6%. Not bad at all! We can now apply the classifier to our test data to make some final predictions.


```python
model = LogisticRegression(penalty='l2', C=93.684210526315795)
model = model.fit(X_train, y_train)
predict_lr = model.predict(X_test)

print(classification_report(y_test, predict))

             precision    recall  f1-score   support

          0       0.97      0.98      0.98      3204
          1       0.98      0.97      0.98      3196

avg / total       0.98      0.98      0.98      6400
```

The classification report gives

* A precision value of 0.98
* A recall value of 0.98
* An F1-score of 0.98

Let's plot the confusion matrix.


```python
figure, axes = mp.subplots(figsize=(8,6))
cm = confusion_matrix(y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

sns.heatmap(cm, annot=True, cmap='Blues');
mp.xlabel('True Label')
mp.ylabel('Predicted Label')
```




    <matplotlib.text.Text at 0x25f507400>




![png](output_29_1.png)


The confusion matrix suggests that we misclassify clickbait headlines about twice as often as real news headlines. However, these error rates are extremely small. Now let's plot the corresponding ROC curve.


```python
figure, axes = mp.subplots(figsize=(8,8))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, predict)

mp.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))
mp.plot([0, 1], [0, 1], '--k', lw=1)
mp.xlabel('False Positive Rate')
mp.ylabel('True Positive Rate')
mp.title('ROC')
mp.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
```


![png](output_31_0.png)


As expected, the ROC curve is very good and hugs the upper left-hand corner of the figure. The area under the curve (AUC) is 0.98, suggesting near-perfect classification.

If we split the headlines into separate clickbait and non-clickbait groups, we can use tf-idf to compare which kinds of words occur more frequently among them.


```python
tfidf_cb = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                           analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3),
                           use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
cb = tfidf_cb.fit_transform(df_train.loc[df['clickbait']==1, 'text'])

tfidf_ncb = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                           analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3),
                           use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
ncb = tfidf_ncb.fit_transform(df_train.loc[df['clickbait']==0, 'text'])

cb_values = cb.mean(axis=0).tolist()
cb_names = tfidf_cb.get_feature_names()
ncb_values = ncb.mean(axis=0).tolist()
ncb_names = tfidf_ncb.get_feature_names()
```

We can place the results in two dataframes to handle them a little more easily.


```python
import itertools

q_cb = pd.DataFrame()
q_cb['names'] = cb_names
q_cb['values'] = list(itertools.chain.from_iterable(cb_values))
q_cb = q_cb.sort_values('values', ascending=True)

q_ncb = pd.DataFrame()
q_ncb['names'] = ncb_names
q_ncb['values'] = list(itertools.chain.from_iterable(ncb_values))
q_ncb = q_ncb.sort_values('values', ascending=True)
```

And plot the results.


```python
figure, axes = mp.subplots(nrows=1, ncols=2, figsize=(12,7))
mp.tight_layout(8,1)

plot = q_cb[-20:].plot.barh(x='names', y='values', ax=axes[0], rot=0)
plot.set_xlabel('Mean tf-idf')
plot.set_ylabel('Words')
plot.set_title('Clickbait')

plot = q_ncb[-20:].plot.barh(x='names', y='values', ax=axes[1], rot=0)
plot.set_xlabel('Mean tf-idf')
plot.set_ylabel('Words')
plot.set_title('Non-Clickbait')
```




    <matplotlib.text.Text at 0x261aee2b0>




![png](output_37_1.png)


As we might expect, the words are quite different between the two sets. Clickbait headlines tend to contain words like "best", "character", "love", etc., while the non-clickbait headlines contain more serious subject matter like "president", "iraq", and "government". Also, note that one of the top clickbait features is "number things", representing headlines like "10 things you'll never believe!".

Let's take a look at some of the headlines we misclassified.


```python
df_test['predict_lr'] = predict_lr
df_test['predict_nn'] = predict_nn

df_test.loc[df_test['predict_lr'] != df_test['predict_nn'], ['text','clickbait','predict_lr','predict_nn']]
```

    /Users/degravek/anaconda/lib/python3.5/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    /Users/degravek/anaconda/lib/python3.5/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>clickbait</th>
      <th>predict_lr</th>
      <th>predict_nn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4046</th>
      <td>hear a clip of good charlottes comeback song</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21709</th>
      <td>number pieces of luggage found behind texas pe...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7174</th>
      <td>we can thank unbreakable kimmy schmidt for piz...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2482</th>
      <td>arjun kapoor was in neerja and no one noticed</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12222</th>
      <td>fyi beans from even stevens is working at a ma...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12275</th>
      <td>baconwrapped sriracha onion rings</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8339</th>
      <td>billy corgan named his son augustus juppiter</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3718</th>
      <td>hot chocolate holiday mixes</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18174</th>
      <td>for now the jets are going with an emptier bac...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>this organisation is trying to launch indias f...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15422</th>
      <td>number texts fetuses wish they could send</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14266</th>
      <td>psa get to japan and visit the worlds biggest ...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7583</th>
      <td>a closer look into the world of indias dog pag...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14198</th>
      <td>bfwknd</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4390</th>
      <td>cara delevingne hits back at robb stark after ...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16071</th>
      <td>to get a business loan know how the bank thinks</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18238</th>
      <td>nigeria bill has taylor implications</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19053</th>
      <td>in books on two powerbrokers hints of the future</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27111</th>
      <td>new yorker cover art is painted with an iphone</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21328</th>
      <td>no shortage of advice on mideast for clinton</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18354</th>
      <td>sheep leap to their deaths in turkey</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8315</th>
      <td>for anyone who experiences regret after ejacul...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10628</th>
      <td>awkward selfie stick prank</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>321</th>
      <td>in memoriam shia labeoufs beautiful gorgeous r...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10262</th>
      <td>whales were caught on camera under the norther...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11839</th>
      <td>the cookie heist</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11510</th>
      <td>barbies new collection has curvy petite and ta...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5886</th>
      <td>mcdonalds pies around the world</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10487</th>
      <td>meet the blaxicans of los angeles</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9645</th>
      <td>goodbye americas next top model</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19133</th>
      <td>canadian woman faces number counts of doubledo...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8202</th>
      <td>the triumphant melancholic return of the world...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>bad driving school</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24184</th>
      <td>speedskating provided marsicano a refuge from ...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6016</th>
      <td>empire confronted racism with an homage to cla...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17331</th>
      <td>baby in california born with number functionin...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31855</th>
      <td>how three mutual funds steered away from the c...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8979</th>
      <td>first allegiant trailer still has tris and fou...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13963</th>
      <td>lesbian bride of frankenstein</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14253</th>
      <td>prince harrys ginger beard deserves a damn award</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31568</th>
      <td>male models win the amazing race</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19976</th>
      <td>madoffs shared much but how much</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18415</th>
      <td>woman number raped in her own home in uk</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25729</th>
      <td>in the wrong job monstercom wants you</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26017</th>
      <td>india subsidizes girls education to offset gen...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13660</th>
      <td>recovering addicts talk about the stigmas surr...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4672</th>
      <td>everybody hates mark from matchcom</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22663</th>
      <td>mars rover doing well after memory glitch</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2586</th>
      <td>instafamous the eyebrow queen</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3651</th>
      <td>number moving photos from vigils for paris hel...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17795</th>
      <td>suspicious package closes half of washington d...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5960</th>
      <td>proof that national treasure shia labeouf tran...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18156</th>
      <td>the voiceactivated ilane sorts mail as you drive</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5821</th>
      <td>kris jenner swims around in a pool to trap que...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24412</th>
      <td>straight talk or unhelpful scolding</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14452</th>
      <td>john krasinskis view on fatherhood will explod...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11677</th>
      <td>watch years of womens lingerie in three minutes</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17527</th>
      <td>fans mourn death and celebrate life of michael...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27812</th>
      <td>in reno nev art canvases that include shag carpet</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26328</th>
      <td>how safe is that mutual fund next egg anyhow</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>122 rows × 4 columns</p>
</div>




```python
                                                    text  clickbait  predict
12060               bruno mars might headline super bowl          1        0
13695  priest reportedly suspended for riding a hover...          1        0
15652                     night owls become early risers          1        0
759    number differences between snow days in canada...          1        0
4046        hear a clip of good charlottes comeback song          1        0
```

There actually appear to be a few cases where a headline was deemed to be clickbait, though I personally might not have classified it as such. For example, the headlines "bruno mars might headline superbowl", and "night owls become early risers" were labeled as clickbait by the collectors of this dataset, though these could possibly constitute legitimate news headlines (albeit maybe poor ones).

# Concluding Remarks
In this notebook we used logistic regression and natural language processing to accurately classify clickbait headlines. The results were surprisingly good, as we obtained an accuracy score of 97.6%, an F1-score of 0.98, and an ROC AUC of 0.98. This classifier actually performs better than that of Chakraborty et al., in which an accuracy score of 93% was achieved. An interesting next step would be to use this classifier to deploy a real clickbait detection extension for a web browser like Firefox or Safari. This was actually done in the Chakraborty et al. paper, where an accuracy of 83% in blocking clickbaits was attained.

Well, that’s all I have for now. Thanks for following along!


```python
https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py
```




    <25600x25146 sparse matrix of type '<class 'numpy.float64'>'
    	with 543039 stored elements in Compressed Sparse Row format>




```python
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train.toarray(), y_train, epochs=20, batch_size=128)
```

    Epoch 1/20
    25600/25600 [==============================] - 16s - loss: 0.2696 - acc: 0.9060    
    Epoch 2/20
    25600/25600 [==============================] - 14s - loss: 0.0772 - acc: 0.9746    
    Epoch 3/20
    25600/25600 [==============================] - 15s - loss: 0.0501 - acc: 0.9826    
    Epoch 4/20
    25600/25600 [==============================] - 15s - loss: 0.0391 - acc: 0.9864    
    Epoch 5/20
    25600/25600 [==============================] - 15s - loss: 0.0323 - acc: 0.9886    
    Epoch 6/20
    25600/25600 [==============================] - 17s - loss: 0.0274 - acc: 0.9905    
    Epoch 7/20
    25600/25600 [==============================] - 15s - loss: 0.0245 - acc: 0.9918    
    Epoch 8/20
    25600/25600 [==============================] - 17s - loss: 0.0216 - acc: 0.9926    
    Epoch 9/20
    25600/25600 [==============================] - 13s - loss: 0.0189 - acc: 0.9936    
    Epoch 10/20
    25600/25600 [==============================] - 13s - loss: 0.0167 - acc: 0.9938    
    Epoch 11/20
    25600/25600 [==============================] - 13s - loss: 0.0152 - acc: 0.9950    
    Epoch 12/20
    25600/25600 [==============================] - 12s - loss: 0.0137 - acc: 0.9957    
    Epoch 13/20
    25600/25600 [==============================] - 14s - loss: 0.0120 - acc: 0.9960    
    Epoch 14/20
    25600/25600 [==============================] - 13s - loss: 0.0109 - acc: 0.9964    
    Epoch 15/20
    25600/25600 [==============================] - 13s - loss: 0.0092 - acc: 0.9966    
    Epoch 16/20
    25600/25600 [==============================] - 14s - loss: 0.0083 - acc: 0.9973    
    Epoch 17/20
    25600/25600 [==============================] - 13s - loss: 0.0083 - acc: 0.9975    
    Epoch 18/20
    25600/25600 [==============================] - 13s - loss: 0.0067 - acc: 0.9980    
    Epoch 19/20
    25600/25600 [==============================] - 12s - loss: 0.0063 - acc: 0.9983    
    Epoch 20/20
    25600/25600 [==============================] - 13s - loss: 0.0060 - acc: 0.9983    





    <keras.callbacks.History at 0x146610860>




```python
predict_nn = model.predict_classes(X_test.toarray(), batch_size=128)
score = model.evaluate(X_test.toarray(), y_test, batch_size=128)
```

    6400/6400 [==============================] - 2s     
    6272/6400 [============================>.] - ETA: 0s


```python
score
```




    [0.11483846684568562, 0.98046875]




```python
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html

plot = figure()
plot.circle([1,2], [3,4])

html = file_html(plot, CDN, "my plot")
```

<meta charset="utf-8">
<title>my plot</title>
    
<link rel="stylesheet" href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css" type="text/css" />
<script type="text/javascript" src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
<script type="text/javascript">
    Bokeh.set_log_level("info");
</script>

<script type="text/javascript">
    Bokeh.$(function() {
        var modelid = "27663170-c43e-4dea-a1bc-747229c2084e";
        var modeltype = "Plot";
        var elementid = "b201e9fd-4166-4aa8-8963-0ce26eeb8e73";
        Bokeh.logger.info("Realizing plot:")
        Bokeh.logger.info(" - modeltype: Plot");
        Bokeh.logger.info(" - modelid: 27663170-c43e-4dea-a1bc-747229c2084e");
        Bokeh.logger.info(" - elementid: b201e9fd-4166-4aa8-8963-0ce26eeb8e73");
        var all_models = [{"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e", "attributes": {"x_range": {"type": "DataRange1d", "id": "4839dfaa-3854-4e15-a204-ee072537d72b"}, "right": [], "tags": [], "y_range": {"type": "DataRange1d", "id": "6e07d07d-774a-4068-a104-6e8280e1bd33"}, "renderers": [{"type": "LinearAxis", "id": "5e8c9263-47d7-4d30-9e88-abfb7adb0c93"}, {"type": "Grid", "id": "8606d4ae-e56a-4fa9-ab6e-38a4a6a37098"}, {"type": "LinearAxis", "id": "98819d43-35a5-4c0c-8cdb-667d1f3603cd"}, {"type": "Grid", "id": "0551741c-d3e3-4395-81eb-d8c0190ede3e"}, {"type": "GlyphRenderer", "id": "d8c43984-012f-46b9-8f34-972a83b0ad2f"}], "extra_y_ranges": {}, "extra_x_ranges": {}, "tool_events": {"type": "ToolEvents", "id": "2e8e2bee-6830-45c7-bd6f-9166f5777505"}, "above": [], "doc": null, "id": "27663170-c43e-4dea-a1bc-747229c2084e", "tools": [{"type": "PanTool", "id": "e89e5427-e24e-47b1-bbaa-3aeb4321a9db"}, {"type": "WheelZoomTool", "id": "a37cf199-2570-4c07-9913-da0c5679d023"}, {"type": "BoxZoomTool", "id": "f431de3b-3585-424b-a698-cf3084b897df"}, {"type": "PreviewSaveTool", "id": "93383792-4ecc-4fc9-b735-b87781eae780"}, {"type": "ResizeTool", "id": "674735fd-0c97-43e8-8d1b-8f3f59f5fe42"}, {"type": "ResetTool", "id": "cd488e5c-076e-4027-b087-1dcdc73b6d10"}, {"type": "HelpTool", "id": "0a4e5027-8af4-4bbe-abb0-90caa5e09b44"}], "below": [{"type": "LinearAxis", "id": "5e8c9263-47d7-4d30-9e88-abfb7adb0c93"}], "left": [{"type": "LinearAxis", "id": "98819d43-35a5-4c0c-8cdb-667d1f3603cd"}]}}, {"attributes": {"names": [], "tags": [], "doc": null, "id": "4839dfaa-3854-4e15-a204-ee072537d72b", "renderers": []}, "type": "DataRange1d", "id": "4839dfaa-3854-4e15-a204-ee072537d72b"}, {"attributes": {"column_names": ["x", "y"], "tags": [], "doc": null, "selected": {"2d": {"indices": []}, "1d": {"indices": []}, "0d": {"indices": [], "flag": false}}, "callback": null, "data": {"y": [3, 4], "x": [1, 2]}, "id": "72265f1e-fb4d-4c3d-8fbe-559ad913936c"}, "type": "ColumnDataSource", "id": "72265f1e-fb4d-4c3d-8fbe-559ad913936c"}, {"attributes": {"names": [], "tags": [], "doc": null, "id": "6e07d07d-774a-4068-a104-6e8280e1bd33", "renderers": []}, "type": "DataRange1d", "id": "6e07d07d-774a-4068-a104-6e8280e1bd33"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "formatter": {"type": "BasicTickFormatter", "id": "63b8bab9-a211-4a08-8703-a22f51bfcd0f"}, "ticker": {"type": "BasicTicker", "id": "e3ca1f42-4452-4cb2-b3b4-87db30ba3448"}, "id": "5e8c9263-47d7-4d30-9e88-abfb7adb0c93"}, "type": "LinearAxis", "id": "5e8c9263-47d7-4d30-9e88-abfb7adb0c93"}, {"attributes": {"tags": [], "doc": null, "mantissas": [2, 5, 10], "id": "e3ca1f42-4452-4cb2-b3b4-87db30ba3448", "num_minor_ticks": 5}, "type": "BasicTicker", "id": "e3ca1f42-4452-4cb2-b3b4-87db30ba3448"}, {"attributes": {"line_color": {"value": "#1f77b4"}, "line_alpha": {"value": 1.0}, "fill_color": {"value": "#1f77b4"}, "tags": [], "doc": null, "fill_alpha": {"value": 1.0}, "y": {"field": "y"}, "x": {"field": "x"}, "id": "8d15ed42-6088-40fc-8624-3ae69eedf1e2"}, "type": "Circle", "id": "8d15ed42-6088-40fc-8624-3ae69eedf1e2"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "dimension": 0, "ticker": {"type": "BasicTicker", "id": "e3ca1f42-4452-4cb2-b3b4-87db30ba3448"}, "id": "8606d4ae-e56a-4fa9-ab6e-38a4a6a37098"}, "type": "Grid", "id": "8606d4ae-e56a-4fa9-ab6e-38a4a6a37098"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "formatter": {"type": "BasicTickFormatter", "id": "b17c0304-1823-48c6-be15-f33ffc0fbecb"}, "ticker": {"type": "BasicTicker", "id": "628c864f-3b51-4815-bd0d-2050741e55b4"}, "id": "98819d43-35a5-4c0c-8cdb-667d1f3603cd"}, "type": "LinearAxis", "id": "98819d43-35a5-4c0c-8cdb-667d1f3603cd"}, {"attributes": {"tags": [], "doc": null, "mantissas": [2, 5, 10], "id": "628c864f-3b51-4815-bd0d-2050741e55b4", "num_minor_ticks": 5}, "type": "BasicTicker", "id": "628c864f-3b51-4815-bd0d-2050741e55b4"}, {"attributes": {"doc": null, "id": "b17c0304-1823-48c6-be15-f33ffc0fbecb", "tags": []}, "type": "BasicTickFormatter", "id": "b17c0304-1823-48c6-be15-f33ffc0fbecb"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "dimension": 1, "ticker": {"type": "BasicTicker", "id": "628c864f-3b51-4815-bd0d-2050741e55b4"}, "id": "0551741c-d3e3-4395-81eb-d8c0190ede3e"}, "type": "Grid", "id": "0551741c-d3e3-4395-81eb-d8c0190ede3e"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "dimensions": ["width", "height"], "tags": [], "doc": null, "id": "e89e5427-e24e-47b1-bbaa-3aeb4321a9db"}, "type": "PanTool", "id": "e89e5427-e24e-47b1-bbaa-3aeb4321a9db"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "dimensions": ["width", "height"], "tags": [], "doc": null, "id": "a37cf199-2570-4c07-9913-da0c5679d023"}, "type": "WheelZoomTool", "id": "a37cf199-2570-4c07-9913-da0c5679d023"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "dimensions": ["width", "height"], "tags": [], "doc": null, "id": "f431de3b-3585-424b-a698-cf3084b897df"}, "type": "BoxZoomTool", "id": "f431de3b-3585-424b-a698-cf3084b897df"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "id": "93383792-4ecc-4fc9-b735-b87781eae780"}, "type": "PreviewSaveTool", "id": "93383792-4ecc-4fc9-b735-b87781eae780"}, {"attributes": {"nonselection_glyph": {"type": "Circle", "id": "8862d1e2-f5a3-468d-971b-c58701a194c4"}, "data_source": {"type": "ColumnDataSource", "id": "72265f1e-fb4d-4c3d-8fbe-559ad913936c"}, "name": null, "tags": [], "doc": null, "selection_glyph": null, "id": "d8c43984-012f-46b9-8f34-972a83b0ad2f", "glyph": {"type": "Circle", "id": "8d15ed42-6088-40fc-8624-3ae69eedf1e2"}}, "type": "GlyphRenderer", "id": "d8c43984-012f-46b9-8f34-972a83b0ad2f"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "id": "674735fd-0c97-43e8-8d1b-8f3f59f5fe42"}, "type": "ResizeTool", "id": "674735fd-0c97-43e8-8d1b-8f3f59f5fe42"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "id": "cd488e5c-076e-4027-b087-1dcdc73b6d10"}, "type": "ResetTool", "id": "cd488e5c-076e-4027-b087-1dcdc73b6d10"}, {"attributes": {"plot": {"subtype": "Figure", "type": "Plot", "id": "27663170-c43e-4dea-a1bc-747229c2084e"}, "tags": [], "doc": null, "id": "0a4e5027-8af4-4bbe-abb0-90caa5e09b44"}, "type": "HelpTool", "id": "0a4e5027-8af4-4bbe-abb0-90caa5e09b44"}, {"attributes": {"geometries": [], "tags": [], "doc": null, "id": "2e8e2bee-6830-45c7-bd6f-9166f5777505"}, "type": "ToolEvents", "id": "2e8e2bee-6830-45c7-bd6f-9166f5777505"}, {"attributes": {"doc": null, "id": "63b8bab9-a211-4a08-8703-a22f51bfcd0f", "tags": []}, "type": "BasicTickFormatter", "id": "63b8bab9-a211-4a08-8703-a22f51bfcd0f"}, {"attributes": {"line_color": {"value": "#1f77b4"}, "line_alpha": {"value": 0.1}, "fill_color": {"value": "#1f77b4"}, "tags": [], "doc": null, "fill_alpha": {"value": 0.1}, "y": {"field": "y"}, "x": {"field": "x"}, "id": "8862d1e2-f5a3-468d-971b-c58701a194c4"}, "type": "Circle", "id": "8862d1e2-f5a3-468d-971b-c58701a194c4"}];
        Bokeh.load_models(all_models);
        var model = Bokeh.Collections(modeltype).get(modelid);
        var view = new model.default_view({model: model, el: '#b201e9fd-4166-4aa8-8963-0ce26eeb8e73'});
        Bokeh.index[modelid] = view
    });
</script>

<div class="plotdiv" id="b201e9fd-4166-4aa8-8963-0ce26eeb8e73"></div>


```python

```
