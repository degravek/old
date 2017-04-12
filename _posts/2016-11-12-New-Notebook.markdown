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


# Introduction
The Federal National Mortgage Association (FNMA), also known as Fannie Mae, is a government sponsored corporation founded in 1938 whose primary purpose, according to [this source](https://en.wikipedia.org/wiki/Fannie_Mae), is "to expand the secondary mortgage market by securitizing mortgages in the form of mortgage-backed securities, allowing lenders to reinvest their assets into more lending and in effect increasing the number of lenders in the mortgage market by reducing the reliance on locally based savings and loan associations." In short, Fannie Mae purchases mortgage loans from primary lenders like Bank of America and Wells Fargo, among several others. After these mortgages are acquired, Fannie Mae sells them as securities in the bond market. According to [this source](http://home.howstuffworks.com/real-estate/buying-home/mortgage16.htm), these sales "provide lenders with the liquidity to fund more mortgages, and until 2006, the mortgage-backed securities (MBS) sold by [Fannie Mae] were considered solid investments." Unfortunately, however, not all borrowers whose loans have been purchased by Fannie Mae are able to repay their mortgages in a timely manner, and many end up defaulting at some point. In fact, between 2006 and 2008, many hundreds of thousands of people had defaulted, causing these securities to decreases significantly in value, thereby strongly impacting the global economy.

On its website, Fannie Mae has made a subset of its single family loan performance (SFLP) data available to anyone interested in looking at it. The SFLP data cover the years 2000-2015, and can be downloaded [here](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html). The goal of this project it so see if we can predict from this data, with some accuracy, those borrowers who are most at risk of defaulting on their mortgage loans. Let's get started!

# Cleaning the Data
Once downloaded, one will find that the SFLP data is divided into two files called Acquisition^.txt and Performance^.txt, where the "^" is a placeholder for the particular year and quarter of interest. For the purposes of this project, we're using the quarter 4 data of 2007 which contains a reasonable number of defaults to analyze. The aquisition data contains personal information for each of the borrowers, including an individual's debt-to-income ratio, credit score, and loan amount, among several other things. The perfomance data contains information regarding loan payment history, and whether or not a borrower ended up defaulting on their loan. Additional information regarding the contents of these two files can be found in the [Layout](https://loanperformancedata.fanniemae.com/lppub-docs/lppub_file_layout.pdf) and [Glossary of Terms](https://loanperformancedata.fanniemae.com/lppub-docs/lppub_glossary.pdf) files.

Let’s begin by importing the appropriate Python libraries and reading in the data.


```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as mp
%matplotlib inline

col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
        'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
        'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
        'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd'];

col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
          'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
          'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
          'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
          'FPWA','ServicingIndicator'];

df_acq = pd.read_csv('/Users/degravek/Downloads/2007Q4/Acquisition_2007Q4.txt', sep='|', names=col_acq, index_col=False)
df_per = pd.read_csv('/Users/degravek/Downloads/2007Q4/Performance_2007Q4.txt', sep='|', names=col_per, usecols=[0, 15], index_col=False)
```

In the performance data, we are really only interested in the LoanID and ForeclosureDate columns, as this will give us the borrower identifiaction number and whether or not they ended up defaulting. After reading in the two datasets, we can perform an inner join on the acquisition and performance dataframes using the LoanID column. The resulting dataframe, df, will contain the ForeclosureDate column, and will be our target variable. For clarity, we will also rename this column as Default.


```python
df_per.drop_duplicates(subset='LoanID', keep='last', inplace=True)
df = pd.merge(df_acq, df_per, on='LoanID', how='inner')

df.rename(index=str, columns={"ForeclosureDate": 'Default'}, inplace=True)
```

In the Default column, a 1 is placed next to any borrower that was found to have defaulted, and a 0 is placed next to any borrower that has not defaulted.


```python
df['Default'].fillna(0, inplace=True)
df.loc[df['Default'] != 0, 'Default'] = 1

df['Default'] = df['Default'].astype(int)
```

Let's take a look at the dataframe head.


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanID</th>
      <th>Channel</th>
      <th>SellerName</th>
      <th>OrInterestRate</th>
      <th>OrUnpaidPrinc</th>
      <th>OrLoanTerm</th>
      <th>OrDate</th>
      <th>FirstPayment</th>
      <th>OrLTV</th>
      <th>OrCLTV</th>
      <th>...</th>
      <th>NumUnits</th>
      <th>OccStatus</th>
      <th>PropertyState</th>
      <th>Zip</th>
      <th>MortInsPerc</th>
      <th>ProductType</th>
      <th>CoCreditScore</th>
      <th>MortInsType</th>
      <th>RelMortInd</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002296854</td>
      <td>C</td>
      <td>BANK OF AMERICA, N.A.</td>
      <td>6.625</td>
      <td>343000</td>
      <td>360</td>
      <td>10/2007</td>
      <td>12/2007</td>
      <td>86</td>
      <td>86.0</td>
      <td>...</td>
      <td>1</td>
      <td>P</td>
      <td>CO</td>
      <td>809</td>
      <td>25.0</td>
      <td>FRM</td>
      <td>756.0</td>
      <td>2.0</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100006876815</td>
      <td>C</td>
      <td>BANK OF AMERICA, N.A.</td>
      <td>6.250</td>
      <td>400000</td>
      <td>360</td>
      <td>10/2007</td>
      <td>12/2007</td>
      <td>62</td>
      <td>62.0</td>
      <td>...</td>
      <td>1</td>
      <td>P</td>
      <td>CA</td>
      <td>920</td>
      <td>NaN</td>
      <td>FRM</td>
      <td>790.0</td>
      <td>NaN</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100008184591</td>
      <td>B</td>
      <td>OTHER</td>
      <td>6.625</td>
      <td>81000</td>
      <td>360</td>
      <td>11/2007</td>
      <td>01/2008</td>
      <td>64</td>
      <td>82.0</td>
      <td>...</td>
      <td>1</td>
      <td>P</td>
      <td>LA</td>
      <td>708</td>
      <td>NaN</td>
      <td>FRM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100008870761</td>
      <td>B</td>
      <td>AMTRUST BANK</td>
      <td>6.500</td>
      <td>119000</td>
      <td>360</td>
      <td>11/2007</td>
      <td>01/2008</td>
      <td>71</td>
      <td>71.0</td>
      <td>...</td>
      <td>1</td>
      <td>P</td>
      <td>IL</td>
      <td>600</td>
      <td>NaN</td>
      <td>FRM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100013284158</td>
      <td>B</td>
      <td>BANK OF AMERICA, N.A.</td>
      <td>6.625</td>
      <td>205000</td>
      <td>360</td>
      <td>10/2007</td>
      <td>12/2007</td>
      <td>27</td>
      <td>27.0</td>
      <td>...</td>
      <td>1</td>
      <td>P</td>
      <td>CA</td>
      <td>907</td>
      <td>NaN</td>
      <td>FRM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



The dataframe has 340,516 rows and 26 columns, and contains information regarding loan interest rate, payment dates, property state, and the last few digits of each property ZIP code, among several other things. Many of the columns contain missing values, and these will have to be filled in before we start making our predictions. Let's see how many null values are in each column.


```python
df.apply(lambda x: x.isnull().sum(), axis=0)

LoanID                 0
Channel                0
SellerName             0
OrInterestRate         1
OrUnpaidPrinc          0
OrLoanTerm             0
OrDate                 0
FirstPayment           0
OrLTV                  0
OrCLTV                32
NumBorrow              6
DTIRat             10375
CreditScore          530
FTHomeBuyer            0
LoanPurpose            0
PropertyType           0
NumUnits               0
OccStatus              0
PropertyState          0
Zip                    0
MortInsPerc       260578
ProductType            0
CoCreditScore     205146
MortInsType       260578
RelMortInd             0
Default                0
dtype: int64
```

There appears to be eight data columns that contain at least one missing value. These can be handled in a number of ways; depending on the distribution of data in each column, we can fill in missing values with the column median or mean, or we could sample randomly from a distribution defined by the present values. We could also fit for the missing values using a machine learning algorithm applied to the complete columns, or we could drop the missing data altogether. Columns "OrCLTV", "NumBorrow", "CreditScore", and "OrInterestRate" don't contain too many missing values, and, since we have a lot of data to work with, we could simply drop those particular rows from the analysis with little impact on the final results. However, we'll still try and fill those in later just for fun.

Before filling in missing values, let's first take a quick look at the distribution of values in several of the data columns. We can start with our target variable, Default.


```python
sns.countplot(df['Default'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116fdad68>




![png](output_12_1.png)


The two classes (default = 1 and non-default = 0) are extremely imbalanced here; defaulters make up only about 10% of all borrowers in this particular dataset. For very imbalanced data sets, it is often the case that machine learning algorithms will have a tendency to always predict the more dominant class when presented with new, unseen test data. To avoid an overabundance of false negatives, we can eventually balance the classes so that the dataframe contains equal numbers of defaulters and non-defaulters. However, let's continue looking at some more of the data first.


```python
columns = ['OrCLTV','DTIRat','CreditScore','OrInterestRate']

fig, axes = mp.subplots(nrows=2, ncols=2, figsize=(6,7))
mp.tight_layout(w_pad=2.0, h_pad=3.0)

for i, column in zip(range(1,5), columns):
    mp.subplot(2,2,i)
    sns.boxplot(x="Default", y=column, data=df, linewidth=0.5)
    mp.xlabel('Default')
```


![png](output_14_0.png)


The figures above show boxplots for several columns in our dataset. The green boxes (and whiskers) show the distribution of values spanned by the default class, while the blue boxes show the values spanned by the non-default class. Boxplots are assembled such that 25% of the data values are contained between the lowest whisker and the bottom of the box, 50% of the values are contained within the box itself, and 25% of the values are spanned between the top of the box and the top whisker. The median value of the data is represented by the horizontal line in the middle of each box. The figures show that on average, defaulters have a higher debt-to-income ratio than do non-defaulters, lower credit scores, and higher interest rates. Interestingly, in looking at the various data features, the borrower's location (ZIP code) also seems to be a possible indicator of whether or not a default will occur. The figure below shows the fraction of people that have defaulted from the ten most common ZIP codes having more than 500 borrowers. Comparing certain locations (for example, ZIP code 853 vs. 750), there are significant differences in the fraction of borrowers that defaulted. We will see shortly that the values represented in these figures are some of the most discriptive features in terms of identifying which class a borrower belongs to.


```python
data = df.loc[df['Zip'].isin(df['Zip'].value_counts().index.tolist()[:10])]

xtab = pd.pivot_table(data, index='Zip', columns='ForeclosureDate', aggfunc='size')
xtab = xtab.div(ptab.sum(axis=1), axis=0)
xtab.plot.barh(stacked=True, figsize=(6,4))
mp.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
mp.xlabel('Fraction of Borrowers')
mp.ylabel('ZIP Code')
```




    <matplotlib.text.Text at 0x13381e550>




![png](output_16_1.png)


We can perform a potentially important pre-processing step and split up any date columns into their month and year components, just in case they might have some predictive power later.


```python
df['OrDateMonth'] = df['OrDate'].apply(lambda x: x.split('/')[0].strip()).astype(int)
df['OrDateYear'] = df['OrDate'].apply(lambda x: x.split('/')[1].strip()).astype(int)

df['FirstMonth'] = df['FirstPayment'].apply(lambda x: x.split('/')[0].strip()).astype(int)
df['FirstYear'] = df['FirstPayment'].apply(lambda x: x.split('/')[1].strip()).astype(int)

df.drop(['OrDate','FirstPayment'], axis=1, inplace=True)
```

Finally, before going on let's drop a few columns from our dataframe. These include the columns with many tens of thousands of missing values (MortInsPerc, MortInsType, CoCreditScore), the ProductType column as it contains only one unique value, and the LoanID column.


```python
df.drop(['MortInsPerc','MortInsType','CoCreditScore','ProductType','LoanID'], axis=1, inplace=True)
```

Let's define a function to get dummy variables for the categorical columns having data type 'object'.


```python
def getdummies(df):
    columns = df.columns[df.isnull().any()]
    nan_cols = df[columns]

    df.drop(nan_cols.columns, axis=1, inplace=True)

    cat = df.select_dtypes(include=['object'])
    num = df.drop(cat.columns, axis=1)

    data = pd.DataFrame()
    for i in cat.columns:
        tmp = pd.get_dummies(cat[i], drop_first=True)
        data = pd.concat([data, tmp], axis=1)

    df = pd.concat([num,data,nan_cols], axis=1).reset_index(drop=True)
    return df
```

Okay, now we're ready to fill in some missing values! Rather than simply using the column mean, median, etc., let's do something more complicated and fit for the missing values using a random forest regressor (or classifier, depending on the column data type). We can define a function to loop over columns with missing values.


```python
def fillnan(df):
    columns = df.columns[df.isnull().any()]
    for name in columns:
        y = df.loc[df[name].notnull(), name].values
        X = df.loc[df[name].notnull()].drop(columns, axis=1).values
        X_test = df.loc[df[name].isnull()].drop(columns, axis=1).values
        if df[name].dtypes == 'object':
            model = RandomForestClassifier(n_estimators=400, max_depth=3)
            model.fit(X, y)
            df.loc[df[name].isnull(), name] = model.predict(X_test)
        else:
            model = RandomForestRegressor(n_estimators=400, max_depth=3)
            model.fit(X, y)
            df.loc[df[name].isnull(), name] = model.predict(X_test)
    return df
```

Let's call those functions.


```python
df = getdummies(df)
df = fillnan(df)
```

Okay, before we start predicting defaults, let's balance the classes. To do this, I'll use the Synthetic Minority Oversampling Technique (SMOTE). Rather than simply oversampling the the minority class (using repeated copies of the same data) or undersampling the dominant class, we can actually do both simultaneously while creating "new" instances of the minority class.


```python
from imblearn.combine import SMOTEENN
sm = SMOTEENN()

y = df['Default'].values
X = df.drop(['Default'], axis=1).values

X_resampled, y_resampled = sm.fit_sample(X, y)
```

# Predicting Bad Loans
Alright, now we're ready to make some predictions! We first randomly split the data into a training set and a test set using the Scikit-Learn train_test_split_function. From these two sets, we idenfiy the target ("Default") vector, and feature arrays. We then initialize a random forest classifier composed of 200 random decision trees, fit it to the training data, and then predict the test set classes.


```python
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.25, random_state=0)

model = RandomForestClassifier(n_estimators=200)
model = model.fit(X_train, y_train)
predict = model.predict(X_test)
```

We can evaluate the perfomance of our model by examining the resulting classification report, which contains details regarding the classifier's precision, recall, and F-score.


```python
print(classification_report(y_test, predict))

             precision    recall  f1-score   support

          0       0.92      1.00      0.96     76778
          1       1.00      0.90      0.94     65766

avg / total       0.95      0.95      0.95    142544
```

Let's also look at the confusion matrix. The confusion matrix is a table which shows the percentage of correct (true positives or true negatives) and incorrect (false positives or false negatives) classifications for each positive (default) or negative (non-default) class. In the table below, the true class is given along the x-axis, while the predicted class is given along the y-axis. Graphically, this looks like:


```python
cm = confusion_matrix(y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

fig, ax = mp.subplots()
sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.xaxis.set_label_position('top')
```


![png](output_34_0.png)


The confusion matrix shows that for all non-defaulters in our dataset, the algorithm correlectly identifies them as non-defaulters nearly 100% of the time (these are true negatives), and incorrectly labels them as defaulters only 0.3% of the time (these are false positives). Similarly, for all of the defaulters in our dataset, we are able to correctly identify them 90% of the time (these are true positives), while our algorithm incorrectly misidentifies them 10% of the time (these are false negatives). In terms of profitability to Fannie Mae, false negatives are the most important metric here. This is because Fannie Mae loses money when we incorrectly label a defaulter as being a non-defaulter. The fact that we incorrectly classify some of our non-defaulters is of little consequence, though, because there are so many of them present in the full data set (i.e., we can always find more non-defaulters easily enough).

One may point out that both our training and test sets have been balanced before analysis, and wonder if this predictive capability holds up when the algorithm is presented with new, very imbalanced data. It turns out that this is in fact still the case. Some additional testing suggests that rates of false positives and false negatives are nearly identical to those given above.

To further visualize the performance of our classifier, we can look at the corresponding receiver operating characteristics (ROC) curve. The ROC curve shows the number of true positives vs. the number of false positives labeled by the algorithm for a number of classification threshold values.


```python
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, predict)

mp.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))
mp.plot([0, 1], [0, 1], '--k', lw=1)
mp.xlabel('False Positive Rate')
mp.ylabel('True Positive Rate')
mp.title('Random Forest ROC')
mp.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
```


![png](output_36_0.png)


In the case of a perfect classifier, the ROC curve would hug the top left corver of the figure (the true positive rate would be 1.0, and the false positive rate would be 0.0). The black dashed curve represents a classifier with no predictive power. We see that in our case, the random forest does a very good job; it clearly has predictive capabilities, with an area under the curve (AUC) of 0.95.

The random forest classifier is nice in that it allows one to identify directly those features in the dataframe that were most important in predicting the positive and negative classes. Let's take a look at the top 20 most important features.


```python
feat_labels = df.drop('Default', axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

ncomp = 20
sns.barplot(x=feat_labels[indices[:ncomp]], y=importances[indices[:ncomp]], color=sns.xkcd_rgb["pale red"])
mp.title('Top 20 Feature Importances')
mp.ylabel('Relative Feature Importance')
mp.xticks(rotation=90)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17]), <a list of 18 Text xticklabel objects>)




![png](output_38_1.png)


As we saw earlier, of all the available features, it looks like borrower credit score, ZIP code, and debt-to-income ratio are among the most predictive, though the number of borrowers, loan servicer, and interest rate appear to be very important as well. This sort of analysis of feature importances would be useful for dimensionality reduction if we had many hundreds or thousands of features in our dataframe.

# Concluding Remarks
In this project, we've detailed how to predict bad loans Fannie Mae single family loan performance data. The random forest classifier gave us a nice baseline algorithm by which we could identify loan defaulters with very good accuracy, precision, and recall. The resulting ROC AUC was 0.95.

A number of tests could be conducted to try and further improve the analysis. For example, one could find the optimal number of estimators (trees) to use in the initial random forest classification. A value of 200 was shown to perform quite well, but could be tuned to give an even better performance. We could also compare a number of different tuned algorithms like logistic regression or k-nearest neighbors to see how these perform relative to the algorithm used in this work.

Well, that's all I have for now. Thanks for following along!
