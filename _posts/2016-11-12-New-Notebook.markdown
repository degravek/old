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

```python
PlotReviewBars(sorted_ngram1, sorted_ngram2, sorted_ngram3, 15)
```


<div id="4be7ea3a-ed3a-4cc9-9389-7d8ccfb373ff" style="height: 500px; width: 780px;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("4be7ea3a-ed3a-4cc9-9389-7d8ccfb373ff", [{"x": ["car", "service", "experience", "courtesy", "vehicle", "chevrolet", "customer", "dealership", "work", "honda", "time", "motorcar", "sale", "deal", "thanks", "crap", "scam", "mistake", "trouble", "fault", "liar", "fix", "drop", "complaint", "engine", "hate", "accident", "damage", "rude", "problem"], "orientation": "v", "y": [1035.97, 816.24, 695.75, 567.56, 459.09, 407.96, 398.33, 329.99, 325.28, 317.11, 314.29, 308.14, 290.36, 289.48, 273.46, -4.28, -4.48, -4.76, -5.17, -5.5, -5.58, -5.73, -5.91, -5.98, -8.68, -9.49, -10.16, -14.72, -16.35, -52.43], "name": "importance", "type": "bar", "text": "", "marker": {"line": {"width": 1, "color": "rgba(255, 153, 51, 1.0)"}, "color": "rgba(255, 153, 51, 0.6)"}}], {"width": 780, "paper_bgcolor": "#F5F6F9", "xaxis1": {"zerolinecolor": "#E1E5ED", "tickfont": {"color": "#4D5663"}, "titlefont": {"color": "#4D5663"}, "title": "Aspects", "showgrid": true, "gridcolor": "#E1E5ED"}, "height": 500, "yaxis1": {"zerolinecolor": "#E1E5ED", "tickfont": {"color": "#4D5663"}, "titlefont": {"color": "#4D5663"}, "title": "Importance", "showgrid": true, "gridcolor": "#E1E5ED"}, "titlefont": {"color": "#4D5663"}, "legend": {"bgcolor": "#F5F6F9", "font": {"color": "#4D5663"}}, "margin": {"r": 80, "t": 40, "b": 90, "l": 80}, "plot_bgcolor": "#F5F6F9"}, {"linkText": "Export to plot.ly", "showLink": true})});</script>



<div id="ae6ea20e-1712-4e94-975a-2ba250c41e58" style="height: 600px; width: 780px;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("ae6ea20e-1712-4e94-975a-2ba250c41e58", [{"x": ["courtesy_chevrolet", "customer_service", "great_experience", "motorcar_honda", "new_car", "would_recommend", "highly_recommend", "buying_experience", "make_sure", "great_service", "car_buying", "tim_jackson", "definitely_recommend", "friend_family", "made_sure", "far_worst", "pay_payment", "fix_problem", "wrong_car", "worst_place", "could_happier", "really_bad", "tire_pressure", "horrible_experience", "horrible_service", "worst_customer", "worst_dealership", "bad_experience", "worst_service", "worst_experience"], "orientation": "v", "y": [383.78, 206.65, 164.23, 158.75, 151.33, 137.58, 134.08, 125.71, 125.16, 109.83, 101.17, 101.16, 98.32, 92.66, 90.29, -3.73, -3.85, -3.88, -3.93, -3.95, -4.39, -4.46, -5.06, -5.42, -5.86, -6.6, -6.82, -6.82, -6.93, -9.84], "name": "importance", "type": "bar", "text": "", "marker": {"line": {"width": 1, "color": "rgba(255, 153, 51, 1.0)"}, "color": "rgba(255, 153, 51, 0.6)"}}], {"width": 780, "paper_bgcolor": "#F5F6F9", "xaxis1": {"zerolinecolor": "#E1E5ED", "tickfont": {"color": "#4D5663"}, "titlefont": {"color": "#4D5663"}, "title": "Aspects", "showgrid": true, "gridcolor": "#E1E5ED"}, "height": 600, "yaxis1": {"zerolinecolor": "#E1E5ED", "tickfont": {"color": "#4D5663"}, "titlefont": {"color": "#4D5663"}, "title": "Importance", "showgrid": true, "gridcolor": "#E1E5ED"}, "titlefont": {"color": "#4D5663"}, "legend": {"bgcolor": "#F5F6F9", "font": {"color": "#4D5663"}}, "margin": {"r": 80, "t": 40, "b": 160, "l": 80}, "plot_bgcolor": "#F5F6F9"}, {"linkText": "Export to plot.ly", "showLink": true})});</script>



<div id="c2eb6374-8d68-4730-9308-9b66e6a15d76" style="height: 600px; width: 780px;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("c2eb6374-8d68-4730-9308-9b66e6a15d76", [{"x": ["car_buying_experience", "great_customer_service", "would_definitely_recommend", "would_highly_recommend", "recommend_courtesy_chevrolet", "excellent_customer_service", "experience_courtesy_chevrolet", "recommend_friend_family", "would_recommend_anyone", "best_car_buying", "free_oil_change", "made_u_feel", "definitely_recommend_dealership", "would_recommend_dealership", "recommend_family_friend", "bad_customer_service", "many_bad_experience", "buying_car_always", "check_engine_light", "horrible_customer_service", "worst_service_ever", "worst_service_department", "pressure_sale_tactic", "never_felt_like", "high_pressure_sale", "poor_customer_service", "worst_dealership_ever", "terrible_customer_service", "worst_experience_ever", "worst_customer_service"], "orientation": "v", "y": [82.9, 66.39, 63.48, 59.34, 53.28, 36.58, 34.22, 31.03, 30.86, 30.17, 23.75, 23.3, 23.04, 21.53, 21.15, -1.93, -2.06, -2.17, -2.2, -2.26, -2.3, -2.36, -2.38, -2.51, -2.55, -3.05, -3.06, -3.09, -3.55, -6.6], "name": "importance", "type": "bar", "text": "", "marker": {"line": {"width": 1, "color": "rgba(255, 153, 51, 1.0)"}, "color": "rgba(255, 153, 51, 0.6)"}}], {"width": 780, "paper_bgcolor": "#F5F6F9", "xaxis1": {"zerolinecolor": "#E1E5ED", "tickfont": {"color": "#4D5663"}, "titlefont": {"color": "#4D5663"}, "title": "Aspects", "showgrid": true, "gridcolor": "#E1E5ED"}, "height": 600, "yaxis1": {"zerolinecolor": "#E1E5ED", "tickfont": {"color": "#4D5663"}, "titlefont": {"color": "#4D5663"}, "title": "Importance", "showgrid": true, "gridcolor": "#E1E5ED"}, "titlefont": {"color": "#4D5663"}, "legend": {"bgcolor": "#F5F6F9", "font": {"color": "#4D5663"}}, "margin": {"r": 80, "t": 40, "b": 230, "l": 80}, "plot_bgcolor": "#F5F6F9"}, {"linkText": "Export to plot.ly", "showLink": true})});</script>



```python
def PlotWordCloud(input_df, sent=None):
    if sent == 'pos':
        words = ' '.join(input_df[input_df['sentiment']>0]['aspects'])
    elif sent == 'neg':
        words = ' '.join(input_df[input_df['sentiment']<0]['aspects'])
    else:
        words = ' '.join(input_df['words'])

    mp.figure(figsize=(8,12))
    wordcloud = WordCloud(max_words=100, background_color='white').generate(words)
    mp.imshow(wordcloud)
    mp.axis('off')
```


```python
PlotWordCloud(df_ngram2, 'pos')
```


![png](output_39_0.png)



```python

```
