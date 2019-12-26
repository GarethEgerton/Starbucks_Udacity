## Udacity Starbucks Data Science Nano Degree Capstone Project

<a href="https://medium.com/@gareth.egerton/starbucks-udacity-capstone-project-5d3d6776d73">Medium Blogpost</a>

Jupyter Notebooks:
<a href="https://medium.com/@gareth.egerton/starbucks-udacity-capstone-project-5d3d6776d73">1. Data_wrangling.ipynb</a>
<a href="https://medium.com/@gareth.egerton/starbucks-udacity-capstone-project-5d3d6776d73">2. Feature_engineering.ipynb</a>
<a href="https://medium.com/@gareth.egerton/starbucks-udacity-capstone-project-5d3d6776d73">3. Label_engineering.ipynb</a>
<a href="https://medium.com/@gareth.egerton/starbucks-udacity-capstone-project-5d3d6776d73">4. Grid_Search.ipynb</a>
<a href="https://medium.com/@gareth.egerton/starbucks-udacity-capstone-project-5d3d6776d73">5. Feature_Importance_and_Conclusion.ipynb</a>

### Project Overview

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app, provided by Starbucks as a Capstone Project as part of the Udacity Data Science Nano Degree.

### Description

Once every few days, Starbucks sends out an offer to users of their mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free).

Some users might not receive any offer during certain weeks.
Not all users receive the same offer, and that is the challenge to solve with this data set.
The task is to combine transactional, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. Informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, it can be assumed that the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

Transactional data is given showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

A key consideration is that someone using the app might make a purchase through the app without having received an offer or seen an offer.

### Problem
Build a machine learning model that given customer demographic and historical customer data, can predict whether a particular offer will be completed.

For any customer at any point in time therefore determine the probability that each of the 10 possible offers will be completed and thereby choose the offer that would yield the best revenue or profit maximisation (dependent on business targets).

Can this model be extended to distinguish between responsive offer completions (where a customer views and responds to an offer by actively trying to complete it vs completing it unintentionally as a results of normal spending habits? From a business perspective, customers in the latter case might not be ideal targets.

### Strategy
There is extensive historical data provided for each customer as time stamped transactional, offer received, viewed and completed data. My assumption is that this historical data will help distinguish the spending characterics of an individual consumer. Features will need to be engineered that are correlated with a customers offer spending reactivity.

Demographic features (income, gender, age) in addition to specific historical customer feature characteristics will be combined and trained using a gradient boosting machine learning algorithm to make a predictive model.

### Results and insights
* 85.1% Accuracy achieved using a CatBoost Binary Classification model to determine whether a customer will complete an offer.
* These results show the strength of the gradient boosting algorithm to solve this kind of problem.
* In this case we used CatBoost, but other algorithms such as XGBoost or LightGBM could also have been used and we would expect similarly strong results.
* Many of the features created were highly correlated with each other. One of the strengths in particular of gradient boosted decision trees is the ability to handle and ignore correlation between features. Using a method such as linear regression on the other hand would have been problematic with highly correlated features.


Project Organization
------------

    ├── LICENSE
    ├── README.md          
    ├── data
    │   ├── interim        <- Intermediate and processed data that has been transformed.
    │   └── raw            <- The original, immutable data 
    │
    ├── models             <- Trained and serialized models, model predictions, model summaries and results
    │
    ├── notebooks          
    │   └── exploratory    <- Jupyter notebooks.
    │       ├── 1. Data_wrangling.ipynb
    │       ├── 2. Feature_engineering.ipynb
    │       ├── 3. Label_engineering.ipynb
    │       ├── 4. Grid_Search.ipynb
    │       └── 5. Feature_Importance_and_Conclusion.ipynb
    │
    ├── requirements.txt   
    └── src                <- Source code scripts.
        ├── __init__.py    
        │   
        ├── utilities      <- helper scripts 
        │   └── cf_matrix.py*
        │
        ├── data           <- Scripts to wrangle data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn wrangled data into feature engineered data for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to train models and then use trained models to make predictions
            └── train_model.py


Acknowledgements and References

> https://catboost.ai/docs/concepts/python-reference_catboost.html
> https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/
> https://github.com/DTrimarchi10/confusion_matrix *confusion matrix code cf_matrix.py visualisation
> https://stackoverflow.com/
> https://www.kaggle.com/

Dependencies used:<br>
numpy==1.18.0 <br>
pandas==0.25.1 <br>
joblib==0.13.2 <br>
progress==1.5 <br>
progressbar2==3.42.0 <br>
catboost==0.20.2 <br>
scikit-image==0.15.0 <br>
scikit-learn==0.21.3 <br>
scikit-multilearn==0.2.0 <br>
scikit-plot==0.3.7 <br>
scipy==1.3.1 <br> 
matplotlib==3.1.1 <br>
seaborn==0.9.0 <br> 
scikit-plot==0.3.7 <br>


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
