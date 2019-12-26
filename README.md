Starbucks_Udacity
==============================

Udacity Starbucks Data Science Nano Degree Capstone Project

Project Overview
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app, provided by Starbucks as a Capstone Project as part of the Udacity Data Science Nano Degree.

Description
Once every few days, Starbucks sends out an offer to users of their mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free).

Some users might not receive any offer during certain weeks.
Not all users receive the same offer, and that is the challenge to solve with this data set.
The task is to combine transactional, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. Informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, it can be assumed that the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

Transactional data is given showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

A key consideration is that someone using the app might make a purchase through the app without having received an offer or seen an offer.

Problem
Build a machine learning model that given customer demographic and historical customer data, can predict whether a particular offer will be completed.

For any customer at any point in time therefore determine the probability that each of the 10 possible offers will be completed and thereby choose the offer that would yield the best revenue or profit maximisation (dependent on business targets).

Can this model be extended to distinguish between responsive offer completions (where a customer views and responds to an offer by actively trying to complete it vs completing it unintentionally as a results of normal spending habits? From a business perspective, customers in the latter case might not be ideal targets.

Strategy
There is extensive historical data provided for each customer as time stamped transactional, offer received, viewed and completed data. My assumption is that this historical data will help distinguish the spending characterics of an individual consumer. Features will need to be engineered that are correlated with a customers offer spending reactivity.

Demographic features (income, gender, age) in addition to specific historical customer feature characteristics will be combined and trained using a gradient boosting machine learning algorithm to make a predictive model.

Intuition
Higher income more wealthy customers are likley to spend more and hence complete more offers. Due to coffee being a relatively low proportion of their total income and seen in today's working culture as an essential good, I would expect wealthy high income customers to be price inelastic with respect to coffee and hence have the lowest responsiveness rate to offers. High income customers will therefore often complete offers regardless of having viewed them or not, and their base spending rate will not vary much when under the influence of an offer or not.
Higher income individuals are also more likely to have more senior, potentially higher stress jobs with a greater degree of responsibility. Coffee used as a stimulant for improved cognition and to combat tiredness may therefore be prevalent, contributing to greater offer completions with low offer responsiveness.
Younger consumers are generally more tech savvy and engaged with mobile and social media channels in particular. Due to a higher proportion of time spent utilising technology, these individuals will more likely view and hence respond to offers. Furthermore younger consumers are likeley to have lower disposable income with relatively expensive Starbucks coffee seen as a luxury good. They are therefore likley to be more price elastic and hence more responsive to offers.
My expectation overall is that by virtue of greater disposable income and least engagement with technology, older more wealth individuals will complete the most offers without having viewed them. Younger lower income customers will overall complete less offers, however will show the greatest reactivity.
The greater a customers historical spending per day, the more likely they will spend more in the future and hence the more likely they will complete future offers.
Customer historical offer completions indicate a higher likelihood of completing offers in the future and in particular the more likely they are to complete offers of a similar type.
Metrics
Due to the balanced nature of this dataset with offer completion / failure approximately equal in proportions, accuracy will be a good measure of the performance of the model.
To optimise based on business requirements we may need to use f1-score and tweak the model to ensure as many potentially responsive customers are targetted and as few unresponsive customers. Once the primary objective of highest accuracy model has been determined, these additional options will be explored.
Labels to be predicted:
Offer completed
Offer failed to complete
Offer completed without having been viewed (measure of low offer elasticity of demand)



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate and processed data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, model summaries and results
    │
    ├── notebooks          
    │   └── exploratory    <- Jupyter notebooks.
    │       ├── 1. Data_wrangling.ipynb
    │       ├── 2. Feature_engineering.ipynb
    │       ├── 3. Label_engineering.ipynb
    │       ├── 4. Grid_Search.ipynb
    │       └── 5. Feature_Importance_and_ Conclusion.ipynb
    │
    ├── requirements.txt   <- Required dependencies
    │   └── cf_matrix.py
    └── src                <- Source code scripts.
        ├── __init__.py    <- Makes src a Python module
        │   
        ├── utilities      <- helper scripts 
        │   └── cf_matrix.py *
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
> * confusion matrix code cf_matrix.py visualisation from https://github.com/DTrimarchi10/confusion_matrix Dennit T
> https://stackoverflow.com/
> https://www.kaggle.com/

Dependencies used:
numpy==1.18.0
pandas==0.25.1
joblib==0.13.2
progress==1.5
progressbar2==3.42.0
catboost==0.20.2
scikit-image==0.15.0
scikit-learn==0.21.3
scikit-multilearn==0.2.0
scikit-plot==0.3.7
scipy==1.3.1
matplotlib==3.1.1
seaborn==0.9.0
scikit-plot==0.3.7


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
