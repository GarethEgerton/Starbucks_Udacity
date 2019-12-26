Starbucks_Udacity
==============================

Udacity Starbucks Data Science Nano Degree Capstone Project

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
