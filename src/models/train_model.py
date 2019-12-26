import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from src.data.make_dataset import save_file
import catboost
from catboost import CatBoostClassifier
from catboost import Pool
from src.utilities import cf_matrix

def label_creater(df, label_grid=None):
    
    '''
    Creates label column based on label_grid dictionary mapping criteria
    Drops features that would lead to data leakage
        
    Parameters
    -----------
    df:  DataFrame
    label_grid: dictionary to assign labels
    dictionary keys as follows, numerical label values to be assigned.
    e.g. 
    {'unresponsive': 7,
     'no_complete_no_view': 6,
     'incomplete_responsive': 5,
     'completed_responsive': 4,
     'complete_anyway': 3,
     'completed_before_viewed': 2, 
     'completed_not_viewed': 1}
            
    Returns
    -------
    DataFrame
    '''  
       
    df['completed_not_viewed'] = ((df.completed == 1) & (df.viewed == 0))\
                                  * label_grid['completed_not_viewed']
    
    df['completed_before_viewed'] = (((df.completed == 1) & (df.viewed == 1)) 
                                       & (df.received_spend > df.difficulty)
                                     ) * label_grid['completed_before_viewed']
    
    # Required completion spending per time left < base rate
    df['complete_anyway'] = (((df.completed == 1) & (df.viewed == 1)) 
                                & (df.viewed_spend / df.viewed_days_left\
                                   <= df.amount_per_day_not_offer)\
                             )* label_grid['complete_anyway']
    
    # Required completion spending per time left > base rate
    df['completed_responsive'] = (((df.completed == 1) & (df.viewed == 1) \
                                   & ~(df.received_spend > df.difficulty))
                                   & (((df.viewed_spend / df.viewed_days_left \
                                        > df.amount_per_day_not_offer) 
                                    | (pd.isna(df.amount_per_day_not_offer) 
                                       & ~(df.received_spend > df.difficulty))
                                    )))* label_grid['completed_responsive']
    
    # Didn't complete, spending wasn't increased above base spending
    df['incomplete_responsive'] = (((df.completed == 0) & (df.viewed == 1)) 
                                     & ((((df.viewed_spend / df.viewed_days_left) \
                                          > df.amount_per_day_not_offer))
                                       | ((df.viewed_spend > 0) \
                                          & pd.isna(df.amount_per_day_not_offer)
                                         )))* label_grid['incomplete_responsive']
        
    df['no_complete_no_view'] = ((df.completed == 0) & (df.viewed == 0))* \
                                label_grid['no_complete_no_view']
    
    df['unresponsive'] = ((df['completed_not_viewed']==0) & 
                          (df['completed_before_viewed']==0) & 
                          (df['complete_anyway']==0) & 
                          (df['completed_responsive']==0) & 
                          (df['incomplete_responsive']==0) & 
                          (df['no_complete_no_view']==0)
                          )* label_grid['unresponsive']  
    
    df2 = (df[['completed_not_viewed', 'completed_before_viewed', 'complete_anyway', 
                      'completed_responsive', 'incomplete_responsive', 'unresponsive', 
                      'no_complete_no_view']])
    
        
    df['label'] = df[['completed_not_viewed', 'completed_before_viewed', 'complete_anyway', 
                      'completed_responsive', 'incomplete_responsive', 'unresponsive', 
                      'no_complete_no_view']].sum(axis=1)
    
    df.drop(['received_spend',
            'viewed_spend',
            'viewed_days_left',
            'remaining_to_complete',
            'viewed_in_valid',
            'viewed',
            'spend>required',
            'offer_spend',
            'completed',
            'completed_not_viewed',
            'completed_before_viewed',
            'complete_anyway',
            'completed_responsive',
            'incomplete_responsive',
            'no_complete_no_view',
            'unresponsive',], axis=1, inplace=True)
    
    return df

def exploratory_training(labels=None, labels_compact=None, drop_features=None, 
                         feature_engineering=True, verbose=500, return_model=False, 
                         data='../../data/interim/transcript_final_optimised.joblib', 
                         **params):
    '''
    Function to help manage and train CatBoost Classifer under
    varying conditions. 
    These include:
    > Which features to remove
    > How to assign target labels
    > CatBoost hyper parameters to use
    > Whether to return model or accuracy score
    > Whether to display detailed results   
        
    Parameters
    -----------
    labels: dictionary to assign labels
    dictionary keys as follows, numerical label values to be assigned.
    e.g. for greatest label category separation: 
    {'unresponsive': 7,
     'no_complete_no_view': 6,
     'incomplete_responsive': 5,
     'completed_responsive': 4,
     'complete_anyway': 3,
     'completed_before_viewed': 2, 
     'completed_not_viewed': 1}
     
    labels_compact: dictionary, assigns labels under multiple 
    categories for purpose of confusion matrix output, e.g.
    {'failed':0, 'completed':1}
            
    drop_features - list of features to remove, default None
    features_engineering - boolean, default True. If not True, removes 
    all engineered features leaving original base features.
    verbose - int, displays results of training at each iteration. If 
    not false also displays classification report and confusion matrix.
    return_model - boolean, default False. If True, returns trained
    model, otherwise returns model accuracy score.
    data - path to .joblib file of DataFrame
    **params - dictionary of CatBoost hyper parameters.    
    '''
    
    df = joblib.load(data)
    
    # assign labels
    df = label_creater(df, label_grid=labels)
    
    # categorical feature assignment
    cat_features_name = ['person', 'gender', 'id', 'offer_7', 'offer_14', 'offer_17', 
                         'offer_21', 'offer_24', 'offer_30']
    

    df.sort_values('time_days', inplace=True)
    X = df.drop('label', axis=1)
    
    # Selects only base features for X
    if not feature_engineering:    
        X = X[['person', 'age', 'income', 'signed_up', 'gender', 'id', 'rewarded',
           'difficulty', 'reward', 'duration', 'mobile', 'web', 'social', 'bogo',
           'discount', 'informational', 'time_days']]
    
    # Removes specified features
    if drop_features is not None:        
        X.drop(drop_features, axis=1, inplace=True)
    
    # Assigns columns index location of categorical features
    cat_features = [X.columns.get_loc(i) for i in cat_features_name if i in X.columns]
    
    y = df.label
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, 
                                                        random_state=42)
    
    # Assigns weights to labels since these are unbalanced
    weights = [df.label.value_counts().sum() / df.label.value_counts()[i] for i in 
               set(labels.values())]

    # utilised by CatBoost to assign train and test data, needed to perform early stopping when
    # test accuracy falls beyond a specified number of interations
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

    # model training
    model = CatBoostClassifier(
        iterations=7000,
        loss_function='MultiClass',
        early_stopping_rounds=50,
        task_type='GPU',
        cat_features=cat_features,
        class_weights= weights,
        verbose=verbose,
        **params)    

    model.fit(train_pool,
          eval_set=test_pool,
          verbose=verbose,
          plot=False);
    
    # if multiple labels under similar categories, specify these categories
    if not labels_compact:
        labels_compact=labels
          
    preds_class = model.predict(X_test)
    print("")

    # displays learning rate, weights and results
    if verbose:
        display(F'Learning Rate set to: {model.get_all_params()["learning_rate"]}')
        display(F'Accuracy Score: {accuracy_score(y_test, preds_class)}')
        display(F'Weights: {weights}')
        matrix = confusion_matrix(y_test, preds_class)
        width = len(labels_compact)*2 + 1
        cf_matrix.make_confusion_matrix(matrix, figsize=(width,width), cbar=False, 
                                        categories=labels_compact.keys(), group_names = 
                                        ['True Neg', 'False Pos', 'False Neg', 'True Pos'])
    
        print(classification_report(y_test, preds_class, target_names=list(labels_compact.keys()))) 
        
    if return_model:
        return model
        
    else:
        return accuracy_score(y_test, preds_class)

def generate_folds(cv, X_train, y_train):
    '''
    Iterate through cv folds and split into list of folds
    Checks that each fold has the same % of positive class
    
    Parameters
    -----------
    cv: cross validation generator
               
    Returns
    -------
    X_train, X_test, y_train, y_test: DataFrames
    '''
    train_X, train_y, test_X, test_y = [], [], [], []
    
    for i in cv:
        train_X.append(X_train.iloc[i[0]])
        train_y.append(y_train.iloc[i[0]])

        test_X.append(X_train.iloc[i[1]])
        test_y.append(y_train.iloc[i[1]])
      
    print('positive classification % per fold and length')
    for i in range(len(train_X)):
        print('train[' + str(i) + ']' , round(train_y[i].sum() / train_y[i].count(), 4), 
              train_y[i].shape)
        print('test[' + str(i) + '] ' , round(test_y[i].sum() / test_y[i].count(), 4), 
              test_y[i].shape)
           
    return train_X, train_y, test_X, test_y

def gridsearch_early_stopping(cv, X, y, folds, grid, cat_features=None, save=None):
    '''
    Perform grid search with early stopping across folds specified by index 
    
    Parameters
    -----------
    cv: cross validation
    X: DataFrame or Numpy array
    y: DataFrame or Numpy array
    fold: list of fold indexes
    grid: parameter grid
    save:   string, excluding file extension (default=None)
            saves results_df for each fold to folder '../../data/interim'
    '''
    
    if np.unique(y).size <= 2:
        loss_function = 'Logloss'
    else:
        loss_function = 'MultiClass'
           
    # generate data folds 
    train_X, train_y, test_X, test_y = generate_folds(cv, X, y)
    
    # iterate through specified folds
    for fold in folds:
        # assign train and test pools
        test_pool = Pool(data=test_X[fold], label=test_y[fold], cat_features=cat_features)
        train_pool = Pool(data=train_X[fold], label=train_y[fold], cat_features=cat_features)

        # creating results_df dataframe
        results_df = pd.DataFrame(columns=['params' + str(fold), loss_function + str(fold), 
                                           'Accuracy'+ str(fold), 'iteration'+ str(fold)])

        best_score = 99999

        # iterate through parameter grid
        for params in ParameterGrid(grid):

            # create catboost classifer with parameter params
            model = CatBoostClassifier(cat_features=cat_features,
                                        early_stopping_rounds=50,
                                        task_type='GPU',
                                        custom_loss=['Accuracy'],
                                        iterations=3000,
                                        #class_weights=weights, 
                                        **params)

            # fit model
            model.fit(train_pool, eval_set=test_pool, verbose=400)

            # append results to results_df
            
            print(model.get_best_score()['validation'])
            results_df = results_df.append(pd.DataFrame(
                            [[params, model.get_best_score()['validation'][loss_function], 
                              model.get_best_score()['validation']['Accuracy'], 
                              model.get_best_iteration()]], 
                              columns=['params' + str(fold), loss_function + str(fold), 
                                       'Accuracy' + str(fold), 'iteration' + str(fold)]))

            # save best score and parameters
            if model.get_best_score()['validation'][loss_function] < best_score:
                best_score = model.get_best_score()['validation'][loss_function]
                best_grid = params

        print("Best logloss: ", best_score)
        print("Grid:", best_grid)

        save_file(results_df, save + str(fold) + '.joblib', dirName='../../models')
        display(results_df)

def grid_search_results(raw_file, num_folds):
    '''
    Loads raw cross validation fold results.
    Displays results highlighting best scores
    
    Parameters
    -----------
    raw_file: string, the name of the file excluding fold number and extension
    extension: string, type of file, e.g '.joblib', '.pkl'
    num_folds: number of cv folds
                
    Returns
    -------
    results DataFrame
    '''
    
    # list of folds
    results_files = [0 for i in range(0, num_folds)]
        
    # read results files for each fold
    for i in range(0, num_folds):
        results_files[i] = joblib.load(f'../../models/{raw_file}{i}.joblib')            
    
    # join results files in one dataframe
    results_df = pd.concat([results_files[i] for i in range(0, num_folds)], axis=1)
    metrics = int(results_df.shape[1] / num_folds - 1)
    
    # drop extra params columns
    results_df.rename(columns={"params0": "Params"}, inplace=True)
    results_df.drop([i for i in results_df.columns if 'params' in i], axis=1, inplace=True)
    
    # convert data columns to numeric 
    def to_numeric_ignore(x, errors='ignore'):
        return pd.to_numeric(x, errors=errors)    
    results_df = results_df.apply(to_numeric_ignore)
    
    # loops through metrics and create mean column for each metric
    metric_names=[]
    for i in results_df.columns[0:metrics+1]:
        i = i[:-1]
        metric_names.append(i)
        results_df[i + '_mean'] = results_df[[x for x in results_df.columns 
                                              if i in x]].mean(axis=1)
    
    results_df.reset_index(drop=True, inplace=True)
        
    # instantiating best_scores dataframe
    best_scores = pd.DataFrame(columns=['Params', 'Metric', 'Score'])
        
    negative_better = ['MultiClass', 'iteration', 'logloss']
    positive_better = ['Accuracy']
    
      
    # get index of best parameters
    best_param_idx = []
    for i in metric_names:
        if i in negative_better:
            best_param_idx = results_df[i+ '_mean'].idxmin(axis=0)
        if i in positive_better:
            best_param_idx = results_df[i+ '_mean'].idxmax(axis=0)

        row = pd.DataFrame({'Metric': [i + '_mean'], 
                            'Params': [results_df.loc[best_param_idx, 'Params']], 
                            'Score': [results_df.loc[best_param_idx, i + '_mean']]})
        best_scores = best_scores.append(row, ignore_index=True)

    results_df.insert(0, 'Parameters', results_df.Params)
    results_df.drop(['Params', 'Param_mean'], axis=1, inplace=True)

    best_scores = best_scores[best_scores.Metric != 'Param_mean']
    
    display(best_scores)
    
    negative_columns = []
    positive_columns = []
    
    # highlight columns where negative metrics are better
    for i in negative_better:
        negative_columns.extend([x for x in results_df.columns if i in x])
    
    # highlight columns where positive metrics are better
    for i in positive_better:
        positive_columns.extend([x for x in results_df.columns if i in x])
        
    display(results_df.style
    .highlight_max(subset = positive_columns, color='lightgreen')
    .highlight_min(subset= negative_columns, color='lightgreen'))
    
    return results_df, best_scores

def gridsearch_early_stopping_importance(cv, X, y, folds, grid, cat_features=[], save=None):
    '''
    Perform grid search with early stopping across each fold using 
    parameter grid, returning feature importance results.
    
    Parameters
    -----------
    cv: cross validation
    X: DataFrame or Numpy array
    y: DataFrame or Numpy array
    fold: list of fold indexes
    grid: parameter grid
    save:   string, excluding file extension (default=None)
            saves results_df for each fold to folder '../../data/interim'
            
    Returns
    -------
    list of DataFrames of importances per fold using metrics:
        > PredictionValuesChange
        > LossFunctionChange (train)
        > LossFunctionChange (test)                      
    '''
        
    # generate data folds
    train_X, train_y, test_X, test_y = generate_folds(cv, X, y)
      
    
    importance_folds = [0 for fold in folds]
    
    # iterate through specified folds
    for fold in folds:
        # assign train and test pools
        test_pool = Pool(data=test_X[fold], label=test_y[fold], cat_features=cat_features)
        train_pool = Pool(data=train_X[fold], label=train_y[fold], cat_features=cat_features)

        # creating results_df dataframe
        results_df = pd.DataFrame(columns=['params' + str(fold), 'logloss'+ str(fold), 
                                           'Accuracy'+ str(fold), 'iteration'+ str(fold)])

        best_score = 99999

        # iterate through parameter grid
        for params in ParameterGrid(grid):

            # create catboost classifer with parameter params
            model = CatBoostClassifier(cat_features=cat_features,
                                        early_stopping_rounds=50,
                                        task_type='GPU',
                                        custom_loss=['Accuracy'],
                                        iterations=7000,
                                        **params)
            # fit model
            model.fit(train_pool, eval_set=test_pool, verbose=400)
            importance_folds[fold] = importances(model, train_pool, test_pool)

    return(importance_folds)

def best_param_grid(grid_results_file, metric='logloss_mean', folds=5):
    '''
    Takes grid search results file(s) and returns optimal parameters
    
    Parameters
    -----------
    grid_search_file: file name excluding extention and fold suffix
    folds: number of folds (or files) 
    metric: which metric to choose best parameters from, default ('logloss_mean')
    
    Returns
    -----------
    grid: dictionary of lists of parameters  
    '''  
    
    results_person, best_scores = grid_search_results(grid_results_file, 5, display_results=False)
    
    grid = best_scores[best_scores.Metric == metric + '_mean'].Params.values[0]
    
    for i in grid:
        grid[i] = [grid[i]]
        
    print('Training with ' + str(grid))
    
    return grid

def importances(model, train_pool, test_pool):
    '''
    Comparison of sorted feature imporances based on:
    
    1. PredictionValuesChange
    2. LossFunctionChange
    3. ShapValues
        
    Parameters
    -----------
    model: CatBoost model
    train_pool: CatBoost train data
    test_pool: CatBoost test data
               
    Returns
    -------
    DataFrame
    '''
    
    importance = pd.DataFrame(np.array(model.get_feature_importance(prettified=True)))
    loss_function_change = pd.DataFrame(
                                        np.array(model.get_feature_importance(
                                        train_pool,
                                        'LossFunctionChange',
                                        prettified=True)))
    loss_function_change_test = pd.DataFrame(
                                            np.array(model.get_feature_importance(
                                            test_pool,
                                            'LossFunctionChange',
                                            prettified=True)))
    
    df = pd.concat([importance, loss_function_change, loss_function_change_test], axis=1)
    df.columns = ['feature0', 'importance0', 'feature1', 'importance1', 'feature2', 'importance2']
    features = [x for x in df.columns if 'feature' in x]
    
    # separate by features and importances
    feature_list =[]
    for i, j in enumerate(features):
        feature_list.append(df[[j, 'importance'+str(i)]].set_index(j))
    
    # join by features as indedx
    features_joined = feature_list[0]
    for i in range(len(feature_list))[1:]:
        features_joined = features_joined.join(feature_list[i])
        
    return features_joined

def compare_importances(path='../../models/importances_multi.joblib'):
    '''
    Takes importances for each cross validation fold and calculates mean per importance metric.
    
    Parameters
    -----------
    path: path to importance_results
    method: string
            'importance1': PredictionValues_Change
            'importance2': LossFunctionChange Train
            'importance3': LossFunctionChange Test
            
    Returns
    -----------
    df: DataFrame    
    '''
    
    importance_type = {'importance0': 'PredictionValuesChange',
                      'importance1': 'LossFunctionChange_Train',
                      'importance2': 'LossFunctionChange_Test'}
    
    fold_result_list = joblib.load(path)
    folds = len(fold_result_list)
    
    importance_columns=[]
    
    for method in importance_type.keys():
           
        method_per_fold = [fold_result_list[i][method].rename(f'fold{i}') for i in range(folds)]

        df = pd.DataFrame(method_per_fold).transpose().rename\
            (columns={'importance2' : 'fold' + str(i) for i in range(folds)})
        df['mean'] = df.sum(axis=1)/folds
        df.sort_values('mean', ascending=False, inplace=True)
        df.rename(columns={'mean': importance_type[method]}, inplace=True)
        df.drop(['fold0', 'fold1', 'fold2', 'fold3', 'fold4'], axis=1, inplace=True)
        
        importance_columns.append(df)
        
    
    results = importance_columns[0].join(importance_columns[1]).join(importance_columns[2])
    
    return results       

def grid_search_results(raw_file, num_folds, display_results=True):
    '''
    Loads raw cross validation fold results.
    Displays results highlighting best scores
    
    Parameters
    -----------
    raw_file: string, the name of the file excluding fold number and extension
    extension: string, type of file, e.g '.joblib', '.pkl'
    num_folds: number of cv folds
                
    Returns
    -------
    results DataFrame
    '''
    
    # list of folds
    results_files = [0 for i in range(0, num_folds)]
        
    # read results files for each fold
    for i in range(0, num_folds):
        results_files[i] = joblib.load(f'../../models/{raw_file}{i}.joblib')            
    
    # join results files in one dataframe
    results_df = pd.concat([results_files[i] for i in range(0, num_folds)], axis=1)
    metrics = int(results_df.shape[1] / num_folds - 1)
    
    # drop extra params columns
    results_df.rename(columns={"params0": "Params"}, inplace=True)
    results_df.drop([i for i in results_df.columns if 'params' in i], axis=1, inplace=True)
    
    # convert data columns to numeric 
    def to_numeric_ignore(x, errors='ignore'):
        return pd.to_numeric(x, errors=errors)    
    results_df = results_df.apply(to_numeric_ignore)
    
    # loops through metrics and create mean column for each metric
    metric_names=[]
    for i in results_df.columns[0:metrics+1]:
        i = i[:-1]
        metric_names.append(i)
        results_df[i + '_mean'] = results_df[[x for x in results_df.columns if i in x]].mean(axis=1)
    
    results_df.reset_index(drop=True, inplace=True)
        

    # instantiating best_scores dataframe
    best_scores = pd.DataFrame(columns=['Params', 'Metric', 'Score'])
        
    negative_better = ['MultiClass', 'iteration', 'logloss']
    positive_better = ['Accuracy']
    
      
    # get index of best parameters
    best_param_idx = []
    for i in metric_names:
        if i in negative_better:
            best_param_idx = results_df[i+ '_mean'].idxmin(axis=0)
        if i in positive_better:
            best_param_idx = results_df[i+ '_mean'].idxmax(axis=0)

        row = pd.DataFrame({'Metric': [i + '_mean'], 'Params': [results_df.loc[best_param_idx, 'Params']], 'Score': [results_df.loc[best_param_idx, i + '_mean']]})
        best_scores = best_scores.append(row, ignore_index=True)

    results_df.insert(0, 'Parameters', results_df.Params)
    results_df.drop(['Params', 'Param_mean'], axis=1, inplace=True)

    best_scores = best_scores[best_scores.Metric != 'Param_mean']
    
    if display_results:
        display(best_scores)
    
    negative_columns = []
    positive_columns = []
    
    # highlight columns where negative metrics are better
    for i in negative_better:
        negative_columns.extend([x for x in results_df.columns if i in x])
    
    # highlight columns where positive metrics are better
    for i in positive_better:
        positive_columns.extend([x for x in results_df.columns if i in x])

    if display_results:

        display(results_df.style
        .highlight_max(subset = positive_columns, color='lightgreen')
        .highlight_min(subset= negative_columns, color='lightgreen'))
    
    return results_df, best_scores

def exploratory_training(labels=None, labels_compact=None, drop_features=None, 
                         feature_engineering=True, verbose=500, return_model=False, **params):
    
    df = joblib.load('../../data/interim/transcript_final_optimised.joblib')
    df = label_creater(df, label_grid=labels)
    cat_features_name = ['person', 'gender', 'id', 'offer_7', 'offer_14', 'offer_17', 
                         'offer_21', 'offer_24', 'offer_30']
    
    df.sort_values('time_days', inplace=True)
    X = df.drop('label', axis=1)
    
    if not feature_engineering:    
        X = X[['person', 'age', 'income', 'signed_up', 'gender', 'id', 'rewarded',
           'difficulty', 'reward', 'duration', 'mobile', 'web', 'social', 'bogo',
           'discount', 'informational', 'time_days']]

    if drop_features is not None:        
        X.drop(drop_features, axis=1, inplace=True)

    cat_features = [X.columns.get_loc(i) for i in cat_features_name if i in X.columns]
       
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, 
                                                        random_state=42)
    
    weights = [df.label.value_counts().sum() / df.label.value_counts()[i] for i in 
                range(0, df.label.nunique())]

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=7000,
        loss_function='MultiClass',
        early_stopping_rounds=50,
        task_type='GPU',
        cat_features=cat_features,
        class_weights= weights,
        verbose=verbose,
        **params)    

    model.fit(train_pool,
          eval_set=test_pool,
          verbose=verbose,
          plot=False);
    
    if not labels_compact:
        labels_compact=labels
          
    preds_class = model.predict(X_test)
    print("")

    if verbose:
        display(F'Learning Rate set to: {model.get_all_params()["learning_rate"]}')
        display(F'Accuracy Score: {accuracy_score(y_test, preds_class)}')
        display(F'Weights: {weights}')
        matrix = confusion_matrix(y_test, preds_class)
        width = len(labels_compact)*2 + 1
        cf_matrix.make_confusion_matrix(matrix, figsize=(width,width), cbar=False, 
                                        categories=labels_compact.keys(), group_names = 
                                        ['True Neg', 'False Pos', 'False Neg', 'True Pos'])
    
        print(classification_report(y_test, preds_class, target_names=list(labels_compact.keys()))) 
    
    if return_model:
        return model, X_test, y_test
        
    else:
        return accuracy_score(y_test, preds_class)

def compare_accuracies(experiments, compact_label, parameters):
    '''
    Trains CatBoost Classifer across specified hyper parameter, label, feature and experiment 
    sets. Compares and returns results in DataFrame
    Saves results as '../../models/results_summary_compare.joblib'
    
    Parameters
    ----------
    experiments: list of experiment name strings
    compact_label: dictionary of dictionary compact labels
    parameters: dictionary of dictionary optimal and default parameters    
            
    Returns
    -------
    DataFrame
    '''  
    results_summary=[]
    
    # train classifer across parameter, label, feature, experiment combinations
    for engineering in [True, False]:    
        for experiment in experiments:
            for param_selection in ['default', experiment]:
                compact = compact_label[experiment] 
                results_summary.append([engineering, experiment, 
                                        parameters[param_selection], 
                                        exploratory_training(
                                            labels=labels[experiment], 
                                            labels_compact=compact_label, 
                                            feature_engineering=engineering, verbose=False, 
                                            return_model=False, **parameters[param_selection])])
                   
    pd.set_option('max_colwidth', 200)
    
    #convert to DataFrame
    results_accuracy = pd.DataFrame(results_summary, 
                                    columns=['Feature Engineering', 'Experiment', 'Parameters', 
                                             'Accuracy'])
    # reorder columns
    results_accuracy = results_accuracy[['Parameters', 'param', 'Experiment', 
                                         'Feature Engineering', 'Accuracy']]
    results_accuracy.sort_values(['Experiment', 'Feature Engineering', 'Accuracy'], inplace=True)
    
    # calculate differences between accuracies
    results_accuracy['Delta'] = results_accuracy.Accuracy.diff(periods=1)
    results_accuracy.fillna(0, inplace=True)
    
    joblib.dump(results_summary, '../../models/results_summary.joblib', compress=True)
        
    return results_accuracy.style.format({'Delta': "{:.2%}"})

def importances(model, train_pool, test_pool):
    '''
    Comparison of sorted feature imporances based on:
    
    1. PredictionValuesChange
    2. LossFunctionChange
    3. ShapValues
        
    Parameters
    -----------
    model: CatBoost model
    train_pool: CatBoost train data
    test_pool: CatBoost test data
               
    Returns
    -------
    DataFrame
    '''
    
    importance = pd.DataFrame(np.array(model.get_feature_importance(prettified=True)))
    loss_function_change = pd.DataFrame(
                                        np.array(model.get_feature_importance(
                                        train_pool,
                                        'LossFunctionChange',
                                        prettified=True)))
    loss_function_change_test = pd.DataFrame(
                                            np.array(model.get_feature_importance(
                                            test_pool,
                                            'LossFunctionChange',
                                            prettified=True)))
    
    df = pd.concat([importance, loss_function_change, loss_function_change_test], axis=1)
    df.columns = ['feature0', 'importance0', 'feature1', 'importance1', 'feature2', 'importance2']
    features = [x for x in df.columns if 'feature' in x]
    
    # separate by features and importances
    feature_list =[]
    for i, j in enumerate(features):
        feature_list.append(df[[j, 'importance'+str(i)]].set_index(j))
    
    # join by features as indedx
    features_joined = feature_list[0]
    for i in range(len(feature_list))[1:]:
        features_joined = features_joined.join(feature_list[i])
        
    return features_joined

def gridsearch_early_stopping_importance(cv, X, y, folds, grid, cat_features=[], save=None):
    '''
    Perform grid search with early stopping across each fold using 
    parameter grid, returning feature importance results.
    
    Parameters
    -----------
    cv: cross validation
    X: DataFrame or Numpy array
    y: DataFrame or Numpy array
    fold: list of fold indexes
    grid: parameter grid
    save:   string, excluding file extension (default=None)
            saves results_df for each fold to folder '../../data/interim'
            
    Returns
    -------
    list of DataFrames of importances per fold using metrics:
        > PredictionValuesChange
        > LossFunctionChange (train)
        > LossFunctionChange (test)                      
    '''
        
    # generate data folds
    train_X, train_y, test_X, test_y = generate_folds(cv, X, y)
      
    
    importance_folds = [0 for fold in folds]
    
    # iterate through specified folds
    for fold in folds:
        # assign train and test pools
        test_pool = Pool(data=test_X[fold], label=test_y[fold], cat_features=cat_features)
        train_pool = Pool(data=train_X[fold], label=train_y[fold], cat_features=cat_features)

        # creating results_df dataframe
        results_df = pd.DataFrame(columns=['params' + str(fold), 'logloss'+ str(fold), 
                                           'Accuracy'+ str(fold), 'iteration'+ str(fold)])

        best_score = 99999

        # iterate through parameter grid
        for params in ParameterGrid(grid):

            # create catboost classifer with parameter params
            model = CatBoostClassifier(cat_features=cat_features,
                                        early_stopping_rounds=50,
                                        task_type='GPU',
                                        custom_loss=['Accuracy'],
                                        iterations=7000,
                                        **params)
            # fit model
            model.fit(train_pool, eval_set=test_pool, verbose=400)
            importance_folds[fold] = importances(model, train_pool, test_pool)

    return(importance_folds)

def best_param_grid(grid_results_file, metric='logloss_mean', folds=5):
    '''
    Takes grid search results file(s) and returns optimal parameters
    
    Parameters
    -----------
    grid_search_file: file name excluding extention and fold suffix
    folds: number of folds (or files) 
    metric: which metric to choose best parameters from, default ('logloss_mean')
    
    Returns
    -----------
    grid: dictionary of lists of parameters  
    '''  
    
    results_person, best_scores = grid_search_results(grid_results_file, 5, display_results=False)
    
    grid = best_scores[best_scores.Metric == metric + '_mean'].Params.values[0]
    
    for i in grid:
        grid[i] = [grid[i]]
        
    print('Training with ' + str(grid))
    
    return grid

def importances(model, train_pool, test_pool):
    '''
    Comparison of sorted feature imporances based on:
    
    1. PredictionValuesChange
    2. LossFunctionChange
    3. ShapValues
        
    Parameters
    -----------
    model: CatBoost model
    train_pool: CatBoost train data
    test_pool: CatBoost test data
               
    Returns
    -------
    DataFrame
    '''
    
    importance = pd.DataFrame(np.array(model.get_feature_importance(prettified=True)))
    loss_function_change = pd.DataFrame(
                                        np.array(model.get_feature_importance(
                                        train_pool,
                                        'LossFunctionChange',
                                        prettified=True)))
    loss_function_change_test = pd.DataFrame(
                                            np.array(model.get_feature_importance(
                                            test_pool,
                                            'LossFunctionChange',
                                            prettified=True)))
    
    df = pd.concat([importance, loss_function_change, loss_function_change_test], axis=1)
    df.columns = ['feature0', 'importance0', 'feature1', 'importance1', 'feature2', 'importance2']
    features = [x for x in df.columns if 'feature' in x]
    
    # separate by features and importances
    feature_list =[]
    for i, j in enumerate(features):
        feature_list.append(df[[j, 'importance'+str(i)]].set_index(j))
    
    # join by features as indedx
    features_joined = feature_list[0]
    for i in range(len(feature_list))[1:]:
        features_joined = features_joined.join(feature_list[i])
        
    return features_joined

def compare_importances(path='../../models/importance_person_all_features_new.joblib'):
    '''
    Takes importances for each cross validation fold and calculates mean per importance metric.
    
    Parameters
    -----------
    path: path to importance_results
    method: string
            'importance1': PredictionValues_Change
            'importance2': LossFunctionChange Train
            'importance3': LossFunctionChange Test
            
    Returns
    -----------
    df: DataFrame    
    '''
    
    importance_type = {'importance0': 'PredictionValuesChange',
                      'importance1': 'LossFunctionChange_Train',
                      'importance2': 'LossFunctionChange_Test'}
    
    fold_result_list = joblib.load(path)
    folds = len(fold_result_list)
    
    importance_columns=[]
    
    for method in importance_type.keys():
           
        method_per_fold = [fold_result_list[i][method].rename(f'fold{i}') for i in range(folds)]

        df = pd.DataFrame(method_per_fold).transpose().rename\
            (columns={'importance2' : 'fold' + str(i) for i in range(folds)})
        df['mean'] = df.sum(axis=1)/folds
        df.sort_values('mean', ascending=False, inplace=True)
        df.rename(columns={'mean': importance_type[method]}, inplace=True)
        df.drop(['fold0', 'fold1', 'fold2', 'fold3', 'fold4'], axis=1, inplace=True)
        
        importance_columns.append(df)
        
    
    results = importance_columns[0].join(importance_columns[1]).join(importance_columns[2])
    
    return results  

def testing_offers(observation, X_test):
    '''
    Utilises binary_model along with X_test data to determine the
    probabilities that each observed offer would have been completed if
    any of the 10 possible offers were given, all other factors 
    remaining equal.
    Offer and demographic data have been added to the output DataFrame.
    
    Parameters
    -----------
    obervation: int position with X_test
    X_test: DataFrame of test Data
            
    Returns
    -----------
    df: DataFrame    
    '''
    
    binary_model = joblib.load('../../models/binary_model.joblib')
         
    # observation - row data within test set
    row = X_test.iloc[observation:observation+1,:].copy()
    
    probabilities=[]
    
    # the actual offer that was given in the test set
    actual = row.id.values[0]
    
    # loop through 10 possible offers and calculate probabilities using 
    # model    
    for i in range(10):
        row.id=i               
        if row.id.values[0] == actual:
            original = binary_model.predict_proba(row)[0].tolist()
            original.extend([row.id.values[0]])
            probabilities.append(original)
        else:
            probabilities.append(binary_model.predict_proba(row)[0])
    
    # create DataFrame of results    
    results = pd.DataFrame(probabilities)
    
    # determine if the model correctly identified the actual offer
    if y_test.iloc[observation] == results[results[2] > 1].max()[:-1].idxmax():
        results[2][actual] = 'correct'
    else:
        results[2][actual] = 'incorrect'
    
    # rename columns and compute total complete probability
    results.columns=['failed', 'complete', 'prediction']
   
    
    # join dataframe to offer id information
    portw = joblib.load('../../data/interim/portw.joblib')
    results = results.join(portw)
       
    results['age'] = row.age.values[0]
    results['income'] = row.income.values[0]
    results['gender'] = row.gender.values[0]
           
    return results

if __name__ == '__main__':
        main()

def main():
    pass
