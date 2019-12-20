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


def drop_completion_features(df):
    '''
    Drops features that are directly related to the current offer
    
    Parameters
    -----------
    df: DataFrame
               
    Returns
    -------
    DataFrame
    '''
    
    df.drop(['received_spend', 
             'viewed_spend', 
             'viewed_days_left', 
             'remaining_to_complete', 
             'viewed_in_valid', 
             'viewed', 
             'viewed_in_valid', 
             'offer_spend', 
             'spend>required', 
            ], axis=1, inplace=True)
    
    return df


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
        print('train[' + str(i) + ']' , round(train_y[i].sum() / train_y[i].count(), 4), train_y[i].shape)
        print('test[' + str(i) + '] ' , round(test_y[i].sum() / test_y[i].count(), 4), test_y[i].shape)
           
    return train_X, train_y, test_X, test_y


def gridsearch_early_stopping(cv, X, y, folds, grid, cat_features=[], save=None):

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
        
    # generate data folds 
    train_X, train_y, test_X, test_y = generate_folds(cv, X, y)
    
    # iterate through specified folds
    for fold in folds:
        # assign train and test pools
        test_pool = Pool(data=test_X[fold], label=test_y[fold], cat_features=cat_features)
        train_pool = Pool(data=train_X[fold], label=train_y[fold], cat_features=cat_features)

        # creating results_df dataframe
        results_df = pd.DataFrame(columns=['params' + str(fold), 'logloss'+ str(fold), 'AUC'+ str(fold), 'iteration'+ str(fold)])

        best_score = 99999

        # iterate through parameter grid
        for params in ParameterGrid(grid):

            # create catboost classifer with parameter params
            model = CatBoostClassifier(cat_features=cat_features,
                                        early_stopping_rounds=50,
                                        task_type='GPU',
                                        custom_loss=['AUC'],
                                        iterations=3000,
                                        **params)

            # fit model
            model.fit(train_pool, eval_set=test_pool, verbose=400)

            # append results to results_df
            
            print(model.get_best_score()['validation'])
            results_df = results_df.append(pd.DataFrame([[params, model.get_best_score()['validation']['Logloss'], 
                                                          model.get_best_score()['validation']['AUC'], 
                                                          model.get_best_iteration()]], 
                                                        columns=['params' + str(fold), 'logloss' + str(fold), 'AUC' + str(fold), 'iteration' + str(fold)]))

            # save best score and parameters
            if model.get_best_score()['validation']['Logloss'] < best_score:
                best_score = model.get_best_score()['validation']['Logloss']
                best_grid = params

        print("Best logloss: ", best_score)
        print("Grid:", best_grid)

        save_file(results_df, save + str(fold) + '.joblib', dirName='../../models')
        display(results_df)



def train_test_strategy(df, split_by='person', frac=.75):
    '''
    Split data by specified train test split strategy
    
    Parameters
    -----------
    df: DataFrame
    split_by: string, optional (default='by_person')
                - 'person' - Splits data by signed_up date. Test split 
                contains latest customers to sign up with no customer overlap 
                between train and test splits. Tests ability to predict new customers.
                - 'last_offer' - Test data is separated as last offer given to 
                each customer. Ignores frac. Tests ability to predict 
                existing customers.
                - 'random' - SKlearn's train_test_split
    frac: - int, optional (default='.75') 
            if 'person', fraction of data to use as train
            
    Returns
    -------
    X_train, X_test, y_train, y_test: DataFrames
    '''
   
    if split_by=='person':
        split_date = df.signed_up.quantile(q=frac)
        test_data = df[df.signed_up >= split_date]
        train_data = df[df.signed_up < split_date]
    
    if split_by=='last_offer':
        idx = df.groupby(['person'])['time_days'].transform(max) == df['time_days']
        test_data = df[idx]
        train_data = df[~idx]
        
    if split_by=='random':
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('completed', axis=1), 
            df.completed, test_size=1-frac, random_state=42)
        return X_train, X_test, y_train, y_test
        
    y_train = train_data.completed
    X_train = train_data.drop('completed', axis=1)

    y_test = test_data.completed
    X_test = test_data.drop('completed', axis=1)
    
    print('Data split by ' + split_by)
    print('X_train', X_train.shape, round(X_train.shape[0] / (X_train.shape[0]+X_test.shape[0]), 4))
    print('X_test', X_test.shape, round(X_test.shape[0] / (X_train.shape[0]+X_test.shape[0]), 4))
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)
    
    return X_train, X_test, y_train, y_test


def show_model_results(X_test, y_test, model):
    '''
    Predicts model with X_test against y_test displaying:
    - confusion matrix
    - accuracy
    - log loss
    - classification_report
    
    Parameters
    -----------
    model: model (default='model')
    X_test: X_test data (default='X_test')
    Y_test: Y_test data (default='Y_test')

    '''
    
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print(confusion_matrix(y_test, pred), 
          ' accuracy:  ', round(accuracy_score(y_test, pred), 4),
          ' log_loss: ', round(log_loss(y_test, proba), 4)
         )
    print()
    print(classification_report(y_test, pred))
    return accuracy_score(y_test, pred)


def cv_verification(X, y):
    '''
    Check group cross validation fold function 
    Confirms whether folds are independent and ratio of positives per fold (stratification)
    
    Parameters
    -----------
    cv: X and y train data, numpy array of DataFrame
    '''
    train_fold=[]
    test_fold=[]
    total=[]
    intersect=[]
    positive_ratio=[]
    train_X=[]
    test_X=[]
    train_y, test_y = [], []
    test_lists = []
    test_overlap = []
    
    for i,j in enumerate(GroupKFold(n_splits=5).split(X_train, y_train, groups=X_train.person)):
        
        # iterating through folds
        train_X.append(X.iloc[j[0]])
        train_y.append(y.iloc[j[0]])
        test_X.append(X.iloc[j[1]])
        test_y.append(y.iloc[j[1]])
        
        # get list of persons per fold across train and test
        train_fold.append(X.iloc[j[0]].person)
        test_fold.append(X.iloc[j[1]].person)
        
        # add total persons in train and test for each fold
        total.append(X.iloc[j[0]].person.nunique() + X.iloc[j[1]].person.nunique())
        
        # check intersecion of person between train and test of each fold
        intersect.append(np.intersect1d(X.iloc[j[0]].person, X.iloc[j[1]].person))
        
        # checking ratio of completes for each test set
        positive_ratio.append(round(y.iloc[j[1]].sum() / y.iloc[j[1]].count(),3))
                
        test_lists.append(X.iloc[j[1]].person)
        
    # check overlap between each test set of folds    
    for i in range(1,5):
        test_overlap.append(np.intersect1d(test_lists[0], test_lists[i]))
                            
    print('Total unique persons across train and test: ', total)
    print('Intersection of persons across train and test: ', intersect)
    print('Percentage of positive class per split: ', positive_ratio)
    print('Test overlap with first fold: ', test_overlap)


def importances(model):
    '''
    Comparison of sorted feature imporances based on:
    1. PredictionValuesChange
    2. LossFunctionChange
    3. ShapValues
        
    Parameters
    -----------
    model: Catboost model
               
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
    df.columns = ['feature1', 'importance', 'feature2', 'loss_train', 'feature3', 'loss_test']
    
    return df


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


def label_creater(df, label_grid=None):
    
    '''
    Creates prediction label column based on label_grid dictionary criteria
        
    Parameters
    -----------
    df:  DataFrame
    label_grid: dictionary to assign labels
            
    Returns
    -------
    DataFrame
    '''  
       
    
    df['completed_not_viewed'] = ((df.completed == 1) & (df.viewed == 0)) * label_grid['completed_not_viewed']
    
    df['completed_before_viewed'] = (((df.completed == 1) & (df.viewed == 1)) 
                                       & (df.received_spend > df.difficulty)
                                     ) * label_grid['completed_before_viewed']
    
    # Required completion spending per time left < base rate
    df['complete_anyway'] = (((df.completed == 1) & (df.viewed == 1)) 
                                & (df.viewed_spend / df.viewed_days_left <= df.amount_per_day_not_offer)
                             )* label_grid['complete_anyway']
    
    # Required completion spending per time left > base rate
    df['completed_responsive'] = (((df.completed == 1) & (df.viewed == 1) & ~(df.received_spend > df.difficulty))
                                 & (((df.viewed_spend / df.viewed_days_left > df.amount_per_day_not_offer) 
                                    | (pd.isna(df.amount_per_day_not_offer) 
                                       & ~(df.received_spend > df.difficulty))
                                    )))* label_grid['completed_responsive']
    
    # Didn't complete, spending wasn't increased above base spending
    df['incomplete_responsive'] = (((df.completed == 0) & (df.viewed == 1)) 
                                     & ((((df.viewed_spend / df.viewed_days_left) > df.amount_per_day_not_offer))
                                       | ((df.viewed_spend > 0) & pd.isna(df.amount_per_day_not_offer)
                                         )))* label_grid['incomplete_responsive']
        
    df['no_complete_no_view'] = ((df.completed == 0) & (df.viewed == 0))* label_grid['no_complete_no_view']
    
    df['unresponsive'] = ((df['completed_not_viewed']==0) & 
                          (df['completed_before_viewed']==0) & 
                          (df['complete_anyway']) & 
                          (df['completed_responsive']) & 
                          (df['incomplete_responsive']) & 
                          (df['no_complete_no_view'])
                          )* label_grid['unresponsive']  
    
    
    df['label'] = df[['completed_not_viewed', 'completed_before_viewed', 'complete_anyway', 'completed_responsive',
       'incomplete_responsive', 'unresponsive', 'no_complete_no_view']].sum(axis=1)
    
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





if __name__ == '__main__':
        main()

def main():
    pass
