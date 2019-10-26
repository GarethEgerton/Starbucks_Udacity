from sklearn.model_selection import train_test_split

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