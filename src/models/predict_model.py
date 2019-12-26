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