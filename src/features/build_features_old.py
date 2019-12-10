import pandas as pd
import numpy as np
import joblib
import os
import progressbar
from src.data.make_dataset import save_file
import re
import datetime
from sklearn.preprocessing import LabelEncoder



def date_features(transcript, save=None):
    ''' 
    Create various date/time features
    
    Parameters
    -----------
    transcript: DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim' 
    
    Returns
    -------
    DataFrame
    '''
            
    transcript['time_days'] = transcript.time / 24
    transcript['date'] = transcript.signed_up + pd.to_timedelta(transcript.time_days, unit='D')
    transcript['day'] = transcript.date.dt.day
    transcript['weekday'] = transcript.date.dt.dayofweek
    transcript['month'] = transcript.date.dt.month
    transcript['year'] = transcript.date.dt.year
    
    save_file(transcript, save) 
    
    return transcript


def df_numpydict(df, df_columns):
    '''
    Converts index and specified columns of a dataframe into a dictionary of numpy arrays
    Speeds up loops.
    
    Parameters
    -----------
    df:  DataFrame
    df_columns: list of required columns names
    
    Returns
    -------
    dictionary of numpy arrays
    '''
    dict_np = {column: df[column].to_numpy() for column in df_columns}
    dict_np['index'] = df.index.to_numpy()
    return dict_np   


def try_join(new_data, transcript):
    '''
    Joins new_data DataFrame to transcript DataFrame
    Parameters
    -----------
    new_data: DataFrame
    transcript: DataFrame
            
    Returns
    -------
    Dataframe    
    
    '''
    try:
        transcript.drop(new_data.columns, axis=1, inplace=True)
    except:
        pass
    
    transcript = transcript.join(new_data)
    return transcript


def create_transaction_ranges(transcript, save=None):
    '''
    Creates time bucket fields for total transaction value and number of transactions going back in time from offer received.
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''
    transaction_days_range = [30,21,14,7,3,1]  
    hist = {}
    for m in transaction_days_range:
        transaction_range = f't_{m}'       
        transaction_range_count = f'{transaction_range}c'
        hist[transaction_range] = np.zeros(transcript.shape[0])
        hist[transaction_range_count] = np.zeros(transcript.shape[0])
    
    # convert required dataframe columns to dictionary of numpy arrays
    t = df_numpydict(transcript, ['event', 'person', 'time_days', 'amount'])
   
    bar = progressbar.ProgressBar()
    
    # loop through each row
    for i in bar(t['index']):
        if t['event'][i] =='offer received':
            
            # loop backwards through events of customer
            for j in t['index'][0:i][::-1]:
                if t['person'][j] != t['person'][i]:
                    break                
                
                # if transaction, how many days before offer received?
                if t['event'][j] == 'transaction':                                       
                    day_diff = t['time_days'][i] - t['time_days'][j]
                    
                    # loop through transaction day ranges and add increment transaction value and
                    # increment transaction count
                    for m in transaction_days_range:
                        transaction_range = 't_' + str(m)
                        transaction_range_count = transaction_range + 'c'
                        
                        if day_diff <= m:                            
                            hist[transaction_range][i] += t['amount'][j]
                            hist[transaction_range_count][i] += 1
                           
                        else:
                            break
                            
    new_data = pd.DataFrame(hist)[['t_1', 't_3', 't_7', 't_14', 't_21', 't_30',
                                     't_1c', 't_3c', 't_7c', 't_14c', 't_21c', 't_30c']]
    
    transcript = try_join(new_data, transcript)

    save_file(transcript, save)  

    return transcript


def overlap_offer_effect(transcript, save=None):
    '''
    Creates overlap offer feature columns [a,b,c,d,e,f,g,h,i,j] with integer value equal to the duration for which the previous offer is still valid.
        
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''              
    # convert required dataframe columns to dictionary of numpy arrays
    t = df_numpydict(transcript, ['event', 'person', 'time_days', 'duration', 'id'])
    
    overlap_offer = np.empty(transcript.shape[0], dtype=str)
    overlap_offer_days = np.full(transcript.shape[0], np.nan)
            
    bar = progressbar.ProgressBar()
    for i in bar(t['index']):
        if t['event'][i] == 'offer received': 

            # loop backwards through events of customer
            for j in t['index'][0:i][::-1]:
                if t['person'][j] != t['person'][i]:
                    break
                if t['event'][j] == 'offer completed':
                    break

                if t['event'][j] == 'offer received':
                    days_left = t['time_days'][j] - t['time_days'][i] + t['duration'][j]
                    
                    if days_left <= 0:
                        continue

                    overlap_offer_days[i] = days_left
                    overlap_offer[i] = t['id'][j]
                               
    offer_overlap_features = pd.get_dummies(overlap_offer, drop_first=True).mul(overlap_offer_days, axis=0).replace(0, np.nan)
    transcript = try_join(offer_overlap_features, transcript)
    
    save_file(transcript, save)  
   
    return transcript


def last_transaction(transcript, save=None):
    '''
    Creates last transaction in days and last amount spent features.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''
    
    t = df_numpydict(transcript, ['event', 'person', 'time_days', 'amount'])
    
    tran_index = transcript.index.to_numpy()
    tran_event = transcript.event.to_numpy()
    person = transcript.person.to_numpy()
    time = transcript.time_days.to_numpy()
    amount = transcript.amount.to_numpy()
    
    last_transaction_days = np.full(transcript.shape[0], np.nan)
    last_amount = np.full(transcript.shape[0], np.nan)

    bar = progressbar.ProgressBar()
    
    # loop through each row
    for i in bar(t['index']):
        if t['event'][i] =='offer received':
            
            # loop backwards through events of customer
            for j in t['index'][0:i][::-1]:
  
                if t['person'][j] != t['person'][i]:
                    break

                if tran_event[j] == 'transaction':
                    last_transaction_days[i] = t['time_days'][i] - t['time_days'][j]
                    last_amount[i] = amount[j]
                    break
    
    transcript['last_transaction_days'] = last_transaction_days
    transcript['last_amount'] = last_amount
            
    save_file(transcript, save)        

    return transcript


def viewed_received_spend(transcript, save=None):
    '''
    Creates received_spend, viewed_spend, viewed_days_left, remaining_to_complete, viewed_in_valid feautures
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''   
    t = df_numpydict(transcript, ['event', 'person', 'difficulty', 'time_days', 'duration', 'amount'])
    
    viewed_in_valid = np.zeros(transcript.shape[0])
    received_spend = np.zeros(transcript.shape[0])
    viewed_days_left = np.full(transcript.shape[0], -1)
    viewed_spend = np.zeros(transcript.shape[0])
    remaining_to_complete = np.full(transcript.shape[0], np.nan)
        
    bar = progressbar.ProgressBar()
    for i in bar(t['index']):
        if t['event'][i] == 'offer received':

            for j in t['index'][i+1:]:                        
                
                # check if still same person
                if t['person'][j] != t['person'][i]:
                    break
                
                # check if period is within duration
                if t['time_days'][j] - t['time_days'][i] > t['duration'][i]:
                    break
                
                # if offer viewed, update how many days left in the offer, update how much remaining spending needed
                if t['event'][j] == 'offer viewed':
                    viewed_in_valid[i] = 1
                                        
                    if received_spend[i] <= t['difficulty'][i]:                        
                        viewed_days_left[i] = t['duration'][i] - (t['time_days'][j] - t['time_days'][i])
                        remaining_to_complete[i] = t['difficulty'][i] - received_spend[i] - viewed_spend[i]
                    else:
                        viewed_days_left[i] = 0
                        remaining_to_complete[i] = 0
                                
                # for transactions
                if t['event'][j] == 'transaction':
                    
                    # update spending when received but not viewed                    
                    if viewed_days_left[i] < 0:
                        received_spend[i] += t['amount'][j]
                                           
                    # update spending when viewed
                    if viewed_days_left[i] >= 0:
                        viewed_spend[i] += t['amount'][j]
    
    transcript['received_spend'] = received_spend
    transcript['viewed_spend'] = viewed_spend
    transcript['viewed_days_left'] = viewed_days_left
    transcript['remaining_to_complete'] = remaining_to_complete
    transcript['viewed_in_valid'] = viewed_in_valid
    
    save_file(transcript, save)       

    return transcript


def match_verification(transcript, event=None):
    '''
    Helper function that validates whether total number of events(offer completed or offer viewed) from raw 
    data matches newly created 'complete' or 'viewed' feature.
    
    Parameters
    -----------
    transcript:  DataFrame
    event: string ('offer received' or 'offer completed')
    '''     
    
    if event == 'offer completed':
        eventid = 'completed'
    elif event == 'offer viewed':
        eventid = 'viewed'
    
    raw = transcript[transcript.event == event].groupby('person').id.count()    
    calculated = transcript.groupby('person')[eventid].sum()

    # joining by id to match persons
    comparison_df = pd.DataFrame(calculated).join(raw)
    comparison_df.id.replace(np.nan,0, inplace=True)
    comparison_df[eventid].replace(np.nan,0, inplace=True)

    # Checking that there are no differences between each user
    if (comparison_df.id - comparison_df[eventid]).value_counts()[0.0] == transcript.person.nunique():
        print(event, ' mapped to offer rows correctly')
    else:
        print('some ' + event + 'mapped incorrectly, please compare:')
        display(comparison_df[comparison_df[eventid] - comparison_df.id != 0])


def mapping_event(transcript, event=None, save=None):
    '''
    Maps the events 'offer completed' or 'offer viewed' to the corresponding 'offer received' row of the dataset
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''     
    
    t = df_numpydict(transcript, ['event', 'person', 'time_days', 'duration', 'id'])
        
    if event == 'offer completed':
        eventid = 'completed'
        event_check = 'already_completed'
    elif event == 'offer viewed':
        eventid = 'viewed'
        event_check = 'already_viewed'
    
    t[event_check] = np.zeros(transcript.shape[0])
    t[eventid] = np.zeros(transcript.shape[0])
       
    bar = progressbar.ProgressBar()

    for i in bar(t['index']):
        if t['event'][i] == 'offer received':

            for j in t['index'][i+1:]:
                
                if t[event_check][j] == 1:
                    continue
                
                # check if still same person
                if t['person'][j] != t['person'][i]:
                    break
                
                # check if period is within duration
                if event == 'offer completed':        
                    if t['time_days'][j] - t['time_days'][i] > t['duration'][i]:
                        break
                
                # if offer viewed, update how many days left in the offer, update how much remaining spending needed
                if t['event'][j] == event:
                    if t['id'][j] == t['id'][i]:
                        t[event_check][j] = 1
                        t[eventid][i] = 1
                        break
    
    transcript[event_check] = t[event_check]
    transcript[eventid] = t[eventid]
       
    match_verification(transcript, event=event)
    transcript.drop([event_check], axis=1, inplace=True)
    
    save_file(transcript, save)         

    return transcript


def feature_cleanup(transcript, save=None):
    '''
    Various features creation and redundant features dropped.
    
    Parameters
    -----------
    transcript:  DataFrame
 
    Returns
    -------
    DataFrame 
    '''
       
    # drop cumulative amount since equal to t-30
    transcript.drop(['cum_amount'], axis=1, inplace=True)
    
    # spending during offer period - to be predicted 
    transcript['offer_spend'] = transcript.received_spend + transcript.viewed_spend
    
    # adding boolean target variable to be predicted - was offer completed?
    transcript['spend>required'] = (transcript.received_spend + transcript.viewed_spend > transcript.difficulty).astype(int)
    
    # filtering event by only offer received, data now all included in this row
    transcript = transcript[transcript.event=='offer received']
    
    # dropping original event column
    transcript.drop(['event'], axis=1, inplace=True)
    
    # dropping offer_id since now each row is a unique offer
    transcript.drop(['offer_id'], axis=1, inplace=True)
    
    # dropping joined since same as signed_up
    transcript.drop('joined', axis=1, inplace=True)
    
    # removing 'amount' and 'transaction' since this data related to individual transactions
    transcript.drop(['amount', 'transaction'], axis=1, inplace=True)
    
    save_file(transcript, save) 
    return transcript


def to_numerical_nan(transcript, save=None):
    '''
    Converts date and other features to numerical in preparation . Where zero values are not correctly descriptive, converts 
    these to NaNs.
        
    Parameters
    -----------
    transcript:  DataFrame
 
    Returns
    -------
    DataFrame 
    '''
       
    # creating columns converting date time and time deltas to floats
    transcript['signed_up0'] = (transcript.signed_up - transcript.signed_up.max()).dt.days
    transcript['date0'] = (transcript.date - transcript.signed_up.max()).dt.days
        
    # replacing old date time and time delta columns with new float versions:
    transcript.signed_up = transcript.signed_up0
    transcript.date = transcript.date0

    # dropping interim new columns:
    transcript.drop(['signed_up0', 'date0'], axis=1, inplace=True)
        
    # replacing zeros with nulls for viewed_days_left and remaining to complete, since this value is only relevent if customer viewed offer
    transcript.viewed_days_left[transcript.viewed_in_valid == 0] = np.nan
    transcript['remaining_to_complete'][transcript.viewed_in_valid == 0] = np.nan
    
    save_file(transcript, save) 
    return transcript


def historical_features(transcript, save=None):
    
    '''
    Creates features based on past customer history
    '''
    
    transcript.reset_index(inplace=True, drop=True)   
    
    hist_feature_names = ['hist_reward_completed', 
                     'hist_reward_possible', 
                     'hist_difficulty_completed', 
                     'hist_difficulty_possible',
                     'hist_previous_completed',
                     'hist_previous_offers',
                     'hist_viewed_and_completed',
                     'hist_complete_not_viewed',
                     'hist_failed_complete',
                     'hist_viewed',
                     'hist_received_spend',
                     'hist_viewed_spend']
    
    hist_features = {features: np.zeros(transcript.shape[0]) for features in hist_feature_names}
    
    t = df_numpydict(transcript, ['person', 'completed', 'reward', 'difficulty', 'viewed', 'received_spend', 'viewed_spend']) 
    
    t = {**t, **hist_features}                          
    
    bar = progressbar.ProgressBar()
       
    for i in bar(t['index']):
        for j in t['index'][0:i][::-1]:
            
            # if different customer, break
            if t['person'][j] != t['person'][i]:
                break            
            
            # looping through previous offers, if completed, make additions to total features:
            if t['completed'][j] == 1: 
                t['hist_reward_completed'][i] += t['reward'][j]
                t['hist_reward_possible'][i] += t['reward'][j]
                t['hist_difficulty_completed'][i] += t['difficulty'][j]
                t['hist_difficulty_possible'][i] += t['difficulty'][j]
                t['hist_previous_completed'][i] += 1
                t['hist_previous_offers'][i] += 1
                
                # if viewed, make additions to view features:
                if t['viewed'][j] == 1: 
                    t['hist_viewed_and_completed'][i] += 1
                else:
                    t['hist_complete_not_viewed'][i] += 1
            
            # if didn't complete offer, make additions to possible features:                                        
            else:
                t['hist_reward_possible'][i] += t['reward'][j]
                t['hist_difficulty_possible'][i] += t['difficulty'][j]
                t['hist_previous_offers'][i] += 1
                t['hist_failed_complete'][i] += 1 
            
            # if viewed, make addition to viewed
            if t['viewed'][j] == 1:
                t['hist_viewed'][i] += 1
            
            # increment viewed and received spend
            t['hist_received_spend'][i] += t['received_spend'][j]
            t['hist_viewed_spend'][i] += t['viewed_spend'][j]
            
    for feature in hist_feature_names:
        transcript[feature] = t[feature]
    
    save_file(transcript, save)       
    return transcript


def column_order(transcript, save=None):
    '''
    Removing redundant features and moving target features 'offer_spend' and 'completed'
    to last columns.
        
    Parameters
    -----------
    transcript:  DataFrame
 
    Returns
    -------
    DataFrame 
    '''
    
    transcript = transcript[['person', 'age', 'income', 'signed_up', 'gender', 'id',
                'rewarded', 'difficulty', 'reward', 'duration', 'mobile', 'web',
                'social', 'bogo', 'discount', 'informational', 'time_days', 'day',
                'weekday', 'month', 'year', 't_1', 't_3', 't_7', 't_14', 't_21', 't_30',
                't_1c', 't_3c', 't_7c', 't_14c', 't_21c', 't_30c', 'last_amount', 'received_spend', 
                'viewed_spend', 'viewed_days_left', 'remaining_to_complete', 'viewed_in_valid', 
                'viewed', 'last_transaction_days', 'spend>required',
                'hist_reward_completed', 'hist_reward_possible',
                'hist_difficulty_completed', 'hist_difficulty_possible',
                'hist_previous_completed', 'hist_previous_offers',
                'hist_viewed_and_completed', 'hist_complete_not_viewed',
                'hist_failed_complete', 'hist_viewed', 'hist_received_spend',
                'hist_viewed_spend', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'offer_spend', 'completed']]
    
    save_file(transcript, save)
    return transcript


def label_encode_categories(transcript, save=None):
    '''
    Label encodes gender and id, removing previous created one hot encoding.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
           
    Returns
    -------
    DataFrame
    '''
    le = LabelEncoder()
    transcript.id = le.fit_transform(transcript.id)
    le.fit_transform(transcript.id)
    transcript['gender'] = transcript[['F', 'M', 'O']].idxmax(1)
    transcript.gender = le.fit_transform(transcript.gender)
    transcript = transcript.drop(['F', 'M', 'O'], axis=1)
    
    
    save_file(transcript, save)
    return transcript


def build_all_features(transcript, save=None, all=True):
    '''
    Runs all feature engineering functions returning processed dataframe
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
    all: Bool (default=True)
        If true saves a time stamped version of the file after each function is run.
            
    Returns
    -------
    DataFrame
    '''   
    
    def save_str(save):
        if all == True:
            saving = re.split('[./]', save)[-2] +'_' + datetime.datetime.now().strftime('%m-%d_%H_%M_%S') + '.joblib'
        else: 
            saving = None
        return saving
       
    transcript = date_features(transcript, save=save_str(save))
    print('> date features added')

    transcript = create_transaction_ranges(transcript, save=save_str(save))
    print('> transaction ranges added')

    transcript = last_transaction_and_amount(transcript, save=save_str(save))

    transcript = viewed_received_spend(transcript, save=save_str(save))
    print('> view received spending added')    
    
    transcript = overlap_offer_effect(transcript, save=save_str(save))
    print('> overlap offer effect added')
    
    transcript = mapping_event(transcript, event='offer viewed', save=save_str(save))
    print('> mapped offer viewed')
    
    transcript = mapping_event(transcript, event='offer completed', save=save_str(save))
    print('> mapped offer completed')
    
    transcript = feature_cleanup(transcript, save=save_str(save))
    print('> cleaned features')
   
    transcript = to_numerical_nan(transcript, save=save_str(save))
    print('> converted to numerical')
        
    transcript = historical_features(transcript, save=save_str(save))
    print('> historical features added')
    
    transcript = label_encode_categories(transcript,save=save_str(save))
    print('categorical variables encoded')
        
    transcript = column_order(transcript, save=save_str(save))
    print('columns reordered')
    print('All feature engineering steps completed successfully')
    
    return transcript
