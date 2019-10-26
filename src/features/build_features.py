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
    '''
        
    transcript['time_days'] = transcript.time / 24
    transcript['date'] = transcript.signed_up + pd.to_timedelta(transcript.time_days, unit='D')
    transcript['day'] = transcript.date.dt.day
    transcript['weekday'] = transcript.date.dt.dayofweek
    transcript['month'] = transcript.date.dt.month
    transcript['year'] = transcript.date.dt.year

    save_file(transcript, save) 
    
    return transcript


def create_transaction_ranges(transcript, save=None):
    
    '''
    Creates time bucket fields for total transaction value and number of transactions going back in time from the offer received event.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''

    transcript['t_1'] = 0
    transcript['t_3'] = 0
    transcript['t_7'] = 0
    transcript['t_14'] = 0
    transcript['t_21'] = 0
    transcript['t_30'] = 0

    transcript['t_1c'] = 0
    transcript['t_3c'] = 0
    transcript['t_7c'] = 0
    transcript['t_14c'] = 0
    transcript['t_21c'] = 0
    transcript['t_30c'] = 0
    
    transaction_days_range = [30,21,14,7,3,1]  

    bar = progressbar.ProgressBar()
    
    # loop through each row
    for i in bar(transcript.index):
        if transcript.loc[i, 'event'] =='offer received':
            
            # loop backwards through events of customer
            for j in transcript.index[0:i+1][::-1]:
                transaction_amount = transcript.loc[j, 'amount']
                                
                if transcript.loc[j, 'person'] != transcript.loc[i, 'person']:
                    break
                
                # if transaction, how many days before offer received?
                if transcript.loc[j, 'event'] == 'transaction':                                       
                    date_diff = transcript.loc[i, 'date'] - transcript.loc[j, 'date']
                    
                    # loop through transaction day ranges and add increment transaction value and
                    # increment transaction count
                    for m in transaction_days_range:
                        transaction_range = 't_' + str(m)
                        transaction_range_count = transaction_range + 'c'
                        
                        if date_diff <= pd.Timedelta(days=m):
                            transcript.loc[i, transaction_range] += transaction_amount
                            transcript.loc[i, transaction_range_count] += 1
                        else:
                            break
    
    save_file(transcript, save)  

    return transcript


def last_transaction_and_amount(transcript, save=None):
    '''
    Creates last transaction and last amount spend features.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''
    
    transcript['last_transaction'] = 0 
    transcript['last_amount'] = 0

    bar = progressbar.ProgressBar()
    
    # loop through each row
    for i in bar(transcript.index):
        if transcript.loc[i, 'event'] =='offer received':
            
            # loop backwards through events of customer
            for j in transcript.index[0:i+1][::-1]:

                transaction_amount = transcript.loc[j, 'amount']
    
                if transcript.loc[j, 'person'] != transcript.loc[i, 'person']:
                    break

                if transcript.loc[j, 'event'] == 'transaction':
                    transcript.loc[i, 'last_transaction'] = transcript.loc[j, 'date']
                    transcript.loc[i, 'last_amount'] = transcript.loc[j, 'amount']
                    break
    
    save_file(transcript, save)        

    return transcript


def viewed_received_spend(transcript, save=None):
    '''
    Creates received_spend, viewed_spend, viewed_days_left, remaining_to_complete, viewed_in_valid feaures
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''   
    
    transcript['received_spend'] = 0
    transcript['viewed_spend'] = 0
    transcript['viewed_days_left'] = pd.Timedelta(days=-1)
    transcript['remaining_to_complete'] = 0
    transcript['viewed_in_valid'] = 0
   
    bar = progressbar.ProgressBar()

    for i in bar(transcript.index):
        if transcript.loc[i, 'event'] == 'offer received':

            for j in transcript.index[i+1:]:                        
                
                # check if still same person
                if transcript.loc[j, 'person'] != transcript.loc[i, 'person']:
                    break
                
                # check if period is within duration
                if transcript.loc[j, 'date'] - transcript.loc[i, 'date'] > pd.Timedelta(days=transcript.loc[i, 'duration']):
                    break
                
                # if offer viewed, update how many days left in the offer, update how much remaining spending needed
                if transcript.loc[j, 'event'] == 'offer viewed':
                    transcript.loc[i, 'viewed_in_valid'] = True
                                        
                    if transcript.loc[i, 'received_spend'] <= transcript.loc[i, 'difficulty']:
                        transcript.loc[i, 'viewed_days_left'] = pd.Timedelta(days=transcript.loc[i, 'duration']) - (transcript.loc[j, 'date'] - transcript.loc[i, 'date'])
                        transcript.loc[i, 'remaining_to_complete'] = transcript.loc[i, 'difficulty'] - transcript.loc[i, 'received_spend'] - transcript.loc[i, 'viewed_spend']
                    else:
                        transcript.loc[i, 'viewed_days_left'] = pd.Timedelta(days=0)
                        transcript.loc[i, 'remaining_to_complete'] = 0
                                
                # for transactions
                if transcript.loc[j, 'event'] == 'transaction':
                    
                    # update spending when received but not viewed                    
                    if transcript.loc[i, 'viewed_days_left'] < pd.Timedelta(days=0):
                        transcript.loc[i, 'received_spend'] += transcript.loc[j, 'amount']
                                           
                    # update spending when viewed
                    if transcript.loc[i, 'viewed_days_left'] >= pd.Timedelta(days=0):
                        transcript.loc[i, 'viewed_spend'] += transcript.loc[j, 'amount']
       
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
    if event == 'offer viewed':
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
    
    if event == 'offer completed':
        eventid = 'completed'
        event_check = 'already_completed'
    if event == 'offer viewed':
        eventid = 'viewed'
        event_check = 'already_viewed'
       
    transcript[event_check] = 0
    transcript[eventid] = 0
       
    bar = progressbar.ProgressBar()

    for i in bar(transcript.index):
        if transcript.loc[i, 'event'] == 'offer received':

            for j in transcript.index[i+1:]:
                
                if transcript.loc[j, event_check] == 1:
                    continue
                
                # check if still same person
                if transcript.loc[j, 'person'] != transcript.loc[i, 'person']:
                    break
                
                # check if period is within duration
                if event == 'offer completed':        
                    if transcript.loc[j, 'date'] - transcript.loc[i, 'date'] > pd.Timedelta(days=transcript.loc[i, 'duration']):
                        break
                
                # if offer viewed, update how many days left in the offer, update how much remaining spending needed
                if transcript.loc[j, 'event'] == event:
                    if transcript.loc[j, 'id'] == transcript.loc[i, 'id']:
                        transcript.loc[j, event_check] = 1
                        transcript.loc[i, eventid] = 1
                        break
       
    match_verification(transcript, event=event)
    transcript.drop([event_check], axis=1, inplace=True)
    save_file(transcript, save)         

    return transcript

def days_since_transaction(transcript, save=None):
    '''
    Calculates number of days since last transaction for each offer received.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''   
    
    transcript['last_transaction_days'] = 0 
    bar = progressbar.ProgressBar()
    for i, j in bar(transcript.iterrows()):
        if j.last_transaction != 0:
            transcript.loc[i, 'last_transaction_days'] = (j.date - j.last_transaction).days      
    
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
    
    # replacing last transaction 0 with NaN since gradient boosting algorithms will be able to differentiate NaNs
    transcript.last_transaction.replace(0, np.nan, inplace=True)
    transcript.last_transaction_days[transcript.last_transaction.isnull()] = np.nan
    transcript.last_transaction = pd.to_datetime(transcript.last_transaction)
    transcript['last_tran'] = (transcript.last_transaction - transcript.signed_up.max()).dt.days
    transcript['last_tran_days'] = transcript.date0 - transcript.last_tran
    
    # replacing old date time and time delta columns with new float versions:
    transcript.signed_up = transcript.signed_up0
    transcript.date = transcript.date0
    transcript.last_transaction = transcript.last_tran
    transcript.last_transaction_days = transcript.last_tran_days

    # dropping interim new columns:
    transcript.drop(['signed_up0', 'date0', 'last_tran', 'last_tran_days'], axis=1, inplace=True)

    # converting viewed_days_left to integers
    transcript.viewed_days_left = transcript.viewed_days_left.dt.days

    # converting viewed_in_valid boolean to integer
    transcript.viewed_in_valid = transcript.viewed_in_valid *1
    
    # converting viewed_in_valid to integer
    transcript.viewed_in_valid = transcript.viewed_in_valid.astype(int)
        
    # replacing zeros with nulls for viewed_days_left and remaining to complete, since this value is only relevent if customer viewed offer
    transcript.viewed_days_left[transcript.viewed_in_valid == 0] = np.nan
    transcript['remaining_to_complete'][transcript.viewed_in_valid == 0] = np.nan
    
    save_file(transcript, save) 
    return transcript

def historical_features(transcript, save=None):
    
    '''
    Creates features based on past customer history
    '''
    transcript['hist_reward_completed'] = 0 
    transcript['hist_reward_possible'] = 0 
    transcript['hist_difficulty_completed'] = 0 
    transcript['hist_difficulty_possible'] = 0 
    transcript['hist_previous_completed'] = 0 
    transcript['hist_previous_offers'] = 0
    transcript['hist_viewed_and_completed'] = 0 
    transcript['hist_complete_not_viewed'] = 0 
    transcript['hist_failed_complete'] = 0
    transcript['hist_viewed'] = 0
    transcript['hist_received_spend'] = 0
    transcript['hist_viewed_spend'] = 0
    
    transcript.reset_index(inplace=True, drop=True)                          
    bar = progressbar.ProgressBar()
    
    
    for i in bar(transcript.index):
        for j in transcript.index[0:i][::-1]:
            
            # if different customer, break
            if transcript.loc[j, 'person'] != transcript.loc[i, 'person']:
                break            
            
            # looping through previous offers, if completed, make additions to total features:
            if transcript.loc[j, 'completed'] == 1: 
                transcript.loc[i, 'hist_reward_completed'] += transcript.loc[j, 'reward']
                transcript.loc[i, 'hist_reward_possible'] += transcript.loc[j, 'reward']
                transcript.loc[i, 'hist_difficulty_completed'] += transcript.loc[j, 'difficulty']
                transcript.loc[i, 'hist_difficulty_possible'] += transcript.loc[j, 'difficulty']
                transcript.loc[i, 'hist_previous_completed'] += 1
                transcript.loc[i, 'hist_previous_offers'] += 1
                
                # if viewed, make additions to view features:
                if transcript.loc[j, 'viewed'] == 1: 
                    transcript.loc[i, 'hist_viewed_and_completed'] += 1
                else:
                    transcript.loc[i, 'hist_complete_not_viewed'] += 1
            
            # if didn't complete offer, make additions to possible features:                                        
            else:
                transcript.loc[i, 'hist_reward_possible'] += transcript.loc[j, 'reward']
                transcript.loc[i, 'hist_difficulty_possible'] += transcript.loc[j, 'difficulty']
                transcript.loc[i, 'hist_previous_offers'] += 1
                transcript.loc[i, 'hist_failed_complete'] += 1 
            
            # if viewed, make addition to viewed
            if transcript.loc[j, 'viewed'] == 1:
                transcript.loc[i, 'hist_viewed'] += 1
            
            # increment viewed and received spend
            transcript.loc[i, 'hist_received_spend'] += transcript.loc[j, 'received_spend']
            transcript.loc[i, 'hist_viewed_spend'] += transcript.loc[j, 'viewed_spend']
            
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
                'social', 'bogo', 'discount', 'informational', 'time_days', 'date', 'day',
                'weekday', 'month', 'year', 't_1', 't_3', 't_7', 't_14', 't_21', 't_30',
                't_1c', 't_3c', 't_7c', 't_14c', 't_21c', 't_30c', 'last_amount', 'received_spend', 
                'viewed_spend', 'viewed_days_left', 'remaining_to_complete', 'viewed_in_valid', 
                'viewed', 'last_transaction_days', 'spend>required',
                'hist_reward_completed', 'hist_reward_possible',
                'hist_difficulty_completed', 'hist_difficulty_possible',
                'hist_previous_completed', 'hist_previous_offers',
                'hist_viewed_and_completed', 'hist_complete_not_viewed',
                'hist_failed_complete', 'hist_viewed', 'hist_received_spend',
                'hist_viewed_spend','offer_spend', 'completed']]
    
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
       
    transcript = src.features.build_features.date_features(transcript, save=save_str(save))
    print('> date features added')

    transcript = src.features.build_features.create_transaction_ranges(transcript, save=save_str(save))
    print('> transaction ranges added')

    transcript = src.features.build_features.last_transaction_and_amount(transcript, save=save_str(save))

    transcript = src.features.build_features.viewed_received_spend(transcript, save=save_str(save))
    print('> view received spending added')    
    
    transcript = src.features.build_features.mapping_event(transcript, event='offer viewed', save=save_str(save))
    print('> mapped offer viewed')
    
    transcript = src.features.build_features.mapping_event(transcript, event='offer completed', save=save_str(save))
    print('> mapped offer completed')

    transcript = src.features.build_features.days_since_transaction(transcript, save=save_str(save))
    print('> added last transaction days')
    
    transcript = src.features.build_features.feature_cleanup(transcript, save=save_str(save))
    print('> cleaned features')
   
    transcript = src.features.build_features.to_numerical_nan(transcript, save=save_str(save))
    print('> converted to numerical')
    
    transcript = src.features.build_features.historical_features(transcript, save=save_str(save))
    print('> historical features added')
    
    transcript = src.features.build_features.label_encode_categories(transcript,save=save_str(save))
    print('categorical variables encoded')
        
    transcript = src.features.build_features.column_order(transcript, save=save_str(save))
    print('columns reordered')
    
    return transcript
