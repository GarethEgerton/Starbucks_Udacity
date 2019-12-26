import pandas as pd
import numpy as np
import joblib
import os
import progressbar
import re
import datetime

from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import save_file

def main():
    """ Runs feature engineering scripts to turn basic cleaned data from (../interim) into
    engineered data ready for training (savied in ../interim).
    """
    transcript = joblib.load('../../data/interim/transcript.joblib', mmap_mode=None)
    transcript = build_features(transcript, save='transcript_final_optimised.joblib', all=True)

def person_data(person):
    '''
    Displays unique customer's event history
    
    Parameters
    -----------
    person: if int then customer index as per the order in which 
    customer appears in transcript data, if string then person 
    referenced by their unique 'person' id           
    '''
    if type(person) == str:
        return transcript[transcript.person == person]
    return transcript[transcript.person == transcript.person.unique()[person]]

def plot_person(person):
    '''
    Plots visualisation of customer's event history.

    Parameters
    -----------
    person: if int then customer index as per the order in which 
    customer appears in transcript data. If string then person 
    referenced by their unique 'person' id                
    '''
    x=[]
    y=[]    
    markers = ['o', 'v', '^', 'o']
    
    for i, event in enumerate(['transaction', 'offer received', 'offer viewed', 
                               'offer completed']):
        # step plot of cumulative transactions
        if event=='transaction':
            x.append(person_data(person).time_days)
            y.append(person_data(person).cum_amount)               
            plt.step(x[i], y[i], alpha=.3, label=event, color='black', where='post')
            
        # scatter plot of events
        else:
            try:
                x.append(person_data(person)[person_data(person).event==event].time_days)
                y.append(person_data(person)[person_data(person).event==event].cum_amount)
                plt.scatter(x[i], y[i], alpha=0.7, label=event, marker=markers[i], s=80)
            except:
                pass
            
        # required spending per time for each offer received     
        if event=='offer received':
                      
            received = person_data(person)[person_data(person).event=='offer received']\
                                          [['time_days', 'difficulty', 'cum_amount', 'duration']]\
                                          .reset_index()
            
            for i in received.index:
                x_diff = [received.iloc[i].time_days, 
                          received.iloc[i].time_days+received.iloc[i].duration]
                y_diff = [received.iloc[i].cum_amount, 
                          received.iloc[i].cum_amount+received.iloc[i].difficulty]
                plt.plot(x_diff, y_diff, color='blue', alpha=.15, linewidth='2')
    
    plt.xlabel('time(days)')
    plt.ylabel('cumulative spend')
    plt.title(f'Customer event history - {person_data(person).person.max()}')
    plt.legend()
    plt.show()

def date_features(transcript, save=None):
    ''' 
    Create various date/time features
    
    Parameters
    -----------
    transcript: DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim' 
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
    Converts index and specified columns of a dataframe into a 
    dictionary of numpy arrays. Speeds up loop operations. 
    
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
    if columns already exist in transcript, drops them before joining.
        
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
    Creates time bucket fields for total transaction value and number 
    of transactions going back in time from offer received.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
    Returns
    -------
    DataFrame
    '''
    
    transaction_days_range = [30,21,14,7,3,1]  
    hist = {}
    
    # initialsiing 
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

def last_transaction(transcript, save=None):
    '''
    Creates last transaction in days and last amount spent features.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
            
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
    Creates received_spend, viewed_spend, viewed_days_left, 
    remaining_to_complete, viewed_in_valid feautures
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''   
    t = df_numpydict(transcript, ['event', 'person', 'difficulty', 'time_days', 
                                  'duration', 'amount', 'id'])
    
    viewed_in_valid = np.zeros(transcript.shape[0])
    received_spend = np.zeros(transcript.shape[0])
    viewed_days_left = np.full(transcript.shape[0], 0.0)
    viewed_spend = np.zeros(transcript.shape[0])
    remaining_to_complete = np.full(transcript.shape[0], np.nan)
    viewed_already = np.zeros(transcript.shape[0])
        
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
                
                # if offer viewed, update how many days left in the 
                # offer, update how much remaining spending needed
                if t['event'][j] == 'offer viewed':                
                    if (viewed_already[j] != 1) and (t['id'][i] == t['id'][j]):
                        viewed_in_valid[i] = 1

                        if received_spend[i] <= t['difficulty'][i]:                        
                            viewed_days_left[i] = t['duration'][i]\
                                                  - (t['time_days'][j] - t['time_days'][i])
                            
                            remaining_to_complete[i] = t['difficulty'][i] - received_spend[i]
                            
                            if remaining_to_complete[i] < 0:
                                remaining_to_complete[i] =0
                            
                            viewed_already[j] = 1

                        if received_spend[i] > t['difficulty'][i]:  
                            viewed_days_left[i] = 0
                            remaining_to_complete[i] = 0
                            viewed_already[j] = 1
                     
                if t['event'][j] == 'transaction':
                    
                    # update spending when received but not viewed      
                    if viewed_days_left[i] <=0:
                        received_spend[i] += t['amount'][j]                        
                                           
                    # update spending when viewed
                    if viewed_days_left[i] > 0:
                        viewed_spend[i] += t['amount'][j]
    
    transcript['received_spend'] = received_spend
    transcript['viewed_spend'] = viewed_spend
    transcript['viewed_days_left'] = viewed_days_left
    transcript['remaining_to_complete'] = remaining_to_complete
    transcript['viewed_in_valid'] = viewed_in_valid
    
    save_file(transcript, save)       

    return transcript

def overlap_offer_effect(transcript, save=None):
    '''
    Creates overlap offer feature columns [a,b,c,d,e,f,g,h,i,j] with 
    integer value equal to the duration for which the previous offer 
    is still valid.
        
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
            
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
                               
    offer_overlap_features = pd.get_dummies(overlap_offer, drop_first=True)\
                                            .mul(overlap_offer_days, axis=0).replace(0, np.nan)
    transcript = try_join(offer_overlap_features, transcript)
    
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
    Maps the events 'offer completed' or 'offer viewed' to the corresponding 'offer received' 
    row of the dataset
    
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
    t[event + '_date'] = np.full(transcript.shape[0], np.nan)
    
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
                
                # if offer viewed, update how many days left in the offer, update how much 
                # remaining spending needed
                if t['event'][j] == event:                                       
                    
                    if t['id'][j] == t['id'][i]:
                        t[event_check][j] = 1
                        t[eventid][i] = 1
                        t[event + '_date'][i] = t['time_days'][j] 
                        break
    
    transcript[event_check] = t[event_check]
    transcript[eventid] = t[eventid]
    transcript[event + '_date'] = t[event + '_date']
       
    match_verification(transcript, event=event)
    transcript.drop([event_check], axis=1, inplace=True)
    
    save_file(transcript, save)         

    return transcript

def start_end_range(transcript, save=None):
    '''
    Calculates date ranges when offers are under influence for each 
    offer received.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''
    
    bar = progressbar.ProgressBar()
    
    t = df_numpydict(transcript, ['offer viewed_date', 'time_days', 'viewed_days_left', 
                                  'offer completed_date', 'difficulty']) 
    
    t['start_range'] = np.full(transcript.shape[0], np.nan)
    t['end_range'] = np.full(transcript.shape[0], np.nan)
    
    for i in bar(t['index']):
        
        # Since informational offers cannot be completed, these are 
        # ignored
        if t['difficulty'][i] == 0:
            continue
        
        # if offer not viewed, start and end dates are NaN
        if not t['offer viewed_date'][i]:
            t['start_range'][i] = np.nan
            t['end_range'][i] = np.nan
            continue
        
        # if viewed on last day, only valid on that day
        if t['viewed_days_left'][i] == 0:
            t['end_range'][i] = t['time_days'][i]
            t['start_range'][i] = t['time_days'][i]
            continue
        
        # if completed, influence range is from viewed to completed date
        if t['offer completed_date'][i] >= 0:
            t['start_range'][i] = t['offer viewed_date'][i]
            t['end_range'][i] = t['offer completed_date'][i]
            continue
        else:
            # if not completed, influence range is from viewed date to 
            # end of offer duration
            t['end_range'][i] = t['offer viewed_date'][i] + t['viewed_days_left'][i]
            t['start_range'][i] = t['offer viewed_date'][i]
            continue
            
    transcript['start_range'] = t['start_range']
    transcript['end_range'] = t['end_range']
            
    return transcript

def ranges(transcript, save=None):
    '''
    Creates features:
    
    amount_while_offer 
    amount_not_offer
    percentage_offer_active 
    offer_active_count    
    no_offer_count       
    offer_cum_amount 
    no_offer_cum_amount 
    amount_per_day_offer 
    amount_per_day_not_offer 
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame
    '''

    bar = progressbar.ProgressBar()
    
    t = df_numpydict(transcript, ['offer viewed_date', 'time_days', 'viewed_days_left', 
                                  'offer completed_date', 'difficulty', 'person', 'start_range', 
                                  'end_range', 'amount']) 
        
    # create base np array of 120 discreet points from 0 to 30 to 
    # represent each possible time increment
    base = np.arange(0, 30.25, .25)

    # create empty list for each offer in transcript
    ranges = [[] for i in range(transcript.shape[0])]
    
    # instansiate a list that will hold an array for each offer of the 
    # 120 discreet time increments possible
    discreet_ranges = []   
    
    # for each offer loop back in time and append array of start and 
    # end ranges to list within list of ranges
    for i in bar(t['index']):
        for j in t['index'][0:i][::-1]:
            
            # if different person break
            if t['person'][j] != t['person'][i]:
                break 
            
            if t['start_range'][j] >= 0:
                ranges[i].append(np.array([t['start_range'][j], t['end_range'][j]]))
       
    for i in ranges:
        # if no offer ranges, add empty list to discreet ranges
        if i == []:
            discreet_ranges.append([])
            continue
        
        # if there is only one range of offer influence
        if len(i) == 1:
            
            # create array of booleans where base array of time 
            # increments satisfies influence range condition
            offer_range = np.where((base >= i[0][0]) & (base <= i[0][1]))[0]/4
            discreet_ranges.append(offer_range)
            continue
        
        # if more than one range of offer influences
        if len(i) >1:
            offer_range = np.empty(0)
            
            # update array of booleans where base array of time
            # increments satisfies all range conditions
            for j in i:
                offer_range = np.append(offer_range, np.where((base >= j[0]) & (base <= j[1]))[0]/4)
        
        # in the case of overlapping influences, only appends unique
        # time increments
        discreet_ranges.append(np.unique(offer_range))
 
    
    offers_active_list = []
    for i in discreet_ranges:
        offer_active = np.full(121, np.nan)
        
        # if offer has no historical influence, append full array of
        # NaNs
        if len(i) == 0:
            offers_active_list.append(offer_active)
            continue         
        
        # if offer has historical influence, replace boolean value from 
        # discreet ranges indicating positive influence with the actual 
        # time value in offers active 
        if len(i) > 0:
            for j in i:
                offer_active[int(j*4)] = j
            offers_active_list.append(offer_active)
     
    t['amount_while_offer'] = np.zeros(transcript.shape[0])
    t['percentage_offer_active'] = np.zeros(transcript.shape[0])
    t['amount_not_offer'] = np.zeros(transcript.shape[0])
    t['offer_active_count'] = np.zeros(transcript.shape[0])
    t['no_offer_count'] = np.zeros(transcript.shape[0])
    
    for i in t['index']:
        time = t['time_days'][i]
        
        if time == 0:
            t['amount_while_offer'][i] = t['amount'][i]
            
            # percentage of discreet time increments that offer is active
            t['percentage_offer_active'][i] = (offers_active_list[i][0:int(time*4)] >= 0).sum()/1
            continue            
        
        
        if time in offers_active_list[i]:
            t['amount_while_offer'][i] = t['amount'][i]
        else:
            t['amount_not_offer'][i] = t['amount'][i]
        
        # calculate percentage of time offers active, as well as absolute length of time offers
        # active
        t['percentage_offer_active'][i] = (offers_active_list[i][0:int(time*4 +1)] >= 0)\
                                          .sum()/(time*4 +1)
        t['offer_active_count'][i] = (offers_active_list[i][0:int(time*4 +1)] >= 0).sum()
        t['no_offer_count'][i] = (~(offers_active_list[i][0:int(time*4 +1)] >= 0)).sum()
    
         
    # add new features to transcript    
    transcript['offers_active'] = offers_active_list
    transcript['amount_while_offer'] = t['amount_while_offer']
    transcript['percentage_offer_active'] = t['percentage_offer_active']
    transcript['amount_not_offer'] = t['amount_not_offer']
    transcript['offer_active_count'] = t['offer_active_count'] / 4
    transcript['no_offer_count'] = t['no_offer_count'] / 4
    
    # calculate cumulative sum amount per person
    transcript['offer_cum_amount'] = transcript.groupby('person').amount_while_offer.cumsum()
    transcript['no_offer_cum_amount'] = transcript.groupby('person').amount_not_offer.cumsum()
    
    # convert cumulative amount to per day average
    transcript['amount_per_day_offer'] = transcript.offer_cum_amount\
                                         .divide(transcript.offer_active_count)
    transcript['amount_per_day_not_offer'] = transcript.no_offer_cum_amount\
                                         .divide(transcript.no_offer_count)
         
    return transcript

def time_buckets(transcript, save=None):
    '''
    Creates time bucket fields for total transaction value within that
    period and which specific offer (if any) was given during that 
    time period.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
    Returns
    -------
    DataFrame
    '''
    transcript.reset_index(inplace=True)
    
    day_buckets = [7, 14, 17, 21, 24, 30]
    
    # creates list of amount and offer bucket column names
    amount_buckets = [f'amount_{bucket}' for bucket in day_buckets]
    offer_buckets = [f'offer_{bucket}' for bucket in day_buckets]
    
    # instansiate a dictionaries of numpy arrays for each bucket
    # column name
    amount_buckets_dict = {bucket: np.zeros(transcript.shape[0]) for bucket in amount_buckets}
    offer_buckets_dict = {bucket: np.full(transcript.shape[0], '') for bucket in offer_buckets}
    
    # merges both dictionaries
    buckets = {**amount_buckets_dict, **offer_buckets_dict}
    
    t = df_numpydict(transcript, ['person', 'time_days', 'event', 'amount', 'id'])
    
    bar = progressbar.ProgressBar()
    
    for i in bar(t['index']):
        
        # loop backwards from each row
        for j in t['index'][0:i][::-1]:
            
            # check if same person
            if t['person'][j] != t['person'][i]:
                break            
            
            # if time_days is below bucket day value, increase the 
            # amount of that bucket. Goes to next incremental bucket 
            # amount if not below and checks again. 
            if t['event'][j] == 'transaction':
                for k in day_buckets:
                    if t['time_days'][j] <= k:
                        amount_buckets_dict[f'amount_{k}'][i] += t['amount'][j]
                        break
            
            # if time_days equauls bucket day value, add offer id to 
            # that time bucket
            if t['event'][j] == 'offer received':
                for k in day_buckets:
                    if t['time_days'][j] == k:
                        offer_buckets_dict[f'offer_{k}'][i] = t['id'][j]
                        break
        
        # replaces amount features with NaN if time_days is below
        # time bucket values. Distinguishes a zero value from an
        # impossible temporal value.
        for m,n in enumerate(day_buckets):
            if t['time_days'][i] < n:
                for z in day_buckets[m+1:]:
                    amount_buckets_dict[f'amount_{z}'][i] = np.nan
                    
    
    # adds new created features to DataFrame
    for bucket in amount_buckets_dict:
        transcript[bucket] = amount_buckets_dict[bucket]
    
    for bucket in offer_buckets_dict:
        transcript[bucket] = offer_buckets_dict[bucket]
  
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
    transcript['spend>required'] = (transcript.received_spend + transcript.viewed_spend\
                                    > transcript.difficulty).astype(int)
    
    # filtering event by only offer received, data now all included in this row
    transcript = transcript[transcript.event=='offer received']
    
    # dropping original event column
    transcript.drop(['event'], axis=1, inplace=True)
    
    # dropping offer_id since now each row is a unique offer
    transcript.drop(['offer_id'], axis=1, inplace=True)
    
    # dropping joined since same as signed_up
    transcript.drop('joined', axis=1, inplace=True)
    
    # removing 'amount' and 'transaction' since this data related to 
    # individual transactions
    transcript.drop(['amount', 'transaction'], axis=1, inplace=True)
    
    save_file(transcript, save) 
    return transcript

def to_numerical_nan(transcript, save=None):
    '''
    Converts date and other features to numerical in preparation. 
    Where zero values are not correctly descriptive, converts these 
    to NaNs.
        
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
        
    # replacing zeros with nulls for viewed_days_left and remaining to complete, since this value 
    # is only relevent if customer viewed offer
    
    # transcript.viewed_days_left[transcript.viewed_in_valid == 0] = np.nan
    transcript['remaining_to_complete'][transcript.viewed_in_valid == 0] = np.nan
    
    save_file(transcript, save) 
    return transcript

def historical_features(transcript, save=None):
    '''
    Creates historical features.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
    Returns
    -------
    DataFrame
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
    
    t = df_numpydict(transcript, ['person', 'completed', 'reward', 'difficulty', 'viewed', 
                                  'received_spend', 'viewed_spend']) 
    
    t = {**t, **hist_features}                          
    
    bar = progressbar.ProgressBar()
       
    for i in bar(t['index']):
        for j in t['index'][0:i][::-1]:
            
            # if different customer, break
            if t['person'][j] != t['person'][i]:
                break            
            
            # looping through previous offers, if completed, make 
            # additions to total features:
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
            
            # if didn't complete offer, make additions to possible 
            # features:                                        
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

def label_encode_categories(transcript, save=None):
    '''
    Label encodes gender and id, removing previous created one hot 
    encoding.
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
           
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

def previous_offer_features(transcript, save=None):
    '''
    Creates previous offers historical features:
    
    offer_received_spend
    offer_viewed_spend
    offer_counts
    
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
    Returns
    -------
    DataFrame
    '''
    
    transcript.reset_index(inplace=True, drop=True)   
    
    offer_received_spend = [f'received_{i}' for i in list('0123456789')]  
    offer_viewed_spend = [f'viewed_{i}' for i in list('0123456789')]
    offer_counts = [f'count_{i}' for i in list('0123456789')]
    
    offer_received_spend_dict = {offer: np.full(transcript.shape[0], np.nan) for offer in 
                                 offer_received_spend}
    offer_viwed_spend_dict = {offer: np.full(transcript.shape[0], np.nan) for offer in 
                              offer_viewed_spend}
    offer_counts_dict = {offer: np.full(transcript.shape[0], np.nan) for offer in offer_counts}
    
    t = df_numpydict(transcript, ['person', 'completed', 'viewed_spend', 'received_spend', 'id']) 
    
    offers = {**offer_received_spend_dict, **offer_viwed_spend_dict, **offer_counts_dict}                          
    
    bar = progressbar.ProgressBar()
       
    for i in bar(t['index']):
        for j in t['index'][0:i][::-1]:
            
            # if different customer, break
            if t['person'][j] != t['person'][i]:
                break            
            
            # looping through previous offers, if completed, make additions to total features:
            previous_offer = t['id'][j]
            
            if offers[f'received_{previous_offer}'][i] >= 0:
                offers[f'received_{previous_offer}'][i] += t['received_spend'][j]
            else:
                offers[f'received_{previous_offer}'][i] = t['received_spend'][j]
            
            if offers[f'viewed_{previous_offer}'][i] >= 0:
                offers[f'viewed_{previous_offer}'][i] += t['viewed_spend'][j]
            else:
                offers[f'viewed_{previous_offer}'][i] = t['viewed_spend'][j]
                       
            if t['completed'][j]:
                if offers[f'count_{previous_offer}'][i] >= 1:
                    offers[f'count_{previous_offer}'][i] += 1
                else:
                    offers[f'count_{previous_offer}'][i] = 1
                            
    for offer in offers:
        transcript[offer] = offers[offer]
    
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
                'hist_viewed_spend', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',                              
                'received_0', 'received_1', 'received_2',
                'received_3', 'received_4', 'received_5', 'received_6', 'received_7',
                'received_8', 'received_9', 'viewed_0', 'viewed_1', 'viewed_2',
                'viewed_3', 'viewed_4', 'viewed_5', 'viewed_6', 'viewed_7', 'viewed_8',
                'viewed_9', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4',
                'count_5', 'count_6', 'count_7', 'count_8', 'count_9',
                'amount_7', 'amount_14', 'amount_17',
                'amount_21', 'amount_24', 'amount_30', 'offer_7', 'offer_14',
                'offer_17', 'offer_21', 'offer_24', 'offer_30',
                'offer_spend', 'completed', 'percentage_offer_active', 'offer_active_count',
                'no_offer_count', 'offer_cum_amount', 'no_offer_cum_amount', 
                'amount_per_day_offer','amount_per_day_not_offer',
                 ]]                
                    
    save_file(transcript, save)
    return transcript

def build_features(transcript, save=None, all=False):

    '''
    Runs all feature engineering functions returning processed dataframe
        
    Parameters
    -----------
    transcript:  DataFrame
    save:   string filename (default=None)
            if filename entered, saves output to folder 
            '../../data/interim'
    all: Bool (default=True)
        If true saves a time stamped version of the transcript file 
        after each function is run.
        
            
    Returns
    -------
    DataFrame
    '''   
    
    def save_str(save):
        if save and all:
            saving = re.split('[./]', save)[-2] +'_' + datetime.datetime.now()\
                     .strftime('%m-%d_%H_%M_%S') + '.joblib'
        else:
            saving = None
        return saving
   
    transcript = date_features(transcript, save=save_str(save))
    print('> date features added')

    transcript = create_transaction_ranges(transcript, save=save_str(save))
    print('> transaction ranges added')

    transcript = last_transaction(transcript, save=save_str(save))

    transcript = viewed_received_spend(transcript, save=save_str(save))
    print('> view received spending added')    
    
    transcript = overlap_offer_effect(transcript, save=save_str(save))
    print('> overlap offer effect added')
    
    transcript = mapping_event(transcript, event='offer viewed', save=save_str(save))
    print('> mapped offer viewed')
        
    transcript = mapping_event(transcript, event='offer completed', save=save_str(save))
    print('> mapped offer completed')
    
    transcript = start_end_range(transcript, save=save_str(save))
    transcript = ranges(transcript, save=save_str(save))
    print('> ranges completed')
    
    transcript = time_buckets(transcript, save=save_str(save))
    print('> time buckets completed')
    
    transcript = feature_cleanup(transcript, save=save_str(save))
    print('> cleaned features')
   
    transcript = to_numerical_nan(transcript, save=save_str(save))
    print('> converted to numerical')
    
    transcript = historical_features(transcript, save=save_str(save))
    print('> historical features added')
    
    transcript = label_encode_categories(transcript, save=save_str(save))
    print('categorical variables encoded')
    
    transcript = previous_offer_features(transcript, save=save_str(save))
    print('adding previous offers')
        
    transcript = column_order(transcript, save=save)
    print('columns reordered')
    print('All feature engineering steps completed successfully')
    
    return transcript


if __name__ == '__main__':
    main()