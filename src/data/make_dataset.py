import pandas as pd
import numpy as np
import joblib
import os


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    return 


def main():
    print('wrangling portfolio data...')
    portfolio = wrangle_portfolio()
    
    print('wrangling profile data...')
    profile = wrangle_profile()
    
    print('wrangling transcript data...')
    transcript = wrangle_transcript()


def save_file(data, save):
    '''
    Saves DataFrame to .joblib format in folder '../../data/interim'
    
    Parameters
    -----------
    data: DataFrame
    save: string filename 
    '''
    if save:

        try:
            dirName='../../data/interim'
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ") 
        except FileExistsError:
            pass

        joblib.dump(data, dirName + '/' + save, compress=True)
        print('saved as {}'.format(dirName + '/' + save))  

    
def save_file(data, save, dirName='../../data/interim'):
    '''
    Helper function saves DataFrame to .joblib format in folder '../../data/interim'
    
    Parameters
    -----------
    data: DataFrame
    save: string filename 
    '''

    if save:
        try:
            dirName=dirName
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ") 
        except FileExistsError:
            pass

        joblib.dump(data, dirName + '/' + save, compress=True)
        print('saved as {}'.format(dirName + '/' + save)) 


def wrangle_portfolio(filepath='../../data/raw/portfolio.json', save='portfolio.joblib'):
    '''
    Wrangles and preprocess portfolio data into usable format
    
    Parameters
    -----------
    filepath: input file and path (default='../../data/raw/portfolio.json')
    save:   string filename (default='None')
            if filename entered, saves output to folder '../../data/interim'
            otherwise just returns DataFrame object
            
    Returns
    -------
    DataFrame of processed data
    '''
    
    portfolio = pd.read_json(filepath, orient='records', lines=True)

    portfolio = portfolio.join(portfolio.channels.str.join('|').str.get_dummies())
    portfolio.drop(['channels', 'email'], axis=1, inplace=True)
    portfolio = portfolio[['id', 'difficulty', 'reward', 'duration', 'offer_type', 'mobile', 'web', 'social']]
    portfolio = portfolio.join(pd.get_dummies(portfolio.offer_type))
    portfolio.drop(['offer_type'], axis=1, inplace=True)
    portfolio.sort_values(['difficulty', 'reward', 'duration'], ascending=False, inplace=True)
    id = list(portfolio['id'])
    portfolio['old_id'] = id
    portfolio.id = portfolio.id.map({a:b for a,b in zip(id, 'abcdefghij')})
    portfolio.reset_index(drop=True, inplace=True)
    
    save_file(portfolio, save)

    return portfolio

def wrangle_profile(filepath='../../data/raw/profile.json', save='profile.joblib'):
    '''
    Wrangles and preprocess profile data into usable format
    
    Parameters
    -----------
    filepath: input file and path (default='../../data/raw/profile.json')
    save:   string filename (default='None')
            if filename entered, saves output to folder '../../data/interim'
            otherwise just returns DataFrame object
            
    Returns
    -------
    DataFrame of processed data
    '''
       
    profile = pd.read_json(filepath, orient='records', lines=True)
        
    profile = profile.join(pd.get_dummies(profile.gender))
    profile.drop(['gender'], axis=1, inplace=True)
    profile.became_member_on = pd.to_datetime(profile.became_member_on, format='%Y%m%d')
    profile = profile[['id', 'age', 'income', 'became_member_on', 'F', 'M', 'O']]
    profile.rename(columns={'id': 'person'}, inplace=True)
    
    if save:
        save_file(profile, 'profile.joblib')
    
    return profile


def wrangle_transcript(filepath='../../data/raw/transcript.json', portfolio=None, profile=None, save='transcript.joblib'):

    '''
    Wrangles and preprocess transcript data into usable format
    
    Parameters
    -----------
    filepath: input file and path (default='../../data/raw/transcript.json')
    portfolio:  DataFrame (default=None)
                - if DataFrame not entered, will load '../../data/interim/profile.joblib'
    profile:    DataFrame (default=None)
                - if DataFrame not entered, will load '../../data/interim/portfolio.joblib'
    save:   string filename (default=None)
            if filename entered, saves output to folder '../../data/interim'
            
    Returns
    -------
    DataFrame of processed data
    '''
    
    if not isinstance(profile, pd.DataFrame):
        profile = joblib.load('../../data/interim/profile.joblib', mmap_mode=None)
    
    if not isinstance(portfolio, pd.DataFrame):
        portfolio = joblib.load('../../data/interim/portfolio.joblib', mmap_mode=None)

    id = portfolio.old_id
    portfolio.drop('old_id', axis=1, inplace=True)
    
    transcript = pd.read_json(filepath, orient='records', lines=True)
    
    transcript = transcript.merge(profile, on='person')
    transcript = transcript.join(pd.DataFrame(list(transcript.value)))
    transcript.drop('value', axis=1, inplace=True)
    
    transcript['offer id'] = transcript['offer id'].fillna(value="")
    transcript['offer_id'] = transcript['offer_id'].fillna(value="")
    
    transcript['offer_id'] = transcript['offer id'].map(str) + transcript.offer_id.map(str)
    transcript.drop('offer id', axis=1, inplace=True)
    
    transcript.offer_id = transcript.offer_id.map({a:b for a,b in zip(id, 'abcdefghij')})
    transcript.rename(columns={'offer_id': 'id', 'reward': 'rewarded', 'became_member_on': 'signed_up'}, inplace=True)
    transcript = transcript.merge(portfolio, how='left', on='id')
    transcript = transcript.fillna(value=0)
    transcript.income.replace({0: np.nan}, inplace=True)
    transcript.age.replace({118: np.nan}, inplace=True)
    transcript['cum_amount'] = transcript.groupby('person').amount.cumsum()
    transcript.event = pd.Categorical(transcript.event, categories=['offer received', 'offer viewed', 'offer completed', 'transaction'], ordered=True)
    transcript['offer_id'] = transcript.person + transcript.id.astype(str)
    transcript = transcript[['offer_id', 'person', 'event', 'time', 'age', 'income', 'signed_up', 'F', 'M', 'O',
            'amount', 'id', 'rewarded', 'difficulty', 'reward', 'duration', 'mobile', 'web', 
            'social', 'bogo', 'discount', 'informational', 'cum_amount']]
    transcript['offer_multi'] = transcript.offer_id + transcript.event.astype(str)
    transcript['offer_multi_correction'] = transcript.groupby('offer_multi').offer_id.apply(lambda n: n + (np.arange(len(n))+1).astype(str))
    transcript.offer_id = transcript.offer_multi_correction
    transcript.drop(['offer_multi', 'offer_multi_correction'], axis=1, inplace=True)
    transcript['joined'] = (transcript.signed_up - transcript.signed_up.max()).dt.days
    transcript = transcript.join(pd.get_dummies(transcript.event))
    transcript['id'] = pd.Categorical(transcript.id, categories=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '0'], ordered=True)
    
    if save:
        save_file(transcript, 'transcript.joblib')
    
    return transcript


if __name__ == '__main__':
        main()
       

