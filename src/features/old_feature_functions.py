def check_completion_match(transcript_df):
    '''
    Checks that total offer completions from raw data matches newly created 'complete' column
    
    '''
    
    
    raw_complete = transcript_df[transcript_df.event =="offer completed"].groupby('person').id.count()
    calculated_complete = transcript_df.groupby('person').completed.sum()

    # joining by id to match persons
    comparison_df = pd.DataFrame(calculated_complete).join(raw_complete)
    comparison_df.id.replace(np.nan,0, inplace=True)
    comparison_df.completed.replace(np.nan,0, inplace=True)

    # Checking that there are no differences between each user
    if (comparison_df.id - comparison_df.completed).value_counts()[0.0] == transcript_df.person.nunique():
        print("Completions mapped to offer rows correctly")
        return True
    else:
        print("Some completions mapped incorrectly, please compare:")
        display(comparison_df[comparison_df.completed - comparison_df.id != 0])
        return False

def check_view_match(transcript_df):
    '''
    Checks that total offer completions from raw data matches newly created 'viewed' column
    
    '''
    raw_viewed = transcript_df[transcript_df.event =="offer viewed"].groupby('person').id.count()
    calculated_viewed = transcript_df.groupby('person').viewed.sum()

    # joining by id to match persons
    comparison_df = pd.DataFrame(calculated_viewed).join(raw_viewed)
    comparison_df.id.replace(np.nan,0, inplace=True)
    comparison_df.viewed.replace(np.nan,0, inplace=True)

    # Checking that there are no differences between each user
    if (comparison_df.id - comparison_df.viewed).value_counts()[0.0] == transcript_df.person.nunique():
        print("Completions mapped to offer rows correctly")
        return True
    else:
        print("Some completions mapped incorrectly, please compare:")
        display(comparison_df[comparison_df.viewed - comparison_df.id != 0])
        return False

def checking_viewed(transcript_df, save=None):
    
    '''
    Loops forward, calculates future spending - these are more target variables

    Y variable - Spending:
    * received but not viewed
    * viwewed and received

    When viewed:
    * how many days left to complete
    * how much spending needed to complete
    * simple boolean was viewed when valid?
    '''    
    
    transcript_df['already_viewed'] = 0
    transcript_df['viewed'] = 0
       
    bar = progressbar.ProgressBar()

    for i in bar(transcript_df.index):
        if transcript_df.loc[i, 'event'] == 'offer received':

            for j in transcript_df.index[i+1:]:
                
                if transcript_df.loc[j, 'already_viewed'] == 1:
                    continue
                
                # check if still same person
                if transcript_df.loc[j, 'person'] != transcript_df.loc[i, 'person']:
                    break
                
                # check if period is within duration
                #if transcript_df.loc[j, 'date'] - transcript_df.loc[i, 'date'] > pd.Timedelta(days=transcript_df.loc[i, 'duration']):
                 #   break
                
                # if offer viewed, update how many days left in the offer, update how much remaining spending needed
                if transcript_df.loc[j, 'event'] == 'offer viewed':
                    if transcript_df.loc[j, 'id'] == transcript_df.loc[i, 'id']:
                        transcript_df.loc[i, 'viewed'] = 1
                        transcript_df.loc[j, 'already_viewed'] = 1
                        break
       
    check_view_match(transcript_df)                        
                                   
    save_file(transcript, save)          

    return transcript_df