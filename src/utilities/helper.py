def pull():
  '''
  Mounts Google Drive
  Moves to Colab Notebook director
  Trys to download repo from Github
  If already exists pulls and rebases
  '''

  #from google.colab import drive
  #drive.mount('/content/drive/')
  print('''
  #!git clone https://GarethEgerton:P400dellpc@github.com/GarethEgerton/Starbucks_Udacity.git
  !git config --global user.email 'gareth.egerton@gmail.com'
  !git config --global use.name 'GarethEgerton'
  !git pull --rebase origin master
  ''')


def push():
  '''
  Adds all changes to staging area
  Commits
  Pushes to remote repo
  '''

  print('''
  !git add --all
  !git commit -m message
  !git push origin master
  ''')


def docstring():
  '''
  Prints a docstring template
  '''

  print('''
    Train test split strategy to utilise 
    
    Parameters
    -----------
    df: DataFrame
    split_method: string, optional (default='by_person')
                  - 'person' - Splits data by signed_up date. Test split 
                  contains latest customers to sign up with no customer overlap 
                  between train and test splits. Tests ability to predict new customers.
                  - 'last_offer' - Test data is separated as last offer given to 
                  each customer. Ignores split_percentage. Tests ability to predict 
                  existing customers.
                  - 'random' - SKlearn's train_test_split
    split_percentage: - int, optional (default='.75') 
                  if 'by_person', percentage of data to use as train
            
    Returns
    -------
    X_train, X_test, y_train, y_test: DataFrames
  ''')
  
  
