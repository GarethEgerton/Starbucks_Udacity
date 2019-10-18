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
  !git config --global user.email 'gareth.egerton@gmail.com'
  !git config --global use.name 'GarethEgerton'
  !git clone https://GarethEgerton:P400dellpc@github.com/GarethEgerton/Starbucks_Udacity.git
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