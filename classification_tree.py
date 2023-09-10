import pandas as pd #to easily grab the data
import numpy as np #to find the mean and std
import matplotlib.pyplot as plt #to graph
from sklearn.tree import DecisionTreeClassifier, plot_tree #to build classification tree and draw tree
from sklearn.model_selection import train_test_split, cross_val_score #for cross validation,
from sklearn.metrics import confusion_matrix, plot_confusion_matrix #to make and plot a confusion matrix.

# read the dataframe
df = pd.read_csv('processed.cleveland.data', header=None)
#print the first five rows
#print(df.head())
#add column names
df.columns = ['age', 'sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','Ca','thal','hd']


#dealing with missing data

#for x in df.columns:
#    print(df[x].dtype) #prints the type of each data in our frame.

#print(df['thal'].unique()) #prints the unique items in the thal row.

len(df.loc[(df['Ca'] =='?') #print out rows containing missing values
    |
           (df['thal'] == '?')])

df_no_missing = df.loc[(df['Ca'] != '?') & (df['thal'] != '?')]

#start formatting the data
X = df_no_missing.drop('hd', axis = 1).copy() #new copy of all columns with exception of heart disease
y = df_no_missing['hd'].copy()
#one hot encoding. converts a scale to a discrete number
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg','slope','thal'])
y_not_zero_index = y > 0
y[y_not_zero_index] = 1
#print(y.unique()) #does he have a heart attack, or don't he. no 1-5 needed.


#creating the tree
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

#plotting the tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt,
          fontsize=20,
          filled=True,
          rounded=True,
          class_names=["No heart disease","Yes Heart disease"],
          feature_names=X_encoded.columns);

#test the data by plotting a confusion matrix
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=['No HD','Yes HD'])

#cost complexity pruning
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas #extract the alpha values
ccp_alphas = ccp_alphas[:-1] #exclude the maximum levels for alpha

clf_dts = []
for ccp_alpha in ccp_alphas: #creates 1 decision tree for each alpha value and appends it to the end
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha = ccp_alpha)
    clf_dt.fit(X_train,y_train)
    clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train,y_train) for clf_dt in clf_dts] #tests each alpha score on test and train data
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('accuracy vs. alpha for training and testing sets')
ax.plot(ccp_alphas, train_scores, marker='o',label="train", drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label = 'test',drawstyle = 'steps-post')
ax.legend()
#plt.show()

#build the new decision tree:
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)
plot_confusion_matrix(clf_dt_pruned, X_test, y_test, display_labels = ['Does not have HD', 'Has HD'])
#plt.show()

#show the pruned tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          fontsize=10,
          filled=True,
          rounded=True,
          class_names=["No heart disease","Yes Heart disease"],
          feature_names=X_encoded.columns);
plt.show()


