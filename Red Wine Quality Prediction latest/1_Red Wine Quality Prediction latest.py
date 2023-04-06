#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries and Reading the data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from IPython.core.display import display, HTML
sns.set_style('darkgrid')


# In[2]:


wine_df = pd.read_csv(r'C:\Users\Lewnovo\winequality-red.csv')


# In[3]:


wine_df


# # wine_df

# In[4]:


wine_df.head()


# In[5]:


wine_df.info()


# In[6]:


# Print the names of the columns
col_name = wine_df.columns
print(col_name)


# In[7]:


wine_df.quality.value_counts()


# In[8]:


wine_df.describe()


# **1) There are 1599 rows and 12 columns**
# 
# **2) Except quality every other variable is float type and quality is integer.**

# # 2. Missing Values

# In[9]:


wine_df.describe()


# In[10]:


# the number of unique values in the dataframe
wine_df.nunique()


# In[11]:


wine_df.count()


# In[12]:


wine_df.isnull().sum()


# - To conclude there are no other missing values in the form of -- or any other form.
# - Next, let's visualize the data to confirm the above statement

# # 3.Visualization of data

# In[13]:


get_ipython().system('pip install plotly')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[14]:


#get the shape of the dataframe
wine_df.shape


# In[15]:


#get the heatmap of the missing values.
px.imshow(img = wine_df.isna(), title='Missing values(yellow: missing, blue: not missing)')


# # 4. Outlier Detection

# In[16]:


# Plot bloxplots for the following variables


#  Categorical Data
a = 6  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter


plt.figure(figsize = (15,50))
for i in col_name:
    plt.subplot(a, b, c)
    plt.boxplot(wine_df[i], whis = 5)
    plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
    #plt.title("Box plot of Year")
    c = c + 1
plt.show()


# In[17]:


# Calculate number of outliers and its percentage in each variable using Tukey's method.


# NumPy's `percentile()` method returns the 
# values of the given percentiles. In this case,
# give `75` and `25` as parameters, which corresponds 
# to the third and the first quartiles.
threshold = 1.5
for var in wine_df:
    q75, q25 = np.percentile(wine_df[var], [75 ,25])
    iqr = q75 - q25
    
    min_val = q25 - (iqr*threshold)
    max_val = q75 + (iqr*threshold)
    print("Number of outliers and its percentage in {} is: {} and {} percent".format(var,
        len((np.where((wine_df[var] > max_val) 
                      | (wine_df[var] < min_val))[0])),
        len((np.where((wine_df[var] > max_val) 
                      | (wine_df[var] < min_val))[0]))*100/1599
    ))


# In[18]:


print(col_name)


# In[19]:


from scipy.stats.mstats import winsorize

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_fixed_acidity = wine_df['fixed acidity']
plt.boxplot(original_fixed_acidity)
plt.title("original_fixed_acidity")

plt.subplot(1,2,2)
winsorized_fixed_acidity= winsorize(wine_df['fixed acidity'], (0.0, 0.05))
plt.boxplot(winsorized_fixed_acidity)
plt.title("winsorized_fixed_acidity")

plt.show()


# In[20]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_volatile_acidity = wine_df['volatile acidity']
plt.boxplot(original_volatile_acidity)
plt.title("original_volatile_acidity")

plt.subplot(1,2,2)
winsorized_volatile_acidity= winsorize(wine_df['volatile acidity'], (0.0, 0.05))
plt.boxplot(winsorized_volatile_acidity)
plt.title("winsorized_volatile_acidity")

plt.show()


# In[21]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_citric_acid = wine_df['citric acid']
plt.boxplot(original_citric_acid)
plt.title("original_citric_acid")

plt.subplot(1,2,2)
winsorized_citric_acid = winsorize(wine_df['citric acid'], (0.0, 0.05))
plt.boxplot(winsorized_citric_acid)
plt.title("winsorized_citric_acid")

plt.show()


# In[22]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_residual_sugar = wine_df['residual sugar']
plt.boxplot(original_residual_sugar)
plt.title("original_residual_sugar")

plt.subplot(1,2,2)
winsorized_residual_sugar = winsorize(wine_df['residual sugar'], (0.0, 0.1))
plt.boxplot(winsorized_residual_sugar)
plt.title("winsorized_residual_sugar")

plt.show()


# In[23]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_chlorides = wine_df['chlorides']
plt.boxplot(original_chlorides)
plt.title("original_chlorides")

plt.subplot(1,2,2)
winsorized_chlorides = winsorize(wine_df['chlorides'], (0.05, 0.1))
plt.boxplot(winsorized_chlorides)
plt.title("winsorized_chlorides")

plt.show()


# In[24]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_free_sulfur_dioxide = wine_df['free sulfur dioxide']
plt.boxplot(original_free_sulfur_dioxide)
plt.title("original_free_sulfur_dioxide")

plt.subplot(1,2,2)
winsorized_free_sulfur_dioxide = winsorize(wine_df['free sulfur dioxide'], (0.0, 0.05))
plt.boxplot(winsorized_free_sulfur_dioxide)
plt.title("winsorized_chlorides")

plt.show()


# In[25]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_total_sulfur_dioxide = wine_df['total sulfur dioxide']
plt.boxplot(original_total_sulfur_dioxide)
plt.title("original_total_sulfur_dioxide")

plt.subplot(1,2,2)
winsorized_total_sulfur_dioxide = winsorize(wine_df['total sulfur dioxide'], (0.0, 0.05))
plt.boxplot(winsorized_total_sulfur_dioxide)
plt.title("winsorized_total_sulfur_dioxide")

plt.show()


# In[26]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_density = wine_df['density']
plt.boxplot(original_density)
plt.title("original_density")

plt.subplot(1,2,2)
winsorized_density = winsorize(wine_df['density'], (0.05, 0.05))
plt.boxplot(winsorized_density)
plt.title("winsorized_density")

plt.show()


# In[27]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_pH = wine_df['pH']
plt.boxplot(original_pH)
plt.title("original_pH")

plt.subplot(1,2,2)
winsorized_pH = winsorize(wine_df['pH'], (0.05, 0.05))
plt.boxplot(winsorized_pH)
plt.title("winsorized_pH")

plt.show()


# In[28]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_sulphates = wine_df['sulphates']
plt.boxplot(original_sulphates)
plt.title("original_sulphates")

plt.subplot(1,2,2)
winsorized_sulphates = winsorize(wine_df['sulphates'], (0.0, 0.05))
plt.boxplot(winsorized_sulphates)
plt.title("winsorized_sulphates")

plt.show()


# In[29]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_alcohol = wine_df['alcohol']
plt.boxplot(original_alcohol)
plt.title("original_alcohol")

plt.subplot(1,2,2)
winsorized_alcohol = winsorize(wine_df['alcohol'], (0.0, 0.05))
plt.boxplot(winsorized_alcohol)
plt.title("winsorized_alcohol")

plt.show()


# In[30]:


plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_quality = wine_df['quality']
plt.boxplot(original_quality)
plt.title("original_quality")

plt.subplot(1,2,2)
winsorized_quality= winsorize(wine_df['quality'], (0.01, 0.05))
plt.boxplot(winsorized_quality)
plt.title("winsorized_quality")

plt.show()


# In[31]:


# Check number of Outliers after Winsorization for each variable.


winsorized_list = [winsorized_fixed_acidity, winsorized_volatile_acidity, winsorized_citric_acid, winsorized_residual_sugar, winsorized_free_sulfur_dioxide, winsorized_total_sulfur_dioxide,  
winsorized_density, winsorized_pH, winsorized_sulphates, winsorized_alcohol]
for var_win in winsorized_list:
    q75, q25 = np.percentile(var_win, [75 ,25])
    iqr = q75 - q25

    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    
    print("Number of outliers after winsorization : {}".format(len(np.where((var_win > max_val) | (var_win < min_val))[0])))


# In[32]:


# Adding winsorized variables to the data frame.

wine_df['winsorized_fixed_acidity'] =  winsorized_fixed_acidity
wine_df['winsorized_volatile_acidity'] =  winsorized_volatile_acidity
wine_df['winsorized_citric_acid'] =  winsorized_citric_acid
wine_df['winsorized_residual_sugar'] =  winsorized_residual_sugar
wine_df['winsorized_chlorides'] =  winsorized_chlorides
wine_df['winsorized_free_sulfur_dioxide'] =  winsorized_free_sulfur_dioxide
wine_df['winsorized_total_sulfur_dioxide'] =  winsorized_total_sulfur_dioxide
wine_df['winsorized_density'] =  winsorized_density
wine_df['winsorized_pH'] =  winsorized_pH
wine_df['winsorized_sulphates'] =  winsorized_sulphates
wine_df['winsorized_alcohol'] =  winsorized_alcohol


# In[33]:


wine_df


# # 5.Univariate Analysis

# In[34]:


winsorized_df = wine_df[['winsorized_fixed_acidity', 'winsorized_volatile_acidity', 'winsorized_citric_acid', 'winsorized_residual_sugar','winsorized_chlorides', 'winsorized_free_sulfur_dioxide', 'winsorized_total_sulfur_dioxide',  
'winsorized_density', 'winsorized_pH', 'winsorized_sulphates', 'winsorized_alcohol']]
winsorized_df.describe()
winsorized_df.shape
winsorized_df.columns


# In[35]:


# Extracting our target variable and creating feature list of dependant variables
target = 'quality'
features_list = list(winsorized_df.columns)


# In[36]:


wine_df[features_list].hist(bins=40, edgecolor='b', linewidth=1.0,
                          xlabelsize=8, ylabelsize=8, grid=False, 
                          figsize=(16,6), color='red')    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
plt.suptitle('Red Wine Univariate Plots', x=0.65, y=1.25, fontsize=14);


# In[37]:


wine_df[target].hist(bins=40, edgecolor='b', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False, figsize=(6,2), color='red')    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
plt.suptitle('Red Wine Quality Plot', x=0.65, y=1.25, fontsize=14); 


# # Bivariate Analysis

# In[38]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_fixed_acidity', data = wine_df)


# In[39]:


#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'winsorized_fixed_acidity', data = wine_df)


# In[40]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_volatile_acidity', data = wine_df)


# In[41]:


sns.barplot('quality', 'winsorized_volatile_acidity', data = wine_df)


# In[42]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_citric_acid', data = wine_df)


# In[43]:


sns.barplot('quality', 'winsorized_citric_acid', data = wine_df)


# In[44]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_residual_sugar', data = wine_df)


# In[45]:


sns.barplot('quality', 'winsorized_residual_sugar', data = wine_df)


# In[46]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_chlorides', data = wine_df)


# In[47]:


sns.barplot('quality', 'winsorized_chlorides', data = wine_df)


# In[48]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_free_sulfur_dioxide', data = wine_df)


# In[49]:


sns.barplot('quality', 'winsorized_free_sulfur_dioxide', data = wine_df)


# In[50]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_total_sulfur_dioxide', data = wine_df)


# In[51]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_density', data = wine_df)


# In[52]:


sns.barplot('quality', 'winsorized_density', data = wine_df)


# In[53]:


sns.boxplot('quality', 'winsorized_pH', data = wine_df)


# In[54]:


sns.barplot('quality', 'winsorized_pH', data = wine_df)


# In[55]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_sulphates', data = wine_df)


# In[56]:


sns.barplot('quality', 'winsorized_sulphates', data = wine_df)


# In[57]:


#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'winsorized_alcohol', data = wine_df)


# In[58]:


sns.barplot('quality', 'winsorized_alcohol', data = wine_df)


# # 6.Correlation Coefficient

# In[59]:


# for visualizing correlations
f, ax = plt.subplots(figsize=(10, 6))
corr = winsorized_df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="Reds",fmt='.2f',
            linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)


# In[60]:


# for visualizing correlations
f, ax = plt.subplots(figsize=(10, 6))
corr = wine_df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="Reds",fmt='.2f',
            linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)


# # 7. Create a new target "Reviews"

# In[61]:


# create a new column 'Review' with the values of 0 and 1 


reviews = []
for i in wine_df['quality']:
    if i <= 5:
        reviews.append('0')
    else:
        reviews.append('1')

winsorized_df['Reviews'] = reviews
winsorized_df.head(10)


# In[62]:


winsorized_df.columns


# In[63]:


winsorized_df['Reviews'].unique()


# In[64]:


Counter(winsorized_df['Reviews'])


# In[65]:


winsorized_df.shape


# In[66]:


x = winsorized_df.iloc[:,:11]
y = winsorized_df['Reviews']


# 8. Model - Machine Learning
# We will start modelling the data. For this Capstone we will try the following models.
# 
# A. Logistic Regression
# B. Decision Trees
# C. Random Forests
# D. SVM
# E. SVC linear kernel
# F. Stochastic Gradient Descent
# G. KNN
# We will start with splitting the data

# In[67]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# In[68]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# # A. Logistic Regression
# 
# **(i). penalty = 'none'**

# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
lr = LogisticRegression(solver='lbfgs', penalty='none', max_iter=10000, random_state=2)
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)
print(classification_report(y_test, lr_predict))


# In[70]:


test_score = lr.score(x_test, y_test)
train_score = lr.score(x_train, y_train)

print('Score on training data: ', train_score)
print('Score on test data: ', test_score)


# In[71]:


#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score_1 = accuracy_score(y_test, lr_predict)


# In[72]:


print(lr_conf_matrix)


# In[73]:


print(lr_acc_score_1*100)


# In[76]:


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[77]:


plot_confusion_matrix(cm           = np.array([[127, 49], [51, 173]]), 
                      normalize    = False,
                      target_names = ['0', '1'],
                      title        = "Confusion Matrix")


# # (ii). penalty='l2'

# In[78]:


lr_regularized = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=10000, random_state=2)
lr_regularized.fit(x_train, y_train)
lr_regularized_predict = lr_regularized.predict(x_test)
print(classification_report(y_test, lr_regularized_predict))


# In[79]:


#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_regularized_predict)
lr_acc_score = accuracy_score(y_test, lr_regularized_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)


# # B. Decision Trees

# **(i). criterion = 'gini'**

# 

# In[81]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
print(classification_report(y_test, dt_predict))


# In[82]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
print(classification_report(y_test, dt_predict))


# In[83]:


#print confusion matrix and accuracy score
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score_2 = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_acc_score_2*100)


# # C. Random Forests

# **(i). criterion = 'gini'**

# In[85]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)
print(classification_report(y_test, rf_predict))


# In[86]:


#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)


# In[87]:


# get importance
importance = rf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # "Alcohol seems like the important variable"
# 
# **(ii). criterion = 'entropy'**

# In[88]:


print(classification_report(y_test, rf_predict))


# In[89]:


#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score_2 = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score_2*100)


# In[90]:


# get importance
importance = rf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # D. SVM Classifier
# 
# **(i). kernel = 'rbf'**

# In[91]:


from sklearn.svm import SVC
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
lin_svc_predict=rf.predict(x_test)
print(classification_report(y_test, lin_svc_predict))


# In[92]:


#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, lin_svc_predict)
lin_svc_acc_score = accuracy_score(y_test, lin_svc_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)


# # (ii). kernel = 'poly'

# In[93]:


from sklearn.svm import SVC
lin_svc = SVC(kernel='poly')
lin_svc.fit(x_train, y_train)
lin_svc_predict=rf.predict(x_test)
print(classification_report(y_test, lin_svc_predict))


# In[94]:


#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, lin_svc_predict)
lin_svc_acc_score_2 = accuracy_score(y_test, lin_svc_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score_2*100)


# In[95]:


# get importance
importance = rf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # (iii). kernel = 'sigmoid'

# In[97]:


from sklearn.svm import SVC
lin_svc = SVC(kernel='sigmoid')
lin_svc.fit(x_train, y_train)
lin_svc_predict=rf.predict(x_test)
print(classification_report(y_test, lin_svc_predict))


# In[98]:


#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, lin_svc_predict)
lin_svc_acc_score = accuracy_score(y_test, lin_svc_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)


# # (iv). kernel = 'linear'

# In[99]:


rbf_svc = SVC(kernel='linear')
rbf_svc.fit(x_train, y_train)
rbf_svc_predict=rf.predict(x_test)
print(classification_report(y_test, rbf_svc_predict))


# In[100]:


rbf_svc_conf_matrix = confusion_matrix(y_test, rbf_svc_predict)
rbf_svc_acc_score = accuracy_score(y_test, rbf_svc_predict)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)


# # E. Stochastic Gradient Descent
# 
# (i). penalty=None

# In[101]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
sgd = SGDClassifier(penalty=None)
sgd.fit(x_train, y_train)
sgd_predict = sgd.predict(x_test)
print(classification_report(y_test, sgd_predict))


# In[102]:


sgd_conf_matrix = confusion_matrix(y_test, sgd_predict)
sgd_acc_score_1 = accuracy_score(y_test, sgd_predict)
print(sgd_conf_matrix)
print(sgd_acc_score_1*100)


# # (ii). penalty=l1

# In[103]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
sgd = SGDClassifier(penalty= 'l1')
sgd.fit(x_train, y_train)
sgd_predict = sgd.predict(x_test)
print(classification_report(y_test, sgd_predict))


# In[104]:


sgd_conf_matrix = confusion_matrix(y_test, sgd_predict)
sgd_acc_score_2 = accuracy_score(y_test, sgd_predict)
print(sgd_conf_matrix)
print(sgd_acc_score_2*100)


# # (iii). penality = l2

# In[105]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
sgd = SGDClassifier(penalty= 'l2')
sgd.fit(x_train, y_train)
sgd_predict = sgd.predict(x_test)
print(classification_report(y_test, sgd_predict))


# In[106]:


sgd_conf_matrix = confusion_matrix(y_test, sgd_predict)
sgd_acc_score = accuracy_score(y_test, sgd_predict)
print(sgd_conf_matrix)
print(sgd_acc_score*100)


# # (iv). penalty= 'elasticnet'

# In[107]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
sgd = SGDClassifier(penalty= 'elasticnet')
sgd.fit(x_train, y_train)
sgd_predict = sgd.predict(x_test)
print(classification_report(y_test, sgd_predict))


# In[108]:


sgd_conf_matrix = confusion_matrix(y_test, sgd_predict)
sgd_acc_score = accuracy_score(y_test, sgd_predict)
print(sgd_conf_matrix)
print(sgd_acc_score*100)


# # F. KNN
# 
# (i). weights = 'uniform'

# In[109]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_predict = knn.predict(x_test)
print(classification_report(y_test, knn_predict))


# In[110]:


knn_conf_matrix = confusion_matrix(y_test, knn_predict)
knn_acc_score = accuracy_score(y_test, knn_predict)
print(knn_conf_matrix)
print(knn_acc_score*100)


# # (ii). weights = ' distance'

# In[111]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(weights = 'distance')
knn.fit(x_train, y_train)
knn_predict = knn.predict(x_test)
print(classification_report(y_test, knn_predict))


# In[112]:


knn_conf_matrix = confusion_matrix(y_test, knn_predict)
knn_acc_score_2 = accuracy_score(y_test, knn_predict)
print(knn_conf_matrix)
print(knn_acc_score_2*100)


# # 8. Visualizing Model Performance

# In[113]:


models = [('Logistic Regression', lr_acc_score),
          ('Decision Trees', dt_acc_score_2),
          ('Random Forest', rf_acc_score_2),
          ('SVM', lin_svc_acc_score_2),
          ('Stochastic Gradient Descent', sgd_acc_score_2),
          ('KNN', knn_acc_score_2)
         ]


# In[114]:


accuracy_vis_pd = pd.DataFrame(data = models, columns=['Model', 'Accuracy(test)'])
accuracy_vis_pd


# In[115]:


accuracy_vis_pd.sort_values(by=(['Accuracy(test)']), ascending=True, inplace=True)

f, axe = plt.subplots(1,1, figsize=(24,8))
sns.barplot(x = 'Model', y='Accuracy(test)',data = accuracy_vis_pd, ax = axe)
axe.set_xlabel('Model', size=20)
axe.set_ylabel('Accuracy(test)', size=20)


# The top three machine learning models for the current dataset are the following
#        
# - 1.SVM ------------ 78.5%
#     
# - 2.Random Forest--- 78.5%
#      
# - 3.Decision Trees-- 76.5%

# 9. Conclusions:
# The red wine data was analyzed for quality with various physiochemical properties to identify what makes a ‘good’ wine.
# 
# It was found that quality of the wine is mostly correlated to alcohol
# 
# Data was split into training and test set. Machine learning models were trained in training set and tested for accuracy on the test set.
# 
# SVM model and Random Forest does the best with an accuracy of 78.5%
# 
# Machine learning also conclude that “alcohol” is the most important variable that determines the quality of wine
# 
# Future Scope: ML models accuracy is not high. The low prevalence of quality levels 3, 4 and 8 and the large distribution overlapping area stratified by quality is a reason. We do not have information about the composition of grape varieties in each wine, the mix of experts that evaluated wine quality, or the production year. Lack of information about how the dataset was created may impact the prediction of quality using the physicochemical properties as predictors.

# In[ ]:




