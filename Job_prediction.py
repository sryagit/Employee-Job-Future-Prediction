#!/usr/bin/env python
# coding: utf-8

# In[63]:


# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,classification_report

pd.set_option('display.max_columns', None)


# In[80]:


# Load dataset into a dataframe
df0 = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\HR_analytics_jobPrediction.zip")

# Display first few rows of the dataframe
df0.head(10)


# In[5]:


df0.info()


# In[6]:


df0.describe()


# In[7]:


print('Departments: ') 
print(df0['Department'].describe())
print('Departments unique names: ')
print(*list(df0['Department'].unique()), sep=', ')


# For categorical features
print('\nSalary: ') 
print(df0['salary'].describe())
print('Salary unique categories: ')
print(*list(df0['salary'].unique()), sep=', ')


# In[8]:


df0.columns


# In[9]:


df0.columns = ['satisfaction','evaluation','number_project','monthly_hours','tenure','work_accident',
              'left','promoted','department','salary']
df0.head()


# In[10]:


df0[df0.duplicated()].sort_values(by=['satisfaction', 'evaluation','number_project', 'monthly_hours'])


# In[11]:


df0.isna().sum()
#
#
# In[12]:


df0[df0.duplicated()].sort_values(by=['satisfaction', 'evaluation','number_project', 'monthly_hours'])


# In[13]:


# Drop duplicates and save resulting dataframe in a new variable as needed

df1 = df0.drop_duplicates().reset_index(drop=True)

print('Shape of dataset: ', df0.shape)
print('Shape of dataset without duplicates: ', df1.shape)

# Display first few rows of new dataframe as needed
df1.head(10)


# In[14]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
threshold = 1.5

plt.figure(figsize=(8,2))
sns.set_style("whitegrid") 

#whis is Proportion of the IQR past the low and high quartiles to extend the plot whiskers
bp = sns.boxplot(x = df1['tenure'], whis=threshold, color='#22A884FF')
bp.set(title='Box plot for Tenure in Company')
bp.set_xlabel('years', fontsize=11)

plt.show()


# In[15]:


# Determine the number of rows containing outliers
q1 = df1.tenure.quantile(0.25)
q3 = df1.tenure.quantile(0.75)
iqr = q3 - q1

outliers = df1[(df1['tenure'] < q1 - threshold * iqr) | (df1['tenure'] > q3 + threshold * iqr)]

print('Q1: %.0f'% q1)
print('Q3: %.0f'% q3)
print("IQR: %.2f"% iqr)
print('Upper limit: %.1f'% (q3 + threshold * iqr))
print('Lower limit: %.1f'% (q1 - threshold * iqr))


outliers


# In[16]:


# Get numbers of people who left vs. stayed
print(df1.left.value_counts(),'\n')

# Get percentages of people who left vs. stayed
print(df1.left.value_counts(normalize=True))


# In[17]:


sns.pairplot(df1, hue='left',plot_kws={"s": 1},palette='viridis')


# In[18]:


df1


# In[19]:


sns.set_style('white')

corr_matr = df1.drop(columns=['department','salary']).corr()

plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(corr_matr, dtype='bool'))
heatmap = sns.heatmap(corr_matr, mask=mask,vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle correlation heatmap', fontdict={'fontsize':12}, pad=12);


# In[21]:


# get_ipython().run_cell_magic('time', '', "sns.regplot(data=df1, x='monthly_hours', y='left', ci=90, n_boot=5, logistic=True, color='#2A788EFF')\nplt.title('Stayed\\left by monthly_hours')\nplt.show()\n")


# In[22]:


max_hours_stayed = df1[df1['left']==0]['monthly_hours'].max()
print('Maximum monthly working hours for people who stayed in company: ',max_hours_stayed)

min_hours_left = df1[df1['left']==1]['monthly_hours'].min()
print('Minimum monthly working hours for people who left the company: ',min_hours_left)


# In[23]:


sns.regplot(data=df1, x='satisfaction', y='left',color='#2A788EFF')
plt.title('Stayed\left by satisfaction')
plt.show()


# In[24]:


max_satisfaction = df1[df1['left']==1]['satisfaction'].max()
print('Maximum satisfaction for those, who left the company: ',max_satisfaction)

min_satisfaction = df1[df1['left']==0]['satisfaction'].min()
print('Minimum satisfaction for people who stayed in company: ',min_satisfaction)


# In[25]:


fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(131)
sns.boxplot(data=df1, x='left', y='satisfaction',palette='viridis')
plt.title('Satisfaction with stayed/left')
ax2 = fig.add_subplot(132)
sns.boxplot(data=df1, x='left', y='evaluation',palette='viridis')
plt.title('Evaluation with stayed/left')
ax3 = fig.add_subplot(133)
sns.boxplot(data=df1, x='left', y='monthly_hours',palette='viridis')
plt.title('Monthly hours with stayed/left')
plt.show()


# In[26]:


# fig = plt.figure(figsize=(12, 5))
# x1 = fig.add_subplot(121)
# sns.boxplot(data=df1, x='number_project', y='satisfaction', palette='viridis')
# plt.title('Satisfaction by number of projects')


# ax2 = fig.add_subplot(122)
# sns.boxplot(data=df1, x='number_project', y='monthly_hours', palette='viridis')
# plt.title('Monthly hours by number of projects')


plt.show()


# In[27]:


pd.crosstab(df1.number_project,df1.left).plot.bar(rot=0,color=['#238A8DFF','#DCE319FF'])
plt.title('Number of projects through left/stayed')
plt.show()


# In[28]:


projects = df1.groupby(['number_project']).agg({'number_project':'count','left':'sum'}).rename(columns={"number_project": "employees"})
projects['percentage']=round(projects['left']/projects['employees']*100,2)
projects = projects.sort_values(by='percentage')
projects


# In[29]:


df1_promoted_left = df1[(df1['left']==1) & (df1['promoted']==1)]
print(f'Number of employees, promoted and left: {df1_promoted_left.shape[0]}')
df1_promoted_left


# In[30]:


promotion = df1.groupby(['promoted']).agg({'promoted':'count','left':'sum'}).rename(columns={"promoted": "employees"})
promotion['percentage']=round(promotion['left']/promotion['employees']*100,2)
promotion = promotion.sort_values(by='percentage')
promotion


# Only 3.94 % (8 employees) of all promoted was left, while for unpromoted employees 1983 from 11788 = 16.82% left

# In[31]:


pd.crosstab(df1.salary,df1.left).plot(kind='bar',color=['#238A8DFF','#DCE319FF'])
plt.title('Employees through levels of salary')
plt.show()


# In[32]:


# salary = df1.groupby(['salary']).agg({'salary':'count','left':'sum'}).rename(columns={"salary": "employees"})
# salary['percentage']=round(salary['left']/salary['employees']*100,2)
# salary = salary.sort_values(by='percentage')
# salary


# In[33]:


pd.crosstab(df1.department,df1.left).plot.bar(rot=45,color=['#238A8DFF','#DCE319FF'])
plt.title('Employees through departments')
plt.show()


# In[34]:


departments = df1.groupby(['department']).agg({'department':'count','left':'sum'}).rename(columns={"department": "employees"})
departments['percentage']=round(departments['left']/departments['employees']*100,2)
departments = departments.sort_values(by='percentage')
departments


# In[35]:


sns.set_color_codes(palette="colorblind")
sns.barplot(x=list(departments.index), y=departments["percentage"],
            label="Total", color="#29AF7FFF")
plt.xticks(rotation=45, ha='right')
plt.title('Turnover rate level through departments')
plt.show()


# In[36]:


# fig = plt.figure(figsize=(12, 5))
# x1 = fig.add_subplot(221)
# sns.boxplot(data=df1, x='department', y='satisfaction', palette='viridis')
# plt.title('Satisfaction by departments')
# plt.xticks(rotation=45, ha='right')

# ax2 = fig.add_subplot(222)
# sns.boxplot(data=df1, x='department', y='monthly_hours', palette='viridis')
# plt.title('Monthly hours by departments')


plt.xticks(rotation=45, ha='right')
plt.show()


# In[37]:


df_1 = df1.copy()
ordinal_salary = {'low': 0, 'medium':1, 'high':2}
df_1['salary'] = df_1['salary'].map(ordinal_salary)
df_1 = pd.get_dummies(df_1, ['department'])


# In[38]:


tenure_mask = (q1 - threshold * iqr <= df_1['tenure']) & (df_1['tenure'] <= q3 + threshold * iqr)
df_1 = df_1[tenure_mask]


# In[39]:


print('Number of examples: ',df1.shape[0])
print('Number of examples without outliers in tenure: ',df_1.shape[0])


# In[40]:


df_1.head()


# In[41]:


y = df_1.left
X = df_1.drop(columns=['left'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=42)


# In[42]:


# get_ipython().run_cell_magic('time', '', "\ncolumns = ['satisfaction', 'evaluation', 'number_project', 'monthly_hours',\n       'tenure', 'work_accident', 'promoted', 'salary', 'department_IT',\n       'department_RandD', 'department_accounting', 'department_hr',\n       'department_management', 'department_marketing',\n       'department_product_mng', 'department_sales', 'department_support',\n       'department_technical']\n\n\nlr_model = LogisticRegression(random_state=42, max_iter=500).fit(X_train[columns], y_train)\n")


# In[45]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[60]:


models = [LogisticRegression(),SVC(),GaussianNB()]


# In[78]:


import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# In[81]:


def check_models():
    for model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f'Accuracy of {model} :{round(acc,2)*100}%')
        
        print(classification_report(y_test, pred))
        
        print("---------------------------------------------------------------------")

check_models()


# ### Accuracy of Naive Bayes model is 84.0%

# In[ ]:


import pickle

# Train your model
naive_bayes_model = GaussianNB().fit(X_train, y_train)

# Save the model
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(naive_bayes_model, f)


