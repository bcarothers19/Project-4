import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import linear_model, cross_validation, preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# load the dataset
su = pd.read_csv('evergreen.tsv',delimiter='\t')

# examine the data
su.head()
su.describe()
su.dtypes
# clean up the data
su.alchemy_category_score = pd.to_numeric(su.alchemy_category_score,
                                            errors='coerce')
su.is_news = pd.to_numeric(su.is_news, errors='coerce')
su.news_front_page = pd.to_numeric(su.news_front_page, errors='coerce')
# delete rows with NaNs
su.dropna(axis=0,how='any',inplace=True)

# use sklearn's logistic regression function to look at variable significance
feature_cols = ['alchemy_category_score','avglinksize','commonlinkratio_1',
        'commonlinkratio_2','commonlinkratio_3','commonlinkratio_4',
        'compression_ratio','embed_ratio','framebased','frameTagRatio',
        'hasDomainLink','html_ratio','image_ratio','is_news',
        'lengthyLinkDomain','linkwordscore','news_front_page',
        'non_markup_alphanum_characters','numberOfLinks','numwords_in_url',
        'parametrizedLinkRatio','spelling_errors_ratio']
X = su[feature_cols]
y = su['label']

# Run a logistic regression predicting evergreen from the numeric columns
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,
                                                                test_size=.2)
log = linear_model.LogisticRegression()
lm = log.fit(X_train,y_train)
predicted = lm.predict(X_test)
acc = accuracy_score(predicted,y_test)
print 'Acc. of predicting evergreen from numeric columns: %f' % acc

# Run a logistic regression predicting evergreen from the numeric columns and a
# categorical variable of alchemy_category
'''Originally created dummy variables for categorical variable alchemy_category
but it is better to map'''
#X['alchemy_category'] = su.loc[:,'alchemy_category']
#dummies = pd.get_dummies(X,columns=['alchemy_category'])
'''Using a map:'''
nums = range(13)
cats = su.alchemy_category.unique()
map_dict = {}
for cat,num in zip(cats,nums):
    map_dict[cat] = num
X['alchemy_category'] = su.alchemy_category.map(map_dict)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,
                                                                test_size=.2)
log = linear_model.LogisticRegression()
lm = log.fit(X_train,y_train)
predicted = lm.predict(X_test)
acc = accuracy_score(predicted,y_test)
print '\nAcc. of predicting evergreen from numeric columns and alchemy_category: %f' % acc

# Use sklearn to to cross-validate the accuracy of the model above
log = linear_model.LogisticRegressionCV(cv=12)
lm = log.fit(X,y)
predicted = lm.predict(X)
acc = accuracy_score(predicted,y)
print '\nAcc. after using cross-validation on the above model: %f' % acc

# Gridsearch regularization parameters for logistic regression
# Feature scaling using Normalizer
norm = preprocessing.Normalizer()
X_norm = norm.fit_transform(X)
params = {'C':[.1,.2,.3,.4,.5,.6,.7,.8,.9,1],'penalty':['l1','l2']}
log = linear_model.LogisticRegression()
grid = GridSearchCV(log,params,scoring='accuracy',cv=12)
grid.fit(X_norm,y)
best_params = grid.best_params_
acc = grid.best_score_
print '\nAcc. of using GridSearchCV on LogReg with scaled feature columns using Normalizer(): %f' % acc
print 'Parameters used to obtain this accuracy score: %s' % str(best_params)

# Feature scaling using Scale
X_scale = preprocessing.scale(X)
params = {'C':[.1,.2,.3,.4,.5,.6,.7,.8,.9,1],'penalty':['l1','l2']}
log = linear_model.LogisticRegression()
grid = GridSearchCV(log,params,scoring='accuracy',cv=12)
grid.fit(X_scale,y)
best_params = grid.best_params_
acc = grid.best_score_
print '\nAcc. of using GridSearchCV on LogReg with scaled feature columns using scale(): %f' % acc
print 'Parameters used to obtain this accuracy score: %s' % str(best_params)

# Grid search neighbors for kNN
k_range = range(1, 100)
param_grid = dict(n_neighbors=k_range)

knn = KNeighborsClassifier(n_neighbors=5)
grid = GridSearchCV(knn,param_grid,cv=12,scoring='accuracy')
grid.fit(X,y)
best_params = grid.best_params_
acc = grid.best_score_
print '\nAcc. of using GridSearchCV on kNN: %f' % acc
print 'Parameters used to obtain this accuracy score: %s' % str(best_params)

print '\nGraph of kNN accuracy for different values of k:'
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
plt.plot(k_range, grid_mean_scores)
plt.xlabel("K for KNN")
plt.ylabel("Accuracy Cross-Validated")
plt.show()

'''Choose a new target from alchemy_category to predict
with logistic regression'''
# Create dummies on alchemy_category to be able to predict one specific category
X = su[feature_cols]
X['alchemy_category'] = su.loc[:,'alchemy_category']
dummies = pd.get_dummies(X,columns=['alchemy_category'])
y = dummies['alchemy_category_recreation']
X = su[feature_cols]

# Run a logistic regression predicting if alchemy_category = recreation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,
                                                                test_size=.2)
log = linear_model.LogisticRegression()
lm = log.fit(X_train,y_train)
predicted = lm.predict(X_test)
acc = accuracy_score(predicted,y_test)
print '\nAcc. of predicting if alchemy_category = recreation: %f' % acc

# Use sklearn to to cross-validate the accuracy of the model above
log = linear_model.LogisticRegressionCV(cv=12)
lm = log.fit(X,y)
predicted = lm.predict(X)
acc = accuracy_score(predicted,y)
print '\nAcc. after using cross-validation on the above model: %f' % acc

# Gridsearch regularization parameters for logistic regression
# Feature scaling with Normalizer
norm = preprocessing.Normalizer()
X_norm = norm.fit_transform(X)
params = {'C':[.1,.2,.3,.4,.5,.6,.7,.8,.9,1],'penalty':['l1','l2']}
log = linear_model.LogisticRegression()
grid = GridSearchCV(log,params,scoring='accuracy',cv=12)
grid.fit(X_norm,y)
best_params = grid.best_params_
acc = grid.best_score_
print '\nAcc. of using GridSearchCV on LogReg with scaled feature columns using Normalizer(): %f' % acc
print 'Parameters used to obtain this accuracy score: %s' % str(best_params)
# Feature scaling with Scale
X_scale = preprocessing.scale(X)
params = {'C':[.1,.2,.3,.4,.5,.6,.7,.8,.9,1],'penalty':['l1','l2']}
log = linear_model.LogisticRegression()
grid = GridSearchCV(log,params,scoring='accuracy',cv=12)
grid.fit(X_scale,y)
best_params = grid.best_params_
acc = grid.best_score_
print '\nAcc. of using GridSearchCV on LogReg with scaled feature columns using scale(): %f' % acc
print 'Parameters used to obtain this accuracy score: %s' % str(best_params)
# Grid search neighbors for kNN
k_range = range(1, 100)
param_grid = dict(n_neighbors=k_range)
knn = KNeighborsClassifier(n_neighbors=5)
grid = GridSearchCV(knn,param_grid,cv=12,scoring='accuracy')
grid.fit(X,y)
best_params = grid.best_params_
acc = grid.best_score_
print '\nAcc. of using GridSearchCV on kNN: %f' % acc
print 'Parameters used to obtain this accuracy score: %s' % str(best_params)
