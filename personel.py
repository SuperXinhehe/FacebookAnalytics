#! /usr/bin/python3.5
## data preprocessing:
from os import listdir
from os.path import isfile,join
import csv
import pandas as pd
import numpy as np 

from nltk.stem import PorterStemmer
from nltk.tokenize import SpaceTokenizer
from nltk.corpus import stopwords
from functools import partial

import re
import string

textpath = "/Users/XinheLovesMom/Google Drive/TCSS555/Train/Text"

mypath = textpath.replace("Text","Profile/")+"Profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

## aggregate the data for personality
pertb = {}
for row in profiletb:
    pertb[row.get("userid")] = [row.get("ope"),row.get("con"),row.get("ext"),row.get("agr"),row.get("neu")]

perl = [(key,value[0],value[1],value[2],value[3],value[4]) for key,value in pertb.items()]
pertb = pd.DataFrame(perl)
pertb.columns = ["usrid","ope","con","ext","agr","neu"]

## get the text cleaned docs
lines = (line.rstrip() for line in open("/Users/XinheLovesMom/docs.txt"))
docs = [line for line in lines]

## function to help with selecting the best features
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False) 

from sklearn import cross_validation


### get the vec
def getTextVec(best1,person_text):
  docs = person_text["text"]
  textl = docs.tolist()
  docs = []
  for doc in textl:
  	docs.append(" ".join(doc))
  docs2 = [doc.replace("\t","").replace("\n","") for doc in docs]
  ## vec and tokenize function
  vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=best1)
  X = vec.fit_transform(docs2)
  return (X)

### five cross validation for each persontality
## 5 cross validation with max features 50000
def getRMSEPersonal(model,numTerms,X,person_text):
  print("Number of Top Features: " + str(numTerms))
  selector = SelectKBest(f_regression,k=numTerms)
  print("finished data preprocessing")
  for i in range(5):
    colname = person_text.columns[i+1]
    print("personality: " + colname)
    real = [float(n) for n in person_text[colname]]
    allvec_new = selector.fit_transform(X.toarray(),real)
    ## cross validation method: 
    print("getting scores....from 5-fold-cross-validation")
    scores = cross_validation.cross_val_score(model, allvec_new, real, scoring='mean_squared_error', cv=5)
    print("MEAN SQUARED ERROR: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


### function to get the top featured terms:
## input top: selector, vec
def getFeaturedTerms(top,selector,vec):
  top_ranked_features = sorted(enumerate(selector.scores_),key=lambda x:x[1], reverse=True)
  top_ranked_features_indices = list(map(list,zip(*top_ranked_features)))[0]
  features1 = []
  for feature_pvalue in zip(np.asarray(vec.get_feature_names())[top_ranked_features_indices],selector.pvalues_[top_ranked_features_indices]):
    features1.append(feature_pvalue)
  return (features1)

### function to get the accuracy and roc based on training and testing datasets:
X = getTextVec(20000,person_text)
# def get1FoldPersonel(model,numTerms,X,person_text):

  
#### model is svm support vector regression rfb kernel function and numTerms = 200 
getRMSEPersonal(model,100,X,person_text)

# Number of Top Features: 200
# finished data preprocessing
# personality: ope
# MEAN SQUARED ERROR: -0.40 (+/- 0.05)
# personality: con
# MEAN SQUARED ERROR: -0.51 (+/- 0.04)
# personality: ext
# MEAN SQUARED ERROR: -0.66 (+/- 0.04)
# personality: agr
# MEAN SQUARED ERROR: -0.44 (+/- 0.02)
# personality: neu
# MEAN SQUARED ERROR: -0.63 (+/- 0.04)
getRMSEPersonal(model,100,person_text)
# Number of Top Features: 100
# finished data preprocessing
# personality: ope
# MEAN SQUARED ERROR: -0.40 (+/- 0.05)
# personality: con
# MEAN SQUARED ERROR: -0.51 (+/- 0.04)
# personality: ext
# MEAN SQUARED ERROR: -0.66 (+/- 0.04)
# personality: agr
# MEAN SQUARED ERROR: -0.43 (+/- 0.03)
# personality: neu
# MEAN SQUARED ERROR: -0.63 (+/- 0.04)
getRMSEPersonal(model,1000,person_text)

# Number of Top Features: 500
# finished data preprocessing
# personality: ope
# MEAN SQUARED ERROR: -0.40 (+/- 0.05)
# personality: con
# MEAN SQUARED ERROR: -0.51 (+/- 0.04)
# personality: ext
# MEAN SQUARED ERROR: -0.66 (+/- 0.04)

# Number of Top Features: 1000
# finished data preprocessing
# personality: ope
# MEAN SQUARED ERROR: -0.40 (+/- 0.05)
# personality: con
# MEAN SQUARED ERROR: -0.52 (+/- 0.04)
# personality: ext
# MEAN SQUARED ERROR: -0.66 (+/- 0.04)
# personality: agr
# MEAN SQUARED ERROR: -0.44 (+/- 0.03)
# personality: neu
# MEAN SQUARED ERROR: -0.63 (+/- 0.04)


#### k = 20000,100
# getRMSEPersonal(model,100,X,person_text)
# Number of Top Features: 100
# finished data preprocessing
# personality: ope
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.40 (+/- 0.05)
# personality: con
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.51 (+/- 0.04)
# personality: ext
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.66 (+/- 0.04)
# personality: agr
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.43 (+/- 0.03)
# personality: neu
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.63 (+/- 0.04)

### pca for regression
pca = sklearn.decomposition.TruncatedSVD(n_components=80)
data = pca.fit_transform(X)
regression = sklearn.linear_model.LinearRegression()
regression.fit(data[:6650], real[:6650])
# 0.357019732792


### function linear regression:and pca with 100
X = getTextVec(20000,person_text)
regression = sklearn.linear_model.LinearRegression()

### using pca rather than selectkbest 
def getRMSEPCA5Fold(model,X,top):
	pca = sklearn.decomposition.TruncatedSVD(n_components=top)
	data = pca.fit_transform(X)
	print("finished data preprocessing")
	for i in range(5):
		colname = person_text.columns[i+1]
		print("personality: " + colname)
		real = [float(n) for n in person_text[colname]]
		scores = cross_validation.cross_val_score(model, data, real, scoring='mean_squared_error', cv=5)
		print("MEAN SQUARED ERROR: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		print(str(scores.tolist()))

getRMSEPCA5Fold(regression,X,100)


# finished data preprocessing linear regression
# personality: ope
# MEAN SQUARED ERROR: -0.38 (+/- 0.06)
# [-0.4230901416644035, -0.43023643177483994, -0.40465381845745924, -0.35557094704236786, -0.3645395788306402, -0.3668578209459978, -0.36356498075455335, -0.3491517009978647, -0.3413159577618458, -0.37533665285296736]
# personality: con
# MEAN SQUARED ERROR: -0.49 (+/- 0.06)
# [-0.5474631824619722, -0.48299124116710496, -0.5059600154972871, -0.46329370026132954, -0.536215791046882, -0.4940228333167081, -0.4891956004895142, -0.44529504317340335, -0.4952944951784414, -0.44985862350654077]
# personality: ext
# MEAN SQUARED ERROR: -0.63 (+/- 0.04)
# [-0.6729436934193758, -0.6545665385306285, -0.6404261818210419, -0.6110325243350414, -0.616545306790338, -0.636336295422962, -0.6095640885062181, -0.6051344413830451, -0.640063564752014, -0.6072285477619045]
# personality: agr
# MEAN SQUARED ERROR: -0.43 (+/- 0.03)
# [-0.4642615970586449, -0.42946108263132404, -0.4398990002064132, -0.41330774588955016, -0.4181424876053962, -0.4044453492318017, -0.43210267000907976, -0.4069658304780665, -0.4385507501504884, -0.419059526306155]
# personality: neu
# MEAN SQUARED ERROR: -0.62 (+/- 0.05)
# [-0.6662273729456879, -0.6434201936000667, -0.5940652249677971, -0.624781072977465, -0.5813460707423039, -0.6233148837508136, -0.5961114135654452, -0.5941824798810886, -0.620532659822629, -0.6287729799629508]
getRMSEPCA5Fold(regression,X,80)
#### naivebayes regression
from sklearn import linear_model
clf = linear_model.BayesianRidge() ##****
getRMSEPCA5Fold(clf,X,100)
# finished data preprocessing
# personality: ope
# MEAN SQUARED ERROR: -0.38 (+/- 0.05)
# [-0.4225980699300548, -0.3751252272264671, -0.36577080458621164, -0.35520590873665475, -0.35731482597360703]
# personality: con
# MEAN SQUARED ERROR: -0.49 (+/- 0.04)
# [-0.5156150892179648, -0.48526105100800143, -0.5179052313644522, -0.46414056495723427, -0.4719391625009413]
# personality: ext
# MEAN SQUARED ERROR: -0.63 (+/- 0.04)
# [-0.6601774766688119, -0.6180095044866141, -0.625742950206195, -0.6037987921779123, -0.6211810530877442]
# personality: agr
# MEAN SQUARED ERROR: -0.42 (+/- 0.02)
# [-0.44173480413357646, -0.42701943869829073, -0.41043685700019983, -0.41637845143733593, -0.4249915382671742]
# personality: neu
# MEAN SQUARED ERROR: -0.61 (+/- 0.04)
# [-0.6520303946560417, -0.6036208951317565, -0.6021407459705803, -0.5914654666860518, -0.6224991612869271]
clf = linear_model.Lasso(alpha = 0.1) ## bad
logistic = linear_model.LogisticRegression()

from sklearn.svm import SVR
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
getRMSEPCA5Fold(svr_lin,X,100)
getRMSEPCA5Fold(svr_poly,X,100)
# personality: ope
# MEAN SQUARED ERROR: -0.38 (+/- 0.06)
# [-0.43806405544737775, -0.3818848665963067, -0.371584153928297, -0.35887308612525154, -0.36347352132360194]
# personality: con
# MEAN SQUARED ERROR: -0.50 (+/- 0.04)
# [-0.5196974776277787, -0.49588103704861625, -0.5228167155446461, -0.4726107368105783, -0.47440152081593123]
# personality: ext

## random lasso method:
from sklearn.linear_model import RandomizedLasso
rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(data[:6650], real[:6650])
print (sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
                 data), reverse=True))

## get the top words:
# vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=20000)
# X = vec.fit_transform(docs2)
# selector = SelectKBest(f_regression,k=300)
# allvec_new = selector.fit_transform(X.toarray(),real)

# np.asarray(vec.get_feature_names())[selector.get_support("features")]

### selector k = 200 and pca = 80  - 50-100 pca
# pca = sklearn.decomposition.TruncatedSVD(n_components=100)

X = getTextVec(20000,person_text)
getRMSEPCA5Fold(regression,X,100)
# finished data preprocessing X = 10000
# personality: ope
# MEAN SQUARED ERROR: -0.38 (+/- 0.05)
# [-0.42565231612810567, -0.3782441352505706, -0.3645907047116833, -0.35817856358162375, -0.3604409306877769]
# personality: con
# MEAN SQUARED ERROR: -0.49 (+/- 0.04)
# [-0.5206757238851975, -0.48923018739496893, -0.5173914232407052, -0.46925430934978735, -0.47345970845690044]
# personality: ext
# MEAN SQUARED ERROR: -0.63 (+/- 0.03)
# [-0.6594921641883058, -0.6263693628189787, -0.6352974825778028, -0.6089326047366584, -0.6255636663293422]
# personality: agr
# MEAN SQUARED ERROR: -0.43 (+/- 0.02)
# [-0.44268679274974854, -0.42877793717761464, -0.4112208885904096, -0.4187537501907635, -0.4288471061342923]
# personality: neu
# MEAN SQUARED ERROR: -0.62 (+/- 0.05)
# [-0.6550603208225506, -0.6070741119710745, -0.603708660553279, -0.5883135141534478, -0.6259307536188674]
getRMSEPCA5Fold(clf,X,100)
getRMSEPersonal(clf,100,X,person_text)
# Number of Top Features: 100 ## selectKBest
# finished data preprocessing
# personality: ope
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.39 (+/- 0.04)
# personality: con
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.50 (+/- 0.04)
# personality: ext
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.64 (+/- 0.04)
# personality: agr
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.43 (+/- 0.02)
# personality: neu
# getting scores....from 5-fold-cross-validation
# MEAN SQUARED ERROR: -0.62 (+/- 0.04)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
regr_1 = DecisionTreeRegressor(max_depth=4)
rng = np.random.RandomState(1)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

getRMSEPCA5Fold(regr_1,X,100) ## high

