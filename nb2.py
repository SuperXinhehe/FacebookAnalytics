### fit model:
import numpy as np
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X = gendertext["text"]
X = X.tolist()

glabel = gendertext["gender"]

# len(allterms)
# 1042902
## build the term matrix on 


docs = []
for doc in X:
	docs.append(" ".join(doc))


### tokenize function
def tokenize(text):
    tokens = text.split(" ")
    tokens = [token.replace("\n","") for token in tokens]
    ## if the word has length greater than 
    words = []
    for w in tokens:
        if len(w) >= 12 and len(w) < 17:
         words.append("12_LONG_WORD")
        elif len(w) >= 17:
         words.append("17_LONG_WORD")
        else:
         words.append(w)
    tokens = []
    for w in words:
    	if w=="!" or w=="?" or len(w)>1:
    		tokens.append(w)
    return tokens

### 8 giga bytes data matrix
vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=80000)
X = vec.fit_transform(docs)

### check all the features 
# vec.get_feature_names()
# a = vec.build_analyzer()
# a(":) 12_MONTH !")

label = gendertext["gender"]
selector = SelectKBest(chi2,k=5000)
allvec_new = selector.fit_transform(X.toarray(),label)

### fit the model
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB(alpha=0.1)
# clf.fit(allvec_new[:6650], label[:6650])
# # 9500*0.70
# # 6650.0
# pred = clf.predict(allvec_new[6650:])
# np.mean(pred == label[6650:])  


# 0.60561403508771927 vec && 10000
## 0.76526315789473687 multinb && 5000 alpha = 0.1


vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=80000)

top_ranked_features = sorted(enumerate(selector.scores_),key=lambda x:x[1], reverse=True)[:1000]
top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]

features1 = []
for feature_pvalue in zip(np.asarray(vec.get_feature_names())[top_ranked_features_indices],selector.pvalues_[top_ranked_features_indices]):
        features1.append(feature_pvalue)


# with open('feature_terms_5000.txt','w') as f:
#    for row in features1:
#    	 f.write("%s,%s\n" % (str(row[0]),str(row[1])))

### two classes classification gaussian model with default setting
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(allvec_new[:6650], label[:6650])
pred = gnb.predict(allvec_new[6650:])

np.mean(pred == label[6650:])  
# 0.79263157894736846 with 5000 terms in dtm matrix 


#### change terms to be 8000
selector = SelectKBest(chi2,k=8000)
allvec_new = selector.fit_transform(X.toarray(),label)
# 0.82280701754385965
##### 10000 terms matrix ######
# 0.82596491228070179

##### prediction using same matrix to predict the age:
ages = agetb["age"].tolist()
def groupAge(age):
    age = int(age)
    if age >= 18 and age <= 24:
        return "18-24"
    elif age >= 25 and age <= 34:
        return "25-34"
    elif age >= 35 and age <= 49:
        return "35-49"
    else:
        return "50-xx"

## convert to agegroups
agegroups = [groupAge(age) for age in ages]
agetb["age"] = agegroups
agetb.columns = ["usrid","agegroup"]

## merge age and text tables
age_text = agetb.merge(texttb,on="usrid")
agetext = age_text[["agegroup","text"]]

##
X = agetext["text"]
X = X.tolist()


X = vec.fit_transform(docs)
label = agetext["agegroup"]
selector = SelectKBest(chi2,k=8000)
allvec_new = selector.fit_transform(X.toarray(),label)

### multiple labels learning:
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
np.mean(pred == label[6650:])  
## 0.6143859649122807
gnb = GaussianNB()
pred = OneVsOneClassifier(gnb).fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.60385964912280699
clf = MultinomialNB(alpha=0.1)
pred = clf.fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])

pred = LinearSVC(random_state=0).fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])

### lasso logistic regression: 
from sklearn import linear_model
clf = linear_model.Lasso(alpha = 0.1)
pred = clf.fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.54491228070175435
### change to 5000
selector = SelectKBest(chi2,k=5000)
allvec_new = selector.fit_transform(X.toarray(),label)
### 
gnb = GaussianNB()
pred = gnb.fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.60736842105263156
### linear svc
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.61122807017543856

#### change to 10000
selector = SelectKBest(chi2,k=10000)
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.6171929824561404
clf = MultinomialNB(alpha=0.1)
# 0.56000000000000005
pred = gnb.fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
np.mean(pred == label[6650:])  
# 0.61122807017543856


##### k = 5000 #######
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

pred = KNeighborsClassifier(n_neighbors=10).fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.5463157894736842
### multi-class classification: svm
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
pred = clf.fit(allvec_new[:6650], label[:6650]).predict(allvec_new[6650:])
# 0.54491228070175435



from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
>>> for param_name in sorted(parameters.keys()):
...     print("%s: %r" % (param_name, best_parameters[param_name]))