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

## get the current path
## os.getcwd()
textpath = "/Users/XinheLovesMom/Downloads/TCSS555/Train/Text"
textfiles = [f for f in listdir(textpath) if isfile(join(textpath, f))]
uids = [n.replace(".txt","") for n in textfiles]


def extText(textfiles):
	texttb = {}
	for tf in textfiles:
		uid = tf.replace(".txt","")
		f = open(textpath+"/"+tf,'r',encoding='latin-1')
		txt = f.read()
		texttb[uid] = txt
		f.close()
	return texttb

texttb = extText(textfiles)

mypath = textpath.replace("Text","Profile/")+"Profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

proftb = [(key,value) for key,value in proftb.items()]

genders = [(key,value[0]) for key,value in proftb]
ages = [(key,value[1]) for key,value in proftb]
texttbl = [(key,value) for key,value in texttb.items()]

gendertb = pd.DataFrame(genders)
agetb = pd.DataFrame(ages)
gendertb.columns = ["usrid","gender"]
agetb.columns = ["usrid","age"]

def format1(label,m):
    puncs = string.punctuation.replace("!","").replace("?","")
    STOPWORDS = set(stopwords.words('english'))
    ## split into words
    words = re.split(' |,|;|//n|//',m)
    ## remove slashes
    words = [w.strip('\\') for w in words]
    words = [w.strip('//') for w in words]
    words = [w.replace("/","") for w in words]
    ## lower case
    words = [w.lower() for w in words]
    ## remove punc except ? and !
    words = [w for w in words if not w in puncs]
    ## remove stop words
    words = [w for w in words if not w in STOPWORDS]
    return (label,words)

texttbl2 = [format1(key,value) for key,value in texttbl]

def format2(label,words):
    stemmed = [w.rstrip('.') for w in words]
    words = []
    for w in stemmed:
        ## check if the word is only contains ! or ?
        regexp = re.compile("^[?]+$|^[!]+$")
        if re.search(regexp,w) is not None:
            for s in list(w):
               words.append(s)
        # if the word ends with many question mark or !
        elif w.endswith(('!','?')):
            l = re.findall('[?!.]',w)
            words + l
        ## if the word only contains the \w leave it
        elif re.match(r'^[_\W]+$',w):
            words.append(w)
        else:
            w = re.sub(r'[?!.]','',w)
            words.append(w)
    return (label,words)

texttbl3 = [format2(key,value) for key,value in texttbl2]
    
def format3(label,words):
    ## split by the \\\n
    STEMMER = PorterStemmer()
    wl = []
    for w in words:
      wl = wl + w.split('\\\n')
    words = []
    for w in wl:
      words = words + w.split('\\')
    wl = []
    for w in words:
      wl = wl + w.split('/')
    words = wl
    words = [w.strip('\\') for w in words]
    words = [w.strip('//') for w in words]
    words = [w.replace("/","") for w in words]
    ## remove all the numbers:
    wl = []
    for w in words:
      if w.isdigit():
      	wl.append("_NUMBERS_")
      else:
        wl.append(''.join(i for i in w if not i.isdigit()))
    ## removing stopping words
    STOPWORDS = set(stopwords.words('english'))
    words = [w for w in wl if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in words]
    ## if the word has length greater than 
    words = []
    for w in stemmed:
        if len(w) >= 12 and len(w) < 17:
         words.append("12_LONG_WORD")
        elif len(w) >= 17:
         words.append("17_LONG_WORD")
        else:
         words.append(w)
    return (label,words)

texttbl4 = [format3(key,value) for key,value in texttbl3]

# with open('texttbl4.txt','w') as f:
#    for row in texttbl4:
#    	 f.write("%s,%s\n" % (str(row[0]),str(row[1])))

# Training set - 70%
# 9500*0.70
# 6650.0
from random import shuffle
shuffle(texttbl4)
texttb = pd.DataFrame(texttbl4)
texttb.columns = ["usrid","text"]

# convert to panda dataframe:
gender_text = gendertb.merge(texttb,on="usrid")
## table gender with raw text
gendertext = gender_text[["gender","text"]]

# ## merge age and text tables
# age_text = agetb.merge(texttb,on="usrid")
# agetext = age_text[["age","text"]]

################## gender classification #####################
X = gendertext["text"]
X = X.tolist()

label = tran_gtexttb["gender"]

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

## vectorize
vec = TfidfVectorizer(encoding="latin-1",tokenizer = tokenize,token_pattern=r'(?u)\b\w\w+\b|^[_\W]+$',lowercase=False,max_features=80000)


from numpy import array
# arr1 = array(train1_glabel)
# arr0 = array(train0_glabel)
X = vec.fit_transform(docs)
## select 10000 significant terms
selector = SelectKBest(chi2,k=50000)
allvec_new = selector.fit_transform(X.toarray(),arr)
### fit the model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(allvec_new[:5000], arr[:5000])

test = test_gtexttb["text"].tolist()
test_docs = []
for doc in test:
	test_docs.append(" ".join(doc))




###### testing: 








