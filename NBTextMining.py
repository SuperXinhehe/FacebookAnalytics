from os import listdir
from os.path import isfile,join

import pandas as pandas
import numpy as np 


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

profile = 

conf = SparkConf().setAppName("facebook").setMaster("yarn")
sc = SparkContext(conf=conf)

## make it as rdd
texttbrdd = sc.parallelize(texttb)

import csv
import StringIO


# profilerdd = sc.textFile(textpath.replace("Text","")+"Profile.csv")
# profiletb = csv.DictReader(open(textpath.replace("Text","Profile/")+"Profile.csv"))
mypath = textpath.replace("Text","Profile/")+"Profile.csv"
o = open(mypath,'rU')
profiletb = csv.DictReader(o)

proftb = {}
for row in profiletb:
    proftb[row.get("userid")] = [row.get("gender"),row.get("age")]

proftb = [(key,value) for key,value in proftb.items()]

genders = [(key,value[1]) for key,value in proftb]
ages = [(key,value[0]) for key,value in proftb]
texttbl = [(key,value) for key,value in texttb.items()]

gendersrdd = sc.parallelize(genders)
textrdd = sc.parallelize(texttbl)

textrdd.join(gendersrdd).collect()
text_gender_rdd = textrdd.join(gendersrdd)
textgender = text_gender_rdd.collect()
[(key,(val[0].split(),val[1])) for key,val in textgender]
### change to labeled point:

### 
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def getWords(pair):
    puncs = string.punctuation.replace("!","").replace("?","")
    STOPWORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    label = pair[1][1]
    m = pair[1][0]
    ## split into words
    words = re.split(' |,|;',m)
    ## lower case
    words = [w.lower() for w in words]
    ## remove punc except ? and !
    words = [w for w in words if not w in puncs]
    ## stemming
    words = [w.decode('utf8') for w in words]
    stemmed = [STEMMER.stem(w) for w in words]
    words = []
    for w in stemmed:
        w = w.encode('utf8')
    	regexp = re.compile("^[?]+$|^[!]+$")
    	if re.search(regexp,w) is not None:
    		for s in list(w):
    			words.append(s)
    	else:
    		words.append(w)
    ## remove stop words
    words = [w for w in words if not w in STOPWORDS]
    return (label,words)

text_gender_rdd2 = text_gender_rdd.map(getWords)

text_gender_rdd2.collect()
    # features = Vectors.dense(x for x in parts[1][0].split(' ')])
    # return LabeledPoint(label, features)


from nltk import word_tokenize
from nltk.corpus import stopwords
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors

def toDTM(data):
	htf = HashingTF(5000)
	uid = data[0]
	words = data[1][1]
	label = data[1][0]
	dtm = htf.transform(words)
	return (label,dtm)

def toTFIDF(data):
	dtm = data[1]
	dtm = sc.parallelize(dtm)
	idf = IDF().fit(dtm)
	tfidf = idf.transform(dtm)
	return LabeledPoint(data[0],tfidf)

usr_tg_dtm = text_gender_rdd2.map(toDTM)

def toLabelFormat(data):
	return LabeledPoint(data[0],data[1])

usrtgdtm = usr_tg_dtm.map(toLabelFormat)
training,test = usrtgdtm.randomSplit([0.7,0.3],seed=0)

model = NaiveBayes.train(training,1.0)
