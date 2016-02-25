import os
from PIL import Image
from pylab import *
import csv
import numpy as np
from sklearn.metrics import accuracy_score
import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from NeuralNetwork import NeuralNetwork
true = 0
false = 0
image_arr = []
reader = csv.reader(open("/Users/yaqunyu/Desktop/test.csv",'rU'))
ip =0
useri_arr = [] 
gender_dic = {}
for number, userid , gender in reader:
    useri_arr.append(userid)
    gender_dic[userid]=gender
print len(useri_arr)
nn = NeuralNetwork([900,16000,1],'logistic')
for id in range(0,900):  
    uid = useri_arr[id]
    v = gender_dic[uid]
    print ip 
    pil_im = Image.open("/Users/yaqunyu/Desktop/image_python/training/"+uid+".jpg").convert('L')
    #pil_im = cv2.imread("/Users/yaqunyu/Desktop/image_python/training/"+uid+".jpg", cv2.IMREAD_GRAYSCALE)
    #pil_im = np.array(pil_im)
    #pil_im = pil_im.resize((100,100))
    pil_im = pil_im.resize((30,30))
    X = np.array(pil_im)
    #print "X:",len(X)
    arr = X.tolist()
    va = []
    for i in range(0,len(arr)):
        va.extend(arr[i])
    X = np.array(va)
    X = X.astype(np.float64)
    X -= X.min()
    X /= X.max()
    X = np.around(X, decimals=4)
    g = int(float(v))
    y=np.array([g])
    print y
    nn.fit(X, y, epochs=1)
    ip=ip+1
    #accuracy_score(y_test, nn.predict(X_test))
for id in range(901,1001):  
    uid = useri_arr[id]
    v = gender_dic[uid]
    print ip
    pil_im = Image.open("/Users/yaqunyu/Desktop/image_python/training/"+uid+".jpg").convert('L')
    #pil_im = cv2.imread("/Users/yaqunyu/Desktop/image_python/training/"+uid+".jpg", cv2.IMREAD_GRAYSCALE)
    #pil_im = np.array(pil_im)
    #pil_im = pil_im.resize((30,30))
    pil_im = pil_im.resize((30,30))
    X = array(pil_im)
    print "X:",len(X)
    arr = X.tolist()
    va = []
    for i in range(0,len(arr)):
        va.extend(arr[i])
    X = np.array(va)
    X = X.astype(np.float64)
    X -= X.min()
    X /= X.max()
    X = np.around(X, decimals=4) 
    g = int(float(v))
    y=np.array([g])
    print len(X)
    r = nn.predict(X)
    if ((r >=0.5)and(g==1))or((r<0.5)&(g==0)):
        true = true +1
    else :
        false = false +1   
    print r
    ip=ip+1
print "--------finish testing--------"
print "true : " ,true
print "false : " , false
total = true + false
print total
accuracy = float(true) / float(total)
print "accuracy: ", accuracy
