import sys
import csv
import cv2
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import os
from PIL import Image
faceCascade = cv2.CascadeClassifier('/Users/yaqunyu/anaconda/lib/python2.7/site-packages/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
# image process
#trainingdata_path_root = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/'
#trainingimage_path_root = trainingdata_path_root + 'Train/Image/'
#traininglabels_path = trainingdata_path_root + 'Train/Profile/Profile.csv'
#read in profile.csv
trainingdata_path_root = '/Users/yaqunyu/Desktop/'
trainingimage_path_root =trainingdata_path_root +'image_python/train/'
traininglabels_path = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv'
gender_dic = {}
useri_arr = [] 
true = 0
false = 0
reader = csv.reader(open("/Users/yaqunyu/Desktop/test.csv",'rU'))
for number, userid , gender in reader:
    useri_arr.append(userid)
    gender_dic[userid]=gender


print len(useri_arr)
print gender_dic
neigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
ip=0
output = []
value1 = []
for key in range(0,901):
    userid = useri_arr[key]
    value = gender_dic[userid]
    print ip
    g = int(float(value)) 
    value1.append(g)
    print value1
    image = Image.open("/Users/yaqunyu/Desktop/image_python/training/"+userid+".jpg").convert('L')
    image = np.array(image)
    #print el
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image, 1.1, 5) #when there is only one image, 5 will change to 2 or 0
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        resize_face = cv2.resize(crop_image,(30,30))
        #X= np.array(resize_face)
        arr=resize_face.tolist()
        #print arr
        va = []
        for i in range(0,len(arr)):
            va.extend(arr[i])
        #print va 
    output.append(va)
    #print output              
    ip=ip+1               

neigh.fit(output,value1)
ip = 901
for key in range(901,1001):
    userid = useri_arr[key]
    value = gender_dic[userid]
    print ip
    g = int(float(value)) 
    value1.append(g)
    print value1
    image = Image.open("/Users/yaqunyu/Desktop/image_python/training/"+userid+".jpg").convert('L')
    image = np.array(image)
    #print el
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image, 1.1, 5) #when there is only one image, 5 will change to 2 or 0
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        resize_face = cv2.resize(crop_image,(30,30))
        #X= np.array(resize_face)
        arr=resize_face.tolist()
        #print arr
        va = []
        for i in range(0,len(arr)):
            va.extend(arr[i])   
        #print va 
    output.append(va)
    #print output              
    ip=ip+1
    r=neigh.predict(output)
    print r
    print len(r)
    print len(value1)
    for j in range(0,len(r)):
        if (r[j] == value1[j]):
            true = true +1
        else:
            false = false +1 

print "--------finish testing--------"
print "true : " ,true
print "false : " , false
total = true + false
print total
accuracy = float(true) / float(total)
print "accuracy: ", accuracy
