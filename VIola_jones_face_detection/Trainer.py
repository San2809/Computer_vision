import pickle
from Detector import Detector
import time
import cv2
import os
import numpy as np


def trainv(tv):
    print("started....")
    clf = Detector(T=tv)
    with open('Training.pkl', 'rb') as f:
        train = pickle.load(f)
    clf.training(train, 700, 700)
    eval(clf, train)
    clf.save(str(tv))
    print("Done Amaze bro")




    
    
if __name__ == '__main__':
    

    Data = []
    for filename in os.listdir("./FaceImages/"):
        if filename.endswith(".jpg"):
            img=cv2.imread('./FaceImages/'+filename)
            img = cv2.resize(img,dsize=(24,24))
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            data = (img,1)
            Data.append(data)
    for filename in os.listdir("./NegativeImages/"):
        if filename.endswith(".jpg"):
            img=cv2.imread('./NegativeImages/'+filename)
            img = cv2.resize(img,dsize=(24,24))
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            data = (img,0)
            
            Data.append(data)
        
    Data = np.array(Data)
    Train = open('./Train-Data.pkl','wb')
    pickle.dump(Data,Train)
    Train.close()



    trainv(50)


