import cv2
import os
import argparse
import numpy as np
from Detector import Detector
import json


def non_max(dabba, Thresh):

	if len(dabba) == 0:
		return []
 

	pick = []
 

	x1 = dabba[:,0]
	y1 = dabba[:,1]
	x2 = dabba[:,2]
	y2 = dabba[:,3]
 

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:

		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		for pos in range(0, last):

			j = idxs[pos]
 

			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 

			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 

			overlap = float(w * h) / area[j]
 

			if overlap > Thresh:
				suppress.append(pos)
 

		idxs = np.delete(idxs, suppress)
 
	return dabba[pick]


json_list = []
def getImages(folder):
    

    
    images = []
    
    for filename in os.listdir(folder):
        
        if filename.endswith('.ppm') or filename.endswith('.jpg'):
            
            images.append(filename)
            
    images.sort()
    
    return images

def Json_parse(filename,locations):
    
    for i in range(0,len(locations)):
        x,y,x2,y2 = locations[i]
        element = {"iname": filename, "bbox": [int(x), int(y), int(x2-x), int(y2-y)]} #first element in json file
        json_list.append(element)


def Jsonfinal():
    output_json = "results.json"
    with open(output_json, 'w') as f:
        json.dump(json_list, f)
    
def parse_args():

    
    parser = argparse.ArgumentParser(description="cse 473/573 project 3.")
    
    parser.add_argument('string', type=str, default="./data/",help="Resources folder,i.e, folder in which images are stored")
    
    args = parser.parse_args()
    
    return args   


if __name__ == '__main__':
    
    args = parse_args()
    
    Test_folder = args.string
    
    Test_img = getImages(Test_folder)
    
    for i in range(0,1):
            imgtest = cv2.imread(Test_folder+'/'+ Test_img[0])
            test_image = cv2.cvtColor(imgtest,cv2.COLOR_RGB2GRAY)
            windowsize_r = 100
            windowsize_c = 100
            listofsegments = []
            filename = "10"
            clf = detector.load(filename)
            # Crop out the window and calculate the histogram
            locations = []
            while windowsize_r<(len(test_image)-2):
                for r in range(0,test_image.shape[0] - windowsize_r, 10):
                    for c in range(0,test_image.shape[1] - windowsize_c, 10):
                        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                        #counter+=1
                        window=cv2.resize(window,dsize=(24,24))
                        prediction = clf.clfy(window)
                        if prediction ==1:
                            arr = [r,c,r+windowsize_r,c+windowsize_c]
                            locations.append(arr)
                windowsize_r+=50
                windowsize_c+=50
            
            locations = non_max(np.array(locations), 0.2)
            Json_parse(Test_img[i],locations)
            

        
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
    print(json_list)
    Jsonfinal()
    

    
    
    
    
        
    
    
    

    
