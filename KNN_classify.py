#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
from KNN import KNN
import pandas as pd
desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
training_path=desktop_path+"/trainingDigits"
testing_path=desktop_path+"/testingDigits"
def get_files(path):
	all_files=os.listdir(path)
	return all_files
def converttoArray(filename,location):
	fullpath=location+"/"+filename
	file=open(fullpath,"r")
	lines=file.readlines()
	X=[0]*1024
	y=filename[0]
	i=0
	for line in lines:
		for element in line:
			if element=='1':
				X[i]=1
				i+=1
			if element=='0':
				i+=1
	return X,y
def converttoVector(location):
	files=get_files(training_path)
	print(str(len(files)) + " have been read from " + location)
	features=[]
	outputs=[]
	for file in files:
		X,y=converttoArray(file,training_path)
		features.append(X)
		outputs+=[[y]]
	nfeatures=np.array(features)
	noutputs=np.array(outputs)
	#print(nfeatures.shape)
	#print(noutputs.shape)
	return nfeatures,noutputs
def getImages(vector,l,b,y):
	fig,axs=plt.subplots(5, 2,figsize=(4,8))
	unique_outputs=np.unique(y,return_index=True)
	print(unique_outputs)
	i=0
	j=0
	for index in unique_outputs[1]:
		image_array=vector[index].reshape(l,b)
		axs[i,j].imshow(image_array, interpolation='nearest')
		axs[i,j].set_title(y[index])
		i+=1
		j+=1
		if i==5: 
			i=0
		if j==2: 
			j=0
	plt.subplot_tool()
	plt.subplots_adjust(hspace=0.66)
	plt.show()
features,outputs=converttoVector(training_path)
#getImages(features,32,32,outputs)
def findError(X,y,x_test,y_test,label,knn):
	c_out=[]
	for test in x_test:
		c_out+=[[knn.predict(X,y,x_test,1)]]
	df=pd.DataFrame()
	df['Actual']=y_test.astype(int).ravel()
	df['Predicted']=np.array(c_out,dtype=np.int16).ravel()
	df['Error']=df['Actual']-df['Predicted']
	total_error=(df['Error']!=0).sum()
	df.to_csv(desktop_path+"/"+label+".csv")
	#percentage_error=total_error/(df.count+1)
	print("The "+label+" error is "+str(total_error*100/y.shape[0])+" percent.")

knn=KNN()
test_feature,test_output=converttoVector(testing_path)
findError(features,outputs,features,outputs,"Training",knn)
findError(features,outputs,test_feature,test_output,"Testing",nb)
