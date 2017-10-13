#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
class Utilities:
	def get_files(self,path):
		all_files=os.listdir(path)
		return all_files
	def converttoArray(self,filename,location):
		fullpath=location+"/"+filename
		file=open(fullpath,"r")
		lines=file.readlines()
		X=[0]*1024
		y=int(filename[0])
		i=0
		for line in lines:
			for element in line:
				if element=='1':
					X[i]=1
					i+=1
				if element=='0':
					i+=1
		return X,y
	def converttoVector(self,location):
		files=self.get_files(location)
		print(str(len(files)) + " have been read from " + location)
		features=[]
		outputs=[]
		for file in files:
			X,y=self.converttoArray(file,location)
			features.append(X)
			outputs+=[[y]]
		nfeatures=np.array(features)
		noutputs=np.array(outputs)
		return nfeatures,noutputs
	def getImages(self,vector,l,b,y):
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