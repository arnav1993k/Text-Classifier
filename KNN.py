#!/usr/bin/env python3
import numpy as np
import pandas as pd
from collections import Counter
class KNN:
	#function for calculating Euclidean distance
	def __euclidean_distance(self,A,B):
		return np.sqrt(np.sum(np.square(A-B)))
	#Function for prediction
	def get_distances(self,X,y,x_test):
		distance=[]
		classes=Counter()
		#calculate and store ditances of all the points from given test data
		for i in range(X.shape[0]):
			dist=self.__euclidean_distance(x_test,X[i,:])
			distance.append([dist,i])
		#sort the distances
		distance=sorted(distance)
		#print(distance)
		return distance
	def get_classes(self,distances,y,k):
		classes=Counter()
		for i in range(k):
			index=distances[i][1]
			classes[y[index][0]]+=1
		#print(classes)
		return classes
	def predict(self,X,y,x_test,k):
		distance=self.get_distances(X,y,x_test)
		classes=self.get_classes(distance,y,k)
		return classes.most_common(1)[0][0]
# def main():
# 	knn=KNN()
# 	X=[[1,1,0,0],[1,0,0,0],[0,1,0,1],[1,0,0,0],[0,0,1,0]]
# 	X=np.array(X)
# 	y=[[2],[1],[1],[2],[0]]
# 	y=np.array(y)
# 	x_test=np.array([1,0,0,1])
# 	print(knn.predict(X,y,x_test,1))
# if __name__ == '__main__':
# 	main()