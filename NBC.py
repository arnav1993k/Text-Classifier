#!/usr/bin/env python3
import numpy as np
from collections import Counter,defaultdict
import pandas as pd
class NBClassifier:
	def __init__(self,alpha):
		self.alpha=alpha
	#fit function for all types of data
	def fit(self,X,y):
		classes=np.unique(y)
		#class_count=np.zeros(len(classes),X.shape[1])
		self.prior=np.zeros(len(classes))
		self.likelihood=np.ones([len(classes),X.shape[1]])*self.alpha
		for c in y:
			self.prior[int(c[0])]+=1
		for i in range(X.shape[1]):
			for j in range(X.shape[0]):
				self.likelihood[int(y[j][0])][i]+=X[j][i]
				#class_count[int(y[j][0])][i]
		print(self.likelihood)
		for i in range(X.shape[1]):
			for j in range(len(classes)):
				#self.likelihood[int(classes[j])][i]/=(self.likelihood[int(classes[j])][i]-self.alpha+len(classes)*self.alpha)
				self.likelihood[int(classes[j])][i]/=(self.prior[int(classes[j])]+len(classes)*self.alpha)
		df=pd.DataFrame(self.likelihood)
		print(self.prior)
		df.to_csv("/home/arnav1993k/Desktop/likelihood.csv")
		return self.prior,self.likelihood
	def getPosterior(self,X,prior,likelihood):
		posterior=0
		#print(likelihood,prior)
		for i in range(X.shape[0]):
				if X[i]==1:
					posterior+=likelihood[1][i]
				else:
					posterior+=likelihood[0][i]
		posterior+=prior
		# print("posterior=")
		# print(posterior)
		return posterior

	def predict(self,X):
		classes=np.linspace(0,len(self.prior)-1,len(self.prior),dtype=np.int16)
		log_likelihood_1=np.log(self.likelihood)
		log_likelihood_0=np.log(1-self.likelihood)
		log_prior=np.log(self.prior)
		max_posterior=-100000000
		prediction=0
		for c in classes:
			prior=log_prior[c]
			likelihood=[log_likelihood_0[c],log_likelihood_1[c]]
			posterior=self.getPosterior(X,prior,likelihood)
			if max_posterior<posterior:
				prediction=c
				max_posterior=posterior
		return prediction


# def main():
# 	X=[[1,1,0,0],[1,0,0,0],[0,1,0,1],[1,0,1,0],[0,0,1,0]]
# 	X=np.array(X)
# 	y=[[2],[1],[1],[2],[0]]
# 	y=np.array(y)
# 	nbc=NBClassifier(0.011)
# 	prior,likelihood=nbc.fit(X,y)
# 	print(likelihood)
# 	c=nbc.predict(np.array([1,0,0,1]))
# 	print(c)

# #main
# if __name__ == '__main__':
# 	main()