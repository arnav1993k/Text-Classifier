#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
from collections import Counter
class Gaussian_NBC:
	def Gaussian_Prob(self,X,mean,var):
		exp=np.exp(-np.power((X-mean),2)/(2*var))
		prob=exp/np.sqrt(2*var*math.pi)
		#print(exp,prob,mean,var)
		return prob
	def log_gaussian(self,X,mean,var):
		return np.log(self.Gaussian_Prob(X,mean,var))
	def get_mean_prior_and_var(self,X,y):
		y=y.ravel()
		self.classes=np.unique(y)
		k=X.shape[1]
		self.mean=np.zeros((len(self.classes),k))
		sq_sum=np.zeros((len(self.classes),k))
		self.prior=np.zeros(10)
		for c in y:
			self.prior[int(c)]+=1
		for i in range(X.shape[0]):
			for j in range(k):
				self.mean[int(y[i])][j]+=(X[i][j]/self.prior[int(y[i])])
				sq_sum[int(y[i])][j]+=(np.power(X[i][j],2))/self.prior[int(y[i])]
		self.var=sq_sum-np.power(self.mean,2)
	
	def fit(self,X,y):
		self.get_mean_prior_and_var(X,y)
	def predict(self,X):
		max_likelihood=-111111111111
		prediction=0
		for i in range(len(self.classes)):
			log_likelihood=0
			for j in range(len(X)):
				log_likelihood+=self.log_gaussian(X[j],self.mean[i][j],self.var[i][j])
			if max_likelihood<log_likelihood:
				max_likelihood=log_likelihood
				prediction=i
		return prediction