#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt
from KNN import KNN
import sklearn.metrics as metrics
from Utilities import Utilities
from sklearn.decomposition import PCA

def findError(X,y,x_test,y_test,label,knn,k):
	c_out=[]
	# classifier=neighbors.KNeighborsClassifier(n_neighbors=k)
	# classifier.fit(X,y.ravel())
	# score=classifier.score(x_test,y_test.ravel())
	for test in x_test:
		c_out+=[knn.predict(X,y,test,k)]
	# df=pd.DataFrame()
	# df['Actual']=y_test.astype(int).ravel()
	# df['Predicted']=np.array(c_out,dtype=np.int16).ravel()
	# df['Error']=df['Actual']-df['Predicted']
	# total_error=(df['Error']!=0).sum()
	#df.to_csv(desktop_path+"/"+label+"_knn.csv")
	y_test=y_test.astype(int).ravel()
	c=np.array(c_out,dtype=np.int16).ravel()
	total_accuracy=metrics.accuracy_score(y_test,c)

	#print(total_accuracy,score)
	#percentage_error=total_error/y_test.shape[0]
	#print("The "+label+" error is "+str()+" percent.")
	return total_accuracy*100
def main():
	desktop_path=os.path.join(os.path.expanduser('~'), 'Desktop')
	training_path=desktop_path+"/trainingDigits"
	testing_path=desktop_path+"/testDigits"
	util=Utilities()
	features,outputs=util.converttoVector(training_path)
	test_feature,test_output=util.converttoVector(testing_path)
	# knn=KNN()
	# errors_train=[]
	# errors_test=[]

	for i in range(1,5):
		# errors_train+=[findError(features,outputs,features,outputs,"Training",knn,i)]
		# errors_test+=[findError(features,outputs,test_feature,test_output,"Testing",knn,i)]
		print("Testing for k="+str(i))
		accuracy_score=[]
		for k in range(6,100):
			print("Testing for n="+str(k))
			knn=KNN()
			pca=PCA(n_components=k)
			X=pca.fit_transform(features)
			X_test=np.matmul(test_feature,np.transpose(pca.components_))
			accuracy_score+=[findError(features,outputs,test_feature,test_output,"Testing",knn,i)]
		print(accuracy_score)
		plt.plot(accuracy_score)
	plt.show()
#main
if __name__ == '__main__':
	main()