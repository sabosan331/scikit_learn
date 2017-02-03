#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search


# 学習データの読み込み f -> f1,f2
traindata = np.loadtxt("/home/seiji/jikken/3DHOG/bestfeature/train.csv", delimiter=",",comments='#')
feat_train, label_train = np.hsplit(traindata,[-1])
print("連結特徴量次元 : " + str(feat_train.shape[1]) )
#print(label_train.shape[1])
#label_train

label_train = label_train.flatten()

#label_train


# parameters = {
#         'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
#         'max_features'      : [3, 5, 10, 15, 20],
#         'random_state'      : [0],
#         'n_jobs'            : [1],
#         'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
#         'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
# }

# clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
# clf.fit(feat_train, label_train)
#print(clf.best_estimator_)

forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features=15, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
forest.fit(feat_train,label_train)



Y=[]
RES=[]
for i in range(25,35):
	testdata = np.loadtxt("/home/seiji/jikken/3DHOG/bestfeature/sequence." + str(i) + ".csv", delimiter=",",comments='#')
	feat_test, label_test = np.hsplit(testdata,[-1])
	label_test = label_test.flatten()
	Y.append(label_test)
	result = forest.predict(feat_test)
	RES.append(result)
	# cmat = confusion_matrix(label_test,result)
	# accuracy = accuracy_score(label_test,result)
	# print(accuracy)
	# print(cmat)
#Y=Y.flatten()
print "confusion matrix"
print confusion_matrix(Y, RES)
print ""
print "classification report"
print classification_report(Y, RES, target_names=map(str, range(10)))
print ""
print "accuracy"
print accuracy_score(Y, RES)