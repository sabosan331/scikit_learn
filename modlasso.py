#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import dump_svmlight_file


# 学習データの読み込み f -> f1,f2
traindata = np.loadtxt("train.csv", delimiter=",",comments='#')
feat_train, label_train = np.hsplit(traindata,[-1])
feat1_train, feat2_train = np.hsplit(feat_train,[feat_train.shape[1]/2])
print("連結特徴量次元 : " + str(feat_train.shape[1]) )
del feat_train
print("1(サンプル数,次元) : " + "(" + str(feat1_train.shape[0]) + "," + str(feat1_train.shape[1]) + ")")
print("2(サンプル数,次元) : " + "(" + str(feat2_train.shape[0]) + "," + str(feat2_train.shape[1]) + ")" )
#print(label_train.shape[1])

label_train = label_train.flatten()

# 学習データを主成分分析1
#float(sys.argv[1])
lasso1 = LinearSVC(C=float(sys.argv[2]), penalty="l1", dual=False).fit(feat1_train, label_train)
#lasso1 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(feat1_train, label_train)
model1 = SelectFromModel(lasso1, prefit=True,threshold=1e-5*float(sys.argv[1]))
#print(model1.threshold_)
reduced1_train = model1.transform(feat1_train)
del feat1_train
print("1つ目の次元圧縮後の次元" + str(reduced1_train.shape) )

# 学習データを主成分分析2
lasso2 = LinearSVC(C=float(sys.argv[2]), penalty="l1", dual=False).fit(feat2_train, label_train)
#lasso2 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(feat2_train, label_train)
model2 = SelectFromModel(lasso2, prefit=True,threshold=1e-5*float(sys.argv[1]))
#print(model2.threshold_)
reduced2_train = model2.transform(feat2_train)
del feat2_train
print("2つ目の次元圧縮後の次元" + str(reduced2_train.shape) )


# 圧縮後のreduced_f1,reduced_f2 -> reduced_f
reduced_train = np.c_[reduced1_train, reduced2_train]

del reduced1_train,reduced2_train
dump_svmlight_file(  reduced_train , label_train , "reduced_train.txt",zero_based=False)
del reduced_train,label_train

for i in range(25,35):
 	print("sequence." + str(i) + ".csv")
 	testdata = np.loadtxt("sequence." + str(i) + ".csv", delimiter=",",comments='#')
 	feat_test, label_test = np.hsplit(testdata,[-1])
 	feat1_test, feat2_test = np.hsplit(feat_test,[feat_test.shape[1]/2])
 	print("sequence." + str(i) + " サンプル数 : " + str( feat2_test.shape[0]) )
 	reduced1_test = model1.transform(feat1_test)
 	del feat1_test
 	print("テスト1つ目" + str(reduced1_test.shape))
 	reduced2_test = model2.transform(feat2_test)
 	del feat2_test
 	print("テスト2つ目" + str(reduced2_test.shape))
 	reduced_test = np.c_[reduced1_test, reduced2_test]
 	label_test = label_test.flatten()
 	del reduced1_test,reduced2_test
 	# テストデータをSVM形式で保存
 	dump_svmlight_file(  reduced_test , label_test , "reduced_sequence." + str(i) + ".txt",zero_based=False)
 	del reduced_test,label_test
