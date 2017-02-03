#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
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

# 学習データを主成分分析1
pca1 = PCA(n_components = float(sys.argv[1]) )
reduced1_train = pca1.fit_transform(feat1_train)
del feat1_train
print("1つ目の次元圧縮後の次元" + str(reduced1_train.shape) )
print(pca1.explained_variance_ratio_)
print(np.cumsum(pca1.explained_variance_ratio_))

# 学習データを主成分分析2
pca2 = PCA(n_components = float(sys.argv[1]) )
reduced2_train = pca2.fit_transform(feat2_train)
del feat2_train
print("2つ目の次元圧縮後の次元" + str(reduced2_train.shape) )
print(pca2.explained_variance_ratio_)
print(np.cumsum(pca2.explained_variance_ratio_))

# 圧縮後のreduced_f1,reduced_f2 -> reduced_f
reduced_train = np.c_[reduced1_train, reduced2_train]
label_train = label_train.flatten()
del reduced1_train,reduced2_train
dump_svmlight_file(  reduced_train , label_train , "reduced_train.txt",zero_based=False)
del reduced_train,label_train

for i in range(25,35):
 	print("sequence." + str(i) + ".csv")
 	testdata = np.loadtxt("sequence." + str(i) + ".csv", delimiter=",",comments='#')
 	feat_test, label_test = np.hsplit(testdata,[-1])
 	feat1_test, feat2_test = np.hsplit(feat_test,[feat_test.shape[1]/2])
 	print("sequence." + str(i) + " サンプル数 : " + str( feat2_test.shape[0]) )
 	reduced1_test = pca1.transform(feat1_test)
 	del feat1_test
 	print("テスト1つ目" + str(reduced1_test.shape))
 	reduced2_test = pca2.transform(feat2_test)
 	del feat2_test
 	print("テスト2つ目" + str(reduced2_test.shape))
 	reduced_test = np.c_[reduced1_test, reduced2_test]
 	label_test = label_test.flatten()
 	del reduced1_test,reduced2_test
 	# テストデータをSVM形式で保存
 	dump_svmlight_file(  reduced_test , label_test , "reduced_sequence." + str(i) + ".txt",zero_based=False)
 	del reduced_test,label_test
