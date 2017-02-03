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
import sklearn.linear_model as lm

# 学習データの読み込み f -> f1,f2
traindata = np.loadtxt("train.csv", delimiter=",",comments='#')
feat_train, label_train = np.hsplit(traindata,[-1])
feat1_train, feat2_train = np.hsplit(feat_train,[feat_train.shape[1]/2])
print("連結特徴量次元 : " + str(feat_train.shape[1]) )
del feat_train
print("1(サンプル数,次元) : " + "(" + str(feat1_train.shape[0]) + "," + str(feat1_train.shape[1]) + ")")
print("2(サンプル数,次元) : " + "(" + str(feat2_train.shape[0]) + "," + str(feat2_train.shape[1]) + ")" )

# ラベル置き換え
for i in range(0,label_train.shape[0]):
	if label_train[i] == -1000:
		label_train[i] = 9

# lasso正則化1
#feat1_train = (feat1_train - feat1_train.mean())/feat1_train.std() # 基準化
# model1 = lm.Lasso(alpha=0.01, max_iter=1000,tol=0.0) # tol=0.0で収束判定なし(上の実装とほぼ同条件?)
model1 = lm.Lasso(alpha=float(sys.argv[1]))
model1.fit(feat1_train, label_train)

reduced1_train = feat1_train[:,np.where(model1.coef_ != 0)[0]]

# reduced1_train = feat1_train * model1.coef_
# reduced1_train = reduced1_train[:,np.where(model1.coef_ != 0)[0]]
del feat1_train


# lasso正則化2
#feat2_train = (feat2_train - feat2_train.mean())/feat2_train.std() # 基準化
# model2 = lm.Lasso(alpha=0.01, max_iter=1000,tol=0.0) # tol=0.0で収束判定なし(上の実装とほぼ同条件?)
model2 = lm.Lasso(alpha=float(sys.argv[1]))
model2.fit(feat2_train, label_train)

reduced2_train = feat2_train[:,np.where(model2.coef_ != 0)[0]]
#reduced2_train = feat2_train * model2.coef_
#reduced2_train = reduced2_train[:,np.where(model2.coef_ != 0)[0]]

del feat2_train

print("red_train1つ目" + str(reduced1_train.shape))
print("red_train2つ目" + str(reduced2_train.shape))

# ラベル置き換え
for i in range(0,label_train.shape[0]):
	if label_train[i] == 9:
		label_train[i] = -1000

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

  	# lasso正則化1
  	# feat1_test = (feat1_test - feat1_test.mean())/feat1_test.std() # 基準化
  	reduced1_test = feat1_test * model1.coef_
  	reduced1_test = feat1_test[:,np.where(model1.coef_ != 0)[0]]

	# lasso正則化2	
  	# feat2_test = (feat2_test - feat2_test.mean())/feat2_test.std() # 基準化
  	reduced2_test = feat2_test * model2.coef_
  	reduced2_test = feat2_test[:,np.where(model2.coef_ != 0)[0]]
  	del feat1_test,feat2_test

  	print("テスト1つ目" + str(reduced1_test.shape))
  	print("テスト2つ目" + str(reduced2_test.shape))

  	reduced_test = np.c_[reduced1_test, reduced2_test]
  	label_test = label_test.flatten()
  	del reduced1_test,reduced2_test
  	# テストデータをSVM形式で保存
  	dump_svmlight_file(  reduced_test , label_test , "reduced_sequence." + str(i) + ".txt",zero_based=False)
  	del reduced_test,label_test
