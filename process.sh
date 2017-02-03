#!/bin/sh

path='/home/seiji/jikken/3DHOG/lasso/'
path_libsvm='/home/seiji/dev/libsvm-3.21/'
path_liblin='/home/seiji/dev/liblinear-1.94/'

j=30

echo "(0,1) scaling"
for j in 9
do
for k in 3
do
    # echo "skip_rate '${j}' "
    echo 'sequence '${i}' is proceeding'
    ${path}/3DHOG/build/3dhog 30 ${j} ${k}

    # concat training samples
    cat sequence.[0-9].csv sequence.1[0-9].csv sequence.2[0-4].csv > train.csv
    rm -f sequence.[0-9].csv sequence.1[0-9].csv sequence.2[0-4].csv
    ${path}/others/csv2svm.py train.csv > train_svm.txt

    # scaling
    # echo "scaling"
    ${path_libsvm}/svm-scale -l 0 -u 1 -s data.minmax train_svm.txt > train.scale


    # training with liblinear
    w=`${path}/others/svm_weight.py train.scale` 
    ${path_liblin}/train -s 6 ${w} train.scale train_svm.txt.model
    rm -f train_svm.txt 

    # detection
    rm -f out.txt truth.csv
    for i in `seq 25 34`
    do
	echo 'detecting sequence '${i}
	${path}/others/csv2svm.py sequence.${i}.csv > ${i}_svm.txt
	${path_libsvm}/svm-scale -l 0 -u 1 -r data.minmax ${i}_svm.txt > ${i}.scale
	${path_liblin}/predict ${i}.scale train_svm.txt.model detected.txt #識別
	${path}/others/voting.py detected.txt 30 >> out.txt #投票処理
	${path}/others/groundtruth.py sequence.${i}.csv >> truth.csv #正解データ作成
	rm -f ${i}_svm.txt detected.txt
    done
    cat sequence.2[5-9].csv sequence.3[0-4].csv > test.csv
    # rm -f sequence.2[5-9].csv sequence.3[0-4].csv

    # evaluate
    echo 'Result'
    ${path}/others/KSCGR_evaluate.py out.txt truth.csv > res${j}.txt
done
done