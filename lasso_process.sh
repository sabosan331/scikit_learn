#!/bin/sh

path='/home/seiji/jikken/3DHOG/pca/'
path_libsvm='/home/seiji/dev/libsvm-3.21/'
path_liblin='/home/seiji/dev/liblinear-1.94/'

for kiyo in 0.95 0.9 0.85 0.80 0.75
do
# echo "(0,1) scaling"
# for j in 9
# do
# for k in 3
# do
#     # echo "skip_rate '${j}' "
#     echo 'sequence '${i}' is proceeding'
#     ${path}/3DHOG/build/3dhog 30 ${j} ${k}

    # concat training samples
    # cat sequence.[0-9].csv sequence.1[0-9].csv sequence.2[0-4].csv > train.csv2svm
    # rm -f sequence.[0-9].csv sequence.1[0-9].csv sequence.2[0-4].csv
    # ${path}/others/csv2svm.py train.csv > train_svm.txt
    echo 'contribution:'${kiyo}
    ${path}/others/pca.py ${kiyo} 

    # scaling
    # echo "scaling"

    ${path_libsvm}/svm-scale -l 0 -u 1 -s data.minmax reduced_train.txt > train.scale

    # training with liblinear
    w=`${path}/others/svm_weight.py train.scale` 
    ${path_liblin}/train -s 6 ${w} train.scale train_svm.txt.model
    #w=`${path}/others/svm_weight.py reduced_train.txt` 
    #${path_liblin}/train -s 6 ${w} reduced_train.txt train_svm.txt.model

    # detection
    rm -f out.txt truth.csv
    for i in `seq 25 34`
    do
	echo 'detecting sequence '${i}

	${path_libsvm}/svm-scale -l 0 -u 1 -r data.minmax reduced_sequence.${i}.txt > ${i}.scale
	${path_liblin}/predict ${i}.scale train_svm.txt.model detected.txt #識別
    #${path_liblin}/predict reduced_sequence.${i}.txt train_svm.txt.model detected.txt #識別


	${path}/others/voting.py detected.txt 30 >> out.txt #投票処理
	${path}/others/groundtruth.py sequence.${i}.csv >> truth.csv #正解データ作成
	rm -f ${i}_svm.txt detected.txt
    done
    cat sequence.2[5-9].csv sequence.3[0-4].csv > test.csv
    # rm -f sequence.2[5-9].csv sequence.3[0-4].csv

    # evaluate
    echo 'Result'
    ${path}/others/KSCGR_evaluate.py out.txt truth.csv > res_cntb${kiyo).txt
done
done