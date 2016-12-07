#!/bin/bash
cat ./generation/model/model.* > ./generation/model/translate.ckpt-1700.data-00000-of-00001
python2.7 ./generation/translate.py --decode_test 1 --data_dir ./generation/data --train_dir ./generation/model \
--num_layers=2 --size=256 --test_data $1 --output_file "./generation/predict.out"
python2.7 ./generation/pos_process.py $1 "./generation/predict.out" $2