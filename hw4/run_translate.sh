#!/bin/bash
cat ./translate/model/model.* > ./translate/model/translate.ckpt-11600.data-00000-of-00001
python2.7 ./translate/translate.py --decode_test 1 --data_dir ./translate/ADL_data --train_dir ./translate/model \
--num_layers=2 --size=256 --test_data $1 --predict_file $2
