#!/bin/bash
python2.7 data_formate.py --raw_test $1
python2.7 run_multi-task_rnn.py --data_dir ./ADL_data --train_dir=model_intent \
--max_sequence_length=50 --task=intent --bidirectional_rnn=True --predict=True --answer_file=$2