#!/bin/bash
python2.7 ./rvnn/rnn_restart.py --type 'test' --mpath './rvnn/rnn_embed=50_l2=0.000000_lr=0.010000.weights.temp' --testing_data $2 --answer $3