#!/bin/bash
for (( i=15; i<=20; i=i+1 ))
do
    python3 -m basic.cli --run_id 0 --len_opt --cluster --batch_size 20 --model './out/basic/00/save/basic-'$i'000'
done

python2.7 pos_process_ensemble.py --data_dir './TOEFL_QA' --test $1 --ans $2
