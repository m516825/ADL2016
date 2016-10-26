### ADL hw1

#### usage:

You should place the training corpus, 'text8' and 'ptt_corpus.txt' in ./corpus folder
before using the following instructions.

Splited version of these two corpus(smaller) have been placed in ./corpus for runing testing

* glove and word2vec
```
> chmod a+x run_phase1.sh
> ./run_phase1.sh $corpus $output_folder
```

* ptt
```
> chmod a+x run_phase2.sh
> ./run_phase2.sh $corpus $output_folder
```