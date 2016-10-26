#!/bin/bash
if [[ ! -e $2 ]]; then
	mkdir $2
fi
echo "start training ptt vector"
python ./src/ptt_glove/vocab_ch.py --corpus $1
python ./src/ptt_glove/corpus_ch.py --corpus $1 --corpus_out ./ptt_corpus_title.txt
python ./src/ptt_glove/cooccur_ch.py 
python ./src/ptt_glove/glove_ch.py --vector $2ptt_vector.txt
python ./src/filterVocab/filterVocab.py ./src/filterVocab/fullVocab_phase2.txt < $2ptt_vector.txt > $2filter_vec.txt
if [[ -f ./ptt_corpus_title.txt ]]; then
	rm ./ptt_corpus_title.txt
fi
if [[ -f $2ptt_vector.txt ]]; then
	rm $2ptt_vector.txt
fi

