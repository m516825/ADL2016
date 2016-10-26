#!/bin/bash
if [[ ! -e $2 ]]; then
	mkdir $2
fi
python ./src/glove/vocab.py --corpus $1
echo "start training glove"
python ./src/glove/cooccur.py --corpus $1
python ./src/glove/glove.py --vector "$2glove.txt"
python ./src/filterVocab/filterVocab.py ./src/filterVocab/fullVocab.txt < $2glove.txt > $2filter_glove.txt
if [[ -f $2glove.txt ]]; then 
	rm $2glove.txt
fi

echo "start training word2vec"
python ./src/word2vec/contextword.py --corpus $1
python ./src/word2vec/word2vec.py --vector "$2word2vec.txt"
python ./src/filterVocab/filterVocab.py ./src/filterVocab/fullVocab.txt < $2word2vec.txt > $2filter_word2vec.txt
if [[ -f $2word2vec.txt ]]; then
	rm $2word2vec.txt
fi