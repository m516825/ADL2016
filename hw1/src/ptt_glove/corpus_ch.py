# encoding=utf-8
import argparse
import sys
import os
import random
import numpy as np

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='./corpus/ptt_corpus.txt', type=str)
	parser.add_argument('--corpus_out', default='./corpus/ptt_corpus_title.txt', type=str)
	parser.add_argument('--vocab', default='./vocab_ch.out', type=str)
	args = parser.parse_args()

	return args

def vocab_indeing(vocab_list):

	w2i = {}
	i2w = {}
	for i, w in enumerate(vocab_list):
		w2i[w] = i
		i2w[i] = w

	return w2i, i2w

def load_vocab(vocab_file):

	vocab = []
	with open(vocab_file, 'r') as f:
		for line in f.readlines():
			word = line.strip().split()[0]
			vocab.append(word)

	return vocab



def build_corpus(args, w2i, i2w, window_size, vocab):

	cur_line = 1
	corpus_cur = ''
	fout = open(args.corpus_out, 'w')
	f = open(args.corpus, 'r')
	while True:
		line = f.readline()
		# if not line:
		# 	break
		cur_line += 1

		if line.startswith('[ ') or line.startswith('< < TITLE > >') or line.startswith('Re : [') or not line:
			if corpus_cur == '':
				continue
			print >> sys.stderr, 'current line : '+str(cur_line)
			fout.write(corpus_cur+'\n')
			corpus_cur = ''

		if not line:
			break
			
		corpus_cur += line.strip()+' '

		
def main():

	args = arg_parse()

	vocab_list = load_vocab(args.vocab)

	w2i, i2w = vocab_indeing(vocab_list)

	build_corpus(args=args, w2i=w2i, i2w=i2w, window_size=7, vocab=vocab_list)

if __name__ == '__main__':

	main()