import json 
import sys
import os
import nltk 
import argparse
import numpy as np

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='./Squad', type=str)
	parser.add_argument('--ans', default='./output.txt', type=str)
	parser.add_argument('--cmp', default='F1', type=str)
	parser.add_argument('--verbose', default=False, type=bool)
	parser.add_argument('--verbose_s', default=False, type=bool)
	parser.add_argument('--t', default=0., type=float)
	parser.add_argument('--test', default='testing_data.json', type=str)
	# parser.add_argument('--pred', default='../dev-020000.json', type=str)
	args = parser.parse_args()
	if args.data_dir == './Squad':
		args.pred = ['./out/basic/00/answer/test-020000.json', './out/basic/00/answer/test-016000.json', \
                    './out/basic/00/answer/test-015000.json', './out/basic/01/answer/test-020000.json', \
                    './out/basic/01/answer/test-015000.json']
	else:
		args.pred = ['./out/basic/00/answer/test-020000.json', './out/basic/00/answer/test-019000.json', \
                    './out/basic/00/answer/test-018000.json', './out/basic/00/answer/test-017000.json', \
                    './out/basic/00/answer/test-016000.json']

	print args.pred

	return args

def load_data(args):
	pred = []
	source = None
	for pfile in args.pred:
		with open(pfile, 'r') as f:
			p = json.load(f)
			pred.append(p)
	with open(args.test, 'r') as f:
		source = json.load(f)
	return pred, source

def w_process(string):

	string = string.replace('"', '')
	tokens = nltk.word_tokenize(string)

	return tokens

def find_max_F1(answer_list, pred):

	score = []
	for ans in answer_list:
		tp = fp = fn = 0.
		for a in ans:
			if a in pred:
				tp += 1.
			else:
				fp += 1.
		p = tp/(tp+fp)
		r = 0. if len(pred) == 0 else tp/float(len(pred))
		s = 0. if (p+r) == 0. else (p*r/(p+r))*2
		score.append(s)

	return np.argmax(np.array(score)), score[np.argmax(np.array(score))]

def main():

	args = arg_parse()

	pred, source = load_data(args)

	answer = []

	for pi in pred:
		a = []
		for i, test in enumerate(source):
			tid = args.data_dir[2:]+'_test_'+str(i)
			answer_list = test['answer_list']

			answer_list = [w_process(ans) for ans in answer_list]
			p = pi[tid]
			p = w_process(p)

			if args.cmp == 'F1':
				index, score = find_max_F1(answer_list, p)
				a.append((index, score))
				if args.verbose_s:
					try:
						print 'pred: {},\tans: {}'.format(' '.join(p), ' '.join(answer_list[index]))
					except:
						print 'unable to print ascii'
			else:
				print 'not implemented'
		answer.append(a)

	count = np.zeros((len(source), len(source[0]['answer_list'])))
	for ans in answer:
		for i, pair in enumerate(ans):
			if pair[1] > args.t:
				count[i][pair[0]] += 1
	answer = [np.argmax(c) for c in count]
	if args.verbose:
		for i, c in enumerate(count):
			print c, answer[i]

	with open(args.ans, 'w') as f:
		for ans in answer:
			f.write(str(ans)+'\n')

if __name__  == '__main__':
	main()
