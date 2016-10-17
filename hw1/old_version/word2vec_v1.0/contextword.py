import argparse
import sys
import os
import random
import numpy as np
import math

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='../data/hw1/text8', type=str)
	parser.add_argument('--vocab', default='./vocab.out', type=str)
	parser.add_argument('--contextword', default='./skip_gram', type=str)
	parser.add_argument('--sample', default=9e-17, type=float)
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

def merge_cooccur(args, file_num, vocab_size):

	fin = []
	file_min = {}
	for i in range(file_num):
		fin.append(open(args.cooccur+'_'+str(i)+'.out', 'r'))

	fout = open(args.cooccur+'_o.out', 'w')
	for i in range(file_num):
		file_min[i] = map(float, fin[i].readline().strip().split(' '))

	cur_dict = {}
	min_id = float('Inf')
	find_flag = 1
	finished = 0

	while True:
		working = file_num
		# fine min number
		if find_flag == 1:
			min_id = float('Inf')
			for i in range(file_num):
				if file_min[i][0] < min_id:
					min_id = file_min[i][0]
			
			if len(cur_dict) > 0:
				for k, v in cur_dict.iteritems():
					fout.write(str(int(k[0]))+' '+str(int(k[1]))+' '+str(v)+'\n')
				cur_dict = {}
				finished += 1
				print >> sys.stderr, '\r'+str(finished)+'/'+str(vocab_size), 

		find_flag = 1

		for i in range(file_num):
			if file_min[i][0] == float('Inf'):
				working -= 1
				continue
			if file_min[i][0] == min_id:
				cur_dict[(int(file_min[i][0]), int(file_min[i][1]))] = cur_dict.get((int(file_min[i][0]), int(file_min[i][1])), 0.) + float(file_min[i][2])
				line = fin[i].readline()
				if not line:
					file_min[i][0] = float('Inf')
					continue
				file_min[i] = map(float, line.strip().split(' '))
				if file_min[i][0] == min_id:
					find_flag = 0

		if working == 0:
			break

	for i in range(file_num):
		os.remove(args.cooccur+'_'+str(i)+'.out')

	print ''

def shuffle(tlist):
	for i in range(len(tlist)):
		r = random.randint(0, len(tlist)-1)
		tmp = tlist[i]
		tlist[i] = tlist[r]
		tlist[r] = tmp

	return tlist

def shuffle_cooccur(args):

	file_num = 0
	orderDat = []
	with open(args.cooccur+'_o.out', 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			dat = line.strip()
			orderDat.append(dat)
			if len(orderDat) >= 5000000:

				orderDat = shuffle(orderDat)

				with open(args.cooccur+'_s_'+str(file_num)+'.out', 'w') as fo:
					for item in orderDat:
						out = str(item)+'\n'
						fo.write(out)
					orderDat = []
					file_num += 1

		if len(orderDat) > 0:
			orderDat = shuffle(orderDat)
			with open(args.cooccur+'_s_'+str(file_num)+'.out', 'w') as fo:
				for item in orderDat:
					out = str(item)+'\n'
					fo.write(out)
				orderDat = []
				file_num += 1
	fin = []				

	for i in range(file_num):
		fin.append(open(args.cooccur+'_s_'+str(i)+'.out', 'r'))

	with open(args.cooccur+'.out', 'w') as fout:
		while True:

			if len(fin) == 0:
				break 
			r = random.randint(0, len(fin)-1)
			
			line = fin[r].readline()

			if not line:
				tmp = []
				for i, f in enumerate(fin):
					if i != r:
						tmp.append(f)
				fin = tmp
				continue
			line = line.strip().split(' ')
			out = str(line[0])+' '+str(line[1])+' '+str(line[2])+'\n'
			fout.write(out)

	for i in range(file_num):
		os.remove(args.cooccur+'_s_'+str(i)+'.out') 

def build_cooccur(args, w2i, i2w, window_size, vocab, sum_c, i2c, dumpAll=True):

	cooccur = []
	file_num = 0
	with open(args.corpus, 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break

			tokens = line.strip().split()

			total = len(tokens)
			i_tokens = []
			for v in tokens:
				vid = w2i.get(v, -1)
				# subsampling in word2vec 'the' 0.629 'of' 0.841 at sample 9e-17
				if vid != -1:
					ran = (math.sqrt(float(i2c[vid])/args.sample * float(sum_c)) + 1) * (args.sample * float(sum_c) / float(i2c[vid]))
					r = random.random()
					if r > ran:
						vid = -1

				i_tokens.append(vid)
			tokens = []

			for index, c_i in enumerate(i_tokens):
				if c_i < 0:
					continue
				lr = np.random.randint(window_size, size=1)[0] + 1
				rr = np.random.randint(window_size, size=1)[0] + 1
				if c_i - lr >= 0:
					for i in range(c_i-lr, c_i):
						if i_tokens[i] >= 0:
							cooccur.append([c_i, i_tokens[i]])
				else:
					for i in range(0, c_i):
						if i_tokens[i] >= 0:
							cooccur.append([c_i, i_tokens[i]])

				if c_i + rr < len(i_tokens):
					for i in range(c_i+1, c_i+rr+1):
						if i_tokens[i] >= 0:
							cooccur.append([c_i, i_tokens[i]])
				else:
					for i in range(c_i+1, len(i_tokens)):
						if i_tokens[i] >= 0:
							cooccur.append([c_i, i_tokens[i]])

				print >> sys.stderr, '\rdone process '+str(index)+'/'+str(total)+' tokens in currnet line',

			print >> sys.stderr, ''
	
	
	if len(cooccur) > 0 and dumpAll:
		np_cooccur = np.array(cooccur)
		np.random.shuffle(np_cooccur)
		np.save(args.contextword+'.npz', np_cooccur)

	print >> sys.stderr, 'done dumping skip-gram pair file'

def load_vocab_info(vocab_file):
	sum_c = 0
	i2c = {}
	count_list = []
	with open(vocab_file, 'r') as f:
		for line in f.readlines():
			count = int(line.strip().split()[1])
			count_list.append(count)
	for i, c in enumerate(count_list):
		sum_c += c
		i2c[i] = c

	return sum_c, i2c
		
def main():

	args = arg_parse()

	vocab_list = load_vocab(args.vocab)

	w2i, i2w = vocab_indeing(vocab_list)

	sum_c, i2c = load_vocab_info(args.vocab)

	build_cooccur(args=args, w2i=w2i, i2w=i2w, window_size=5, vocab=vocab_list, sum_c=sum_c, i2c=i2c, dumpAll=True)

if __name__ == '__main__':
	main()