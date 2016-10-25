import argparse
import sys
import os
import random
import numpy as np

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='./corpus/ptt_corpus_title.txt', type=str)
	parser.add_argument('--vocab', default='./vocab_ch.out', type=str)
	parser.add_argument('--cooccur', default='./cooccur_ch', type=str)
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
	total_line = 0

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
					total_line += 1
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

	print >> sys.stderr, '\ntotal data : '+str(total_line)

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

def np_shuffle(args, file_num):

	data = np.array([])
	for i in range(file_num):
		if i == 0:
			data = np.load(args.cooccur+'_'+str(i)+'.npz.npy')
		else:
			in_data = np.load(args.cooccur+'_'+str(i)+'.npz.npy')
			data = np.concatenate((data, in_data), axis=0)
	print 'done loading'

	p = np.random.permutation(len(data))
	data = data[p]

	np.save(args.cooccur+'.npz.npy', data.astype('float32'))

def to_numpy(args):

	numpy_list = []
	current = 0
	file_num = 0
	with open(args.cooccur+'_o.out', 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			current += 1
			print >> sys.stderr, '\rdone processing '+str(current)+' lines',
			line = line.strip().split()
			numpy_list.append([int(line[0]), int(line[1]), float(line[2])])
			if len(numpy_list) > 30000000:
				np.save(args.cooccur+'_'+str(file_num)+'.npz', np.array(numpy_list).astype('float32'))
				numpy_list = []
				file_num += 1

	os.remove(args.cooccur+'_o.out')

	if len(numpy_list) > 0:
		np.save(args.cooccur+'_'+str(file_num)+'.npz', np.array(numpy_list).astype('float32'))
		numpy_list = []
		file_num += 1

	print >> sys.stderr, '\ndone dumping numpy data, total',file_num

	np_shuffle(args, file_num)

def build_cooccur(args, w2i, i2w, window_size, vocab, symmetric=True, dumpAll=True):

	cooccur = {}
	file_num = 0
	cur_line = 1
	with open(args.corpus, 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break

			# print >> sys.stderr, 'current line : ',cur_line
			cur_line += 1

			tokens = line.strip().split()

			total = len(tokens)
			window = [-1]*window_size
			tail = 0
			# print >> sys.stderr, '-------------------',total,'-------------------'

			for i, token in enumerate(tokens):
				if token == '\n':
					tail = 0
					continue

				w2 = w2i.get(token, -1)
				# if w2 == -1:
				# 	continue

				current = tail - 1
				head = tail - window_size - 1 if tail > window_size else 0 - 1

				for index in range(current, head, -1):
					w1 = window[index%window_size]

					if w1 != -1 and w2 != -1:
						cooccur[(w1, w2)] = cooccur.get((w1, w2), 0.) + 1./float(current-index+1)
						if symmetric:
							cooccur[(w2, w1)] = cooccur.get((w2, w1), 0.) + 1./float(current-index+1)

				window[tail%window_size] = w2
				tail += 1

				# print >> sys.stderr, '\rdone process '+str(i+1)+'/'+str(total)+' tokens in currnet line\t'+str(cur_line),

				if len(cooccur) >= 10000000 and i != 0 and not dumpAll:
					with open(args.cooccur+'_'+str(file_num)+'.out', 'w') as ft:
						for k, v in sorted(cooccur.iteritems(), key=lambda (k, v): int(k[0])):
							out = str(k[0])+' '+str(k[1])+' '+str(v)+'\n'
							ft.write(out)
						cooccur = {}
					file_num += 1

			# print >> sys.stderr, ''
			if cur_line % 1000 == 0:
				print >> sys.stderr, cur_line

		if len(cooccur) > 0 and not dumpAll:
			with open(args.cooccur+'_'+str(file_num)+'.out', 'w') as ft:
				for k, v in sorted(cooccur.iteritems(), key=lambda (k, v): int(k[0])):
					out = str(k[0])+' '+str(k[1])+' '+str(v)+'\n'
					ft.write(out)
				cooccur = {}
			file_num += 1

	
	
	if not dumpAll:
		print >> sys.stderr, 'done dumping all small cooccur file'

		merge_cooccur(args, file_num, len(w2i))

		print >> sys.stderr, 'done merging all small file'

		to_numpy(args)

	else:
		if len(cooccur) > 0 and dumpAll:

			cooccur_list = []
			for k, v in cooccur.iteritems():
				cooccur_list.append([int(k[0]), int(k[1]), float(v)])
			cooccur = {}
			# cooccur_list = shuffle(cooccur_list)
			np_cooccur = np.array(cooccur_list)
			np.save(args.cooccur+'.npz', np_cooccur)

		print >> sys.stderr, 'done dumping cooccur file'
		# sys.exit(0)
			
	# print >> sys.stderr, 'start shuffling cooccur file'

	# shuffle_cooccur(args)

	# print >> sys.stderr, 'done shuffling cooccur file'

		
def main():

	args = arg_parse()

	vocab_list = load_vocab(args.vocab)

	w2i, i2w = vocab_indeing(vocab_list)

	build_cooccur(args=args, w2i=w2i, i2w=i2w, window_size=4, vocab=vocab_list, symmetric=True, dumpAll=False)
	# to_numpy(args)

if __name__ == '__main__':

	main()