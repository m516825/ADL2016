import tensorflow as tf
import argparse
import numpy as np
import sys

class Data(object):
	def __init__(self, train):
		self.train = train
		self.current = 0
		self.length = len(self.train)
		self.iteration = 0
		self.progress = 0
	def next_batch(self, size):
		if self.current + size < self.length:
			vid, co = self.train[self.current : self.current + size, 0:2], self.train[self.current : self.current + size, 2][None,:].T
			max_index = np.where(co >= 100.)[0]
			weight = co**(3./4.)
			weight[max_index] = 100.
			self.current += size
			state = (False, self.progress)
			if int(float(self.current+1)/float(self.length)*100) >= self.progress*5:
				state = (True, self.progress)
				self.progress += 1

			return vid.astype(int), co, weight, state
		else:
			vid, co = self.train[self.current:, 0:2], self.train[self.current:, 2][None,:].T
			max_index = np.where(co >= 100.)[0]
			weight = co**(3./4.)
			weight[max_index] = 100.

			self.current = 0
			self.iteration += 1
			self.progress = 0
			np.random.shuffle(self.train)
			return vid.astype(int), co, weight, (True, 20)


def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='../data/hw1/text8', type=str)
	parser.add_argument('--train', default='./cooccur.npz.npy', type=str)
	parser.add_argument('--vocab', default='./vocab.out', type=str)
	parser.add_argument('--vector', default='./vector.txt', type=str)
	parser.add_argument('--vtype', default=1, type=int)
	args = parser.parse_args()

	return args

def load_vocab(args):
	
	vocab = []
	for line in open(args.vocab, 'r'):
		word = line.strip().split()[0]
		vocab.append(word)

	return vocab

def vocab_indexing(vocab):

	v2i = {}
	i2v = {}
	for i, v in enumerate(vocab):
		v2i[v] = i
		i2v[i] = v

	return v2i, i2v

def load_train_dat(args):

	train = np.load(args.train)
	
	return train

def progress_bar(value):
	out = '\rprogress : ['
	for i in range(value):
		out += '=='
	if value != 20:
		out += '>'
	for i in range(20-value):
		out += '  '
	out += '] '+str(value*5)+'%'

	return out

def dump_vector(args, vocab, wi, wj):
	w = ''
	if args.vtype == 1:
		w = wi + wj
	else:
		w = wi

	fout = open(args.vector, 'w')
	for i, v in enumerate(vocab):
		out = ' '.join([str(val) for val in w[i]])
		out = v + ' ' + out + '\n'
		fout.write(out)

def glove_model(dat, iteration, batch_size, learning_rate, alpha, x_max, vector_size, vocab_size):

	v_pair = tf.placeholder(tf.int32, [None, 2])
	xij = tf.placeholder(tf.float32, [None, 1])
	weight = tf.placeholder(tf.float32, [None, 1])

	wi = tf.Variable(tf.random_normal([vocab_size, vector_size]))
	wj = tf.Variable(tf.random_normal([vocab_size, vector_size]))
	bi = tf.Variable(tf.zeros([vocab_size, 1]))
	bj = tf.Variable(tf.zeros([vocab_size, 1]))

	pred = tf.reduce_sum(tf.mul(tf.gather(wi, v_pair[:, 0]), tf.gather(wj, v_pair[:, 1])), reduction_indices=1, keep_dims=True) + tf.gather(bi, v_pair[:, 0]) + tf.gather(bj, v_pair[:, 1])

	cost = tf.reduce_sum(tf.mul(weight, tf.pow(pred - tf.log(xij), 2)))

	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

	init = tf.initialize_all_variables()

	

	sess = tf.Session()
	sess.run(init)

	for i in range(iteration):
		avg_cost = 0.

		batch_number = int(dat.length/batch_size) 
		batch_number += 1 if dat.length%batch_size > 0 else 0

		print >> sys.stderr, 'Iteration '+str(i)+' :'
		for b in range(batch_number):
			pair, co, w, state = dat.next_batch(batch_size)
			
			_, c = sess.run([optimizer, cost], feed_dict={v_pair:pair, xij:co, weight:w})

			avg_cost += c/float(dat.length)

			if state[0]:
				print >> sys.stderr, progress_bar(state[1]),

		print >> sys.stderr, '\r>>> cost : '+str(avg_cost) + '                                                   '

	return sess.run(wi), sess.run(wj)

def main():

	args = arg_parse()

	vocab_list = load_vocab(args)

	v2i, i2v = vocab_indexing(vocab_list)

	train = load_train_dat(args=args)

	dat = Data(train)

	wi, wj = glove_model(dat=dat, iteration=50, batch_size=200, learning_rate=0.05, alpha=0.75, x_max=100, vector_size=100, vocab_size=len(v2i))

	dump_vector(args, vocab_list, wi, wj)

if __name__ == '__main__':
	main()
