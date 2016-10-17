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
		
		if self.current == 0:
			print >> sys.stderr, 'data shuffling'
			np.random.shuffle(self.train)

		if self.current + size < self.length:
			vid, co = self.train[self.current : self.current + size, 0:2], self.train[self.current : self.current + size, 2][None,:].T
			max_index = np.where(co >= 100.)[0]
			weight = np.power(co, 3./4.)
			weight[max_index] = 100.
			
			self.current += size
			state = (False, self.progress)
			if int(float(self.current+1)/float(self.length)*100) >= self.progress*2:
				state = (True, self.progress)
				self.progress += 1

			return vid.astype(int), co, weight, state
		else:
			vid, co = self.train[self.current:, 0:2], self.train[self.current:, 2][None,:].T
			max_index = np.where(co >= 100.)[0]
			weight = np.power(co, 3./4.)
			weight[max_index] = 100.

			self.current = 0
			self.iteration += 1
			self.progress = 0

			return vid.astype(int), co, weight, (True, 20)


def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='../data/hw1/text8', type=str)
	parser.add_argument('--train', default='./data/cooccur_w10.npz.npy', type=str)
	parser.add_argument('--vocab', default='./vocab.out', type=str)
	parser.add_argument('--vector', default='./g_vector', type=str)
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
	if value != 50:
		out += '>'
	for i in range(50-value):
		out += '  '
	out += '] '+str(value*2)+'%'

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

	with tf.device('/cpu:0'):
		wi = tf.Variable(tf.random_normal([vocab_size, vector_size], -.1, .1))
		wj = tf.Variable(tf.random_normal([vocab_size, vector_size], -.1, .1))
		bi = tf.Variable(tf.zeros([vocab_size, 1]))
		bj = tf.Variable(tf.zeros([vocab_size, 1]))

		emwi = tf.nn.embedding_lookup(wi, v_pair[:, 0])
		emwj = tf.nn.embedding_lookup(wj, v_pair[:, 1])

		embi = tf.nn.embedding_lookup(bi, v_pair[:, 0])
		embj = tf.nn.embedding_lookup(bj, v_pair[:, 1])


	pred = tf.reduce_sum(tf.mul(emwi, emwj), reduction_indices=1, keep_dims=True) + embi + embj

	cost = tf.reduce_mean(tf.mul(weight, tf.pow(pred - tf.log(xij), 2)))

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

			avg_cost += c/float(batch_number)

			if state[0]:
				print >> sys.stderr, progress_bar(state[1]),

		print >> sys.stderr, '\r>>> cost : '+str(avg_cost) + '                                                   '

	args.vector += '_ite['+str(iteration)+']_bat['+str(batch_size)+']_vec['+str(vector_size)+']_eta['+str(learning_rate)+'].txt'

	return sess.run(wi), sess.run(wj)

def main():

	args = arg_parse()

	vocab_list = load_vocab(args)

	v2i, i2v = vocab_indexing(vocab_list)

	train = load_train_dat(args=args)

	dat = Data(train)

	wi, wj = glove_model(dat=dat, iteration=50, batch_size=100, learning_rate=0.05, alpha=0.75, x_max=100, vector_size=300, vocab_size=len(v2i))

	dump_vector(args, vocab_list, wi, wj)

if __name__ == '__main__':
	main()
