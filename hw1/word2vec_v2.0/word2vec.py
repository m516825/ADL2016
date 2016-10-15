import argparse
import tensorflow as tf
import numpy as np
import sys

class Data(object):
	def __init__(self, train):
		self.train = train[:,:2].astype(int)
		self.distribution = train[:,2].astype(float) / float(train[:,2].sum())
		self.indexing = [i for i in range(len(train))]
		self.current = 0
		self.length = len(train)
	def next_batch(self, size):

		index = np.random.choice(self.indexing, size, p=self.distribution)
		word, contextw = self.train[index, 0], self.train[index, 1][None, :].T

		return word, contextw

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='../data/hw1/text8', type=str)
	parser.add_argument('--train', default='./skip_gram2.npz.npy', type=str)
	parser.add_argument('--vocab', default='./vocab.out', type=str)
	parser.add_argument('--vector', default='./w2_vector.txt', type=str)
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

def skip_gram(dat, sample_num, iteration, edge_sample, batch_size, learning_rate, vector_size, vocab_size):

	train_x = tf.placeholder(tf.int32, [None])
	train_y = tf.placeholder(tf.int32, [None, 1])

	with tf.device('/cpu:0'):
		wordv = tf.Variable(tf.random_uniform([vocab_size, vector_size], -.1, .1))
		emvt = tf.nn.embedding_lookup(wordv, train_x)
	w_nce = tf.Variable(tf.random_uniform([vocab_size, vector_size], -.1, .1))
	b_nce = tf.Variable(tf.zeros([vocab_size]))

	nce_loss = tf.nn.nce_loss(weights=w_nce, biases=b_nce, inputs=emvt, labels=train_y, 
				num_sampled=sample_num, num_classes=vocab_size, remove_accidental_hits=True)

	cost = tf.reduce_mean(nce_loss)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init = tf.initialize_all_variables()

	sess = tf.Session()

	sess.run(init)

	for i in range(iteration):
		avg_cost = 0.
		progress = 0
		# batch_number = int(dat.length/batch_size) 
		# batch_number += 1 if dat.length%batch_size > 0 else 0
		print >> sys.stderr, 'Iteration '+str(i)+' :'
		for s in range(edge_sample):
			t_x, t_y = dat.next_batch(batch_size)
			
			_, c = sess.run([optimizer, cost], feed_dict={train_x:t_x, train_y:t_y})

			avg_cost += c/float(edge_sample*batch_size)

			if float(s+1)/float(edge_sample)*100. >= progress*5:
				print >> sys.stderr, progress_bar(progress),
				progress += 1

		print >> sys.stderr, '\r>>> cost : '+str(avg_cost) + '                                                   '

	return sess.run(wordv)

def main():

	args = arg_parse()

	vocab_list = load_vocab(args)

	v2i, i2v = vocab_indexing(vocab_list)

	train = load_train_dat(args=args)

	dat = Data(train)

	w_vector = skip_gram(dat=dat, sample_num=10, iteration=1000, edge_sample=1000000, batch_size=10, learning_rate=0.05, vector_size=100, vocab_size=len(v2i))

if __name__ == '__main__':

	main()