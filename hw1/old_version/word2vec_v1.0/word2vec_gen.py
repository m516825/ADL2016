import argparse
import tensorflow as tf
import numpy as np
import sys

class Data(object):
	def __init__(self, args, v2i, window_size):
		self.tokens = self.load_corpus(args.corpus, v2i)
		self.window = window_size
		self.current = 0
		self.iteration = 0
		self.progress = 0
		self.raw_batch = []
	def load_corpus(self, corpus, v2i):
		f = open(corpus, 'r')
		tokens = []
		i_tokens = []
		for line in f.readlines():
			seq = line.strip().split()
			tokens += seq
		for v in tokens:
			vid = v2i.get(v, -1)
			i_tokens.append(vid)
		return i_tokens

	def next_batch(self, size):
		end = False
		state = (False, self.progress)
		while len(self.raw_batch) < size:
			if self.tokens[self.current] < 0:
				self.current += 1
				continue
			lr = np.random.randint(self.window, size=1)[0] + 1
			rr = np.random.randint(self.window, size=1)[0] + 1
			if self.current - lr >= 0:
				for i in range(self.current-lr, self.current):
					if self.tokens[i] >= 0:
						self.raw_batch.append([self.tokens[self.current], self.tokens[i]])
			else:
				for i in range(0, self.current):
					if self.tokens[i] >= 0:
						self.raw_batch.append([self.tokens[self.current], self.tokens[i]])

			if self.current + rr < len(self.tokens):
				for i in range(self.current+1, self.current+rr+1):
					if self.tokens[i] >= 0:
						self.raw_batch.append([self.tokens[self.current], self.tokens[i]])
			else:
				for i in range(self.current+1, len(self.tokens)):
					if self.tokens[i] >= 0:
						self.raw_batch.append([self.tokens[self.current], self.tokens[i]])

			self.current += 1
			
			if self.current >= len(self.tokens):
				self.iteration += 1
				end = True
				break

		if int(float(self.current+1)/float(len(self.tokens))*100.) >= self.progress*5:
			state = (True, self.progress)
			self.progress += 1
			if self.progress > 20:
				self.progress = 0
		if self.current >= len(self.tokens):
			self.current = 0

		batch = np.array(self.raw_batch).astype(int)
		self.raw_batch = []

		return batch[:,0], batch[:,1][None,:].T, state, end

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='../data/hw1/text8', type=str)
	parser.add_argument('--train', default='./skip_gram.npz.npy', type=str)
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
	v = 0
	if not value[0]:
		v = value[1] - 1
	for i in range(v):
		out += '=='
	if v != 20:
		out += '>'
	for i in range(20-v):
		out += '  '
	out += '] '+str(v*5)+'%'

	return out

def skip_gram(dat, sample_num, iteration, batch_size, learning_rate, vector_size, vocab_size):

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

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	init = tf.initialize_all_variables()

	sess = tf.Session()

	sess.run(init)

	for i in range(iteration):
		avg_cost = 0.
		train_size = 0.
		print >> sys.stderr, 'Iteration '+str(i)+' :'
		while True:
			t_x, t_y, state, end = dat.next_batch(batch_size)

			_, c = sess.run([optimizer, cost], feed_dict={train_x:t_x, train_y:t_y})

			train_size += len(t_x)
			avg_cost += c
			
			if state[0] or dat.current%10 == 0:
				print >> sys.stderr, progress_bar(state)+' '+str(dat.current)+'/'+str(len(dat.tokens)),
			if end:
				break

		print >> sys.stderr, '\r>>> cost : '+str(avg_cost/train_size) + '                                                   '

	return sess.run(wordv)

def dump_vector(args, vocab, w):

	fout = open(args.vector, 'w')
	for i, v in enumerate(vocab):
		out = ' '.join([str(val) for val in w[i]])
		out = v + ' ' + out + '\n'
		fout.write(out)

def main():

	args = arg_parse()

	vocab_list = load_vocab(args)

	v2i, i2v = vocab_indexing(vocab_list)

	dat = Data(args, v2i, window_size=5)

	w_vector = skip_gram(dat=dat, sample_num=10, iteration=1, batch_size=100, learning_rate=0.05, vector_size=100, vocab_size=len(v2i))

	dump_vector(args, vocab_list, w_vector)

if __name__ == '__main__':

	main()