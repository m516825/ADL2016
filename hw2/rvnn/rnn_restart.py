import numpy as np
import tensorflow as tf
import sys
import os
import argparse
import tree
import time
import progressbar as pb
import matplotlib.pyplot as plt
import cPickle as pickle

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--pos_data', default='./dataset_rnn_eng/training_data.pos.tree', type=str)
	parser.add_argument('--neg_data', default='./dataset_rnn_eng/training_data.neg.tree', type=str)
	parser.add_argument('--testing_data', default='./dataset_rnn_eng/testing_data.txt.tree', type=str)
	parser.add_argument('--answer', default='./answer.txt', type=str)
	parser.add_argument('--type', default='train', type=str)
	parser.add_argument('--mpath', default='', type=str)
	args = parser.parse_args()

	return args

class Vocab(object):
	def __init__(self, vocab_list):

		self.build_vocab(vocab_list)
		self.build_indexing()
		self.length = len(self.vocabs)

	def build_vocab(self, vocab_list):
		vocabs = {}
		for v in vocab_list:
			vocabs[v] = vocabs.get(v, 0) + 1
		vocab_list = ['<unk>']
		for k, v in sorted(vocabs.iteritems(), key=lambda (k, v):v):
			vocab_list += [k]
		self.vocabs = vocab_list

	def build_indexing(self):
		self.v2i = {}
		self.i2v = {}
		for i, v in enumerate(self.vocabs):
			self.v2i[v] = i
			self.i2v[i] = v

class Config(object):

	embed_size = 50
	pos_label_size = None
	s_label_size = 2
	# early_stopping = 2
	anneal_threshold = 0.9987
	anneal_by = 1.087
	epochs = 30
	lr = 0.01
	l2 = 0.000
	model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)

class Rnn_Model(object):

	def __init__(self, args, conf, train_tree):
		self.conf = conf
		self.args = args
		self.load_data(train_tree)
		self.vocab.length = 10626 # only for test model

	def load_data(self, train_tree):
		words = []
		for t in train_tree:
			words += t.leaves()
		self.vocab = Vocab(words)
		self.train_size = len(train_tree)
		self.train_tree = train_tree

	def loss(self, logits, labels):

		loss = None
		with tf.variable_scope('Composition', reuse=True):
			W1 = tf.get_variable('W1')
		with tf.variable_scope('Projection', reuse=True):
			Us = tf.get_variable('Us')
		l2loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(Us)

		cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
		loss = cross_entropy + self.conf.l2 * l2loss

		return loss

	def Optimizer(self, loss):

		Optimizer = tf.train.GradientDescentOptimizer(self.conf.lr).minimize(loss)

		return Optimizer

	def init_model_variable(self):
		
		with tf.variable_scope('Composition'):
			tf.get_variable('embedding', shape=[self.vocab.length, self.conf.embed_size])
			tf.get_variable('W1', shape=[2*self.conf.embed_size, self.conf.embed_size])
			tf.get_variable('b1', shape=[1, self.conf.embed_size])
		
		with tf.variable_scope('Projection'):
			tf.get_variable('Us', shape=[self.conf.embed_size, self.conf.s_label_size])
			tf.get_variable('bs', shape=[1, self.conf.s_label_size])

	def get_labels(self, trees):

		labels = []
		for t in trees:
			labels.append([t.sem])

		return labels
	
	def inference(self, trees, predict_only_root=True):

		node_tensors = self.add_model(trees, isRoot=True)
		if predict_only_root:
			node_tensors = node_tensors[trees.tid]

		return node_tensors

	def add_model(self, node, isRoot=False):

		with tf.variable_scope('Composition', reuse=True):
			embedding = tf.get_variable('embedding')
			W1 = tf.get_variable('W1')
			b1 = tf.get_variable('b1')

		with tf.variable_scope('Projection', reuse=True):
			Us = tf.get_variable('Us')
			bs = tf.get_variable('bs')

		node_tensors = dict()
		curr_node_tensor = None
		if node.isLeaf:
			index = self.vocab.v2i.get(node.word, 0)
			vec = tf.gather(embedding, indices=index)
			curr_node_tensor = tf.expand_dims(vec, 0)
		else:
			node_tensors.update(self.add_model(node.left))
			node_tensors.update(self.add_model(node.right))
			VlVr = tf.concat(1, [node_tensors[node.left.tid], node_tensors[node.right.tid]])
			
			curr_node_tensor = tf.nn.relu(tf.matmul(VlVr, W1) + b1)

		if isRoot:
			curr_node_tensor = tf.matmul(curr_node_tensor, Us) + bs

		node_tensors[node.tid] = curr_node_tensor

		return node_tensors

	def predictions(self, y):

		predictions = None
		predictions = tf.argmax(y, 1)

		return predictions

	def predict(self, trees, path=None):
		print >> sys.stderr, 'Start prediction..'
		pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(trees)).start()
		result = []
		step = 0
		RESTART_AFTER = 50
		predict_size = len(trees)
		while step < predict_size:
			with tf.Graph().as_default(), tf.Session() as sess:
				self.init_model_variable()
				saver = tf.train.Saver()
				saver.restore(sess, path)
				for _ in range(RESTART_AFTER):
					if step >= predict_size:
						break
					logit = self.inference(trees[step])
					prediction = self.predictions(logit)
					root_pred = sess.run(prediction)[0]
					result.append(root_pred)
					step += 1
					pbar.update(step)
		pbar.finish()

		return result

	def make_conf(self, predictions, labels):

		confmat = np.zeros([2, 2])
		for p, l in zip(predictions, labels):
			confmat[p, l] += 1
		return confmat

	def run_epoch(self, new=False):
		RESTART_AFTER = 50
		cost = 0.
		#### Visualize process
		pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=self.train_size).start()
		####
		step = 0
		while step < self.train_size:
			with tf.Graph().as_default(), tf.Session() as sess:
				self.init_model_variable()
				if new:
					init = tf.initialize_all_variables()
					sess.run(init)
				else:
					saver = tf.train.Saver()
					saver.restore(sess, './model/%s.temp'%self.conf.model_name)
				for _ in range(RESTART_AFTER):
					if step >= self.train_size:
						break
					tree = self.train_tree[step]
					logit = self.inference(tree)
					label = self.labels[step]
					loss = self.loss(logit, label)
					Optimizer = self.Optimizer(loss)
					c, _ = sess.run([loss, Optimizer])
					cost += float(c)/float(self.train_size)

					step += 1
					#### Visualize process
					pbar.update(step)
					####

				saver = tf.train.Saver()
				if not os.path.exists("./model"):	
					os.makedirs("./model")
				saver.save(sess, './model/%s.temp'%self.conf.model_name)
		pbar.finish()


		train_pred = self.predict(self.train_tree, path='./model/%s.temp'%self.conf.model_name)
		train_label = np.array(self.labels).T[0]
		train_acc = np.equal(train_pred, train_label).mean()

		print '\rcost : {}, training accuracy : {}'.format(cost, train_acc)+'                                         '
		print self.make_conf(train_pred, train_label)

		return train_acc, cost

	def train(self):

		accuracy_history = []
		loss_history = []
		prev_epoch_loss = float('inf')
		self.labels = self.get_labels(self.train_tree)

		for epk in range(self.conf.epochs):
			print >> sys.stderr, "Iteritems {} : ".format(epk+1)
			if epk == 0:
				train_acc, loss = self.run_epoch(new=True)
			else:
				train_acc, loss = self.run_epoch()

			accuracy_history.append(train_acc)
			loss_history.append(loss)

			epoch_loss = np.mean(loss_history)
			if epoch_loss>prev_epoch_loss*self.conf.anneal_threshold:
				self.conf.lr/=self.conf.anneal_by
				print 'annealed lr to %f'%self.conf.lr
			prev_epoch_loss = epoch_loss
			
		return loss_history, accuracy_history


def main():

	args = arg_parse()
	conf = Config()
	train_tree, test_tree, label_size = tree.load_tree_from_data(args)
	rnn = Rnn_Model(args, conf, train_tree)
	if args.type == 'train':
		loss_history, accuracy_history = rnn.train()
	print >> sys.stderr, "Start inference testing data..."
	pred_results = None
	if args.mpath == '':
		pred_results = rnn.predict(test_tree, path='./model/%s.temp'%conf.model_name)
	else:
		v2i = pickle.load(open('./rvnn/indexing', 'rb'))
		rnn.vocab.v2i = v2i
		pred_results = rnn.predict(test_tree, path=args.mpath)

	# print pred_results

	with open(args.answer, 'w') as f:
		for i in pred_results:
			f.write(str(i)+'\n')

	# plt.plot(loss_history)
	# plt.plot(accuracy_history)
	# plt.title('Loss/accuracy history')
	# plt.xlabel('Iteration')
	# plt.ylabel('value')
	# plt.savefig("history.png")

if __name__ == '__main__':
	main()
