import argparse

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', default='../data/hw1/text8', type=str)
	parser.add_argument('--vocab', default='./vocab.out', type=str)
	args = parser.parse_args()

	return args

def build_vocab(corpus):

	vocab = dict()
	with open(corpus, 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			words = line.strip().split()

			for w in words:
				vocab[w] = vocab.get(w, 0) + 1
	return vocab

def dump_vocab(vocab, vocab_file):

	with open(vocab_file, 'w') as f:
		for k, v in sorted(vocab.iteritems(), key=lambda (k, v): v, reverse=True):
			f.write(str(k)+' '+str(v)+'\n')

def main():

	args = arg_parse()

	vocab_dict = build_vocab(args.corpus)

	dump_vocab(vocab_dict, args.vocab)

if __name__ == '__main__':
	main()
