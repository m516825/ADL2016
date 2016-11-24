import sys
import os
import argparse
import random

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='./ADL_data', type=str)
	parser.add_argument('--train_dir', default='./ADL_data/train', type=str)
	parser.add_argument('--test_dir', default='./ADL_data/test', type=str)
	parser.add_argument('--valid_dir', default='./ADL_data/valid', type=str)
	parser.add_argument('--raw_train', default='./atis.train.w-intent.iob', type=str)
	parser.add_argument('--raw_test', default='./atis.test.iob', type=str)
	parser.add_argument('--valid_size', default=200, type=int)
	args = parser.parse_args()

	return args

if __name__ == '__main__':

	args = arg_parse()

	if not os.path.exists(args.data_dir):
		os.makedirs(args.data_dir)

	if not os.path.exists(args.train_dir):
		os.makedirs(args.train_dir)

	if not os.path.exists(args.test_dir):
		os.makedirs(args.test_dir)

	if not os.path.exists(args.valid_dir):
		os.makedirs(args.valid_dir)

	train_data = []
	for line in open(args.raw_train,'r'):
		train_data.append(line.strip())
	# random.shuffle(train_data)

	f_train_lab = open(args.train_dir+'/train.label', 'w')
	f_train_s_i = open(args.train_dir+'/train.seq.in', 'w')
	f_train_s_o = open(args.train_dir+'/train.seq.out', 'w')

	f_valid_lab = open(args.valid_dir+'/valid.label', 'w')
	f_valid_s_i = open(args.valid_dir+'/valid.seq.in', 'w')
	f_valid_s_o = open(args.valid_dir+'/valid.seq.out', 'w')

	f_test_lab = open(args.test_dir+'/test.label', 'w')
	f_test_s_i = open(args.test_dir+'/test.seq.in', 'w')
	f_test_s_o = open(args.test_dir+'/test.seq.out', 'w')

	for i, data in enumerate(train_data):
		s_data = data.split('\t')
		train_seq = s_data[0].split()[:-1]
		train_lab = s_data[1].split()[:-1]
		train_intent = s_data[1].split()[-1]
		seq_out = ' '.join(train_seq)+'\n'
		lab_out = ' '.join(train_lab)+'\n'
		int_out = train_intent+'\n'
		if i < args.valid_size:
			f_valid_lab.write(int_out)
			f_valid_s_i.write(seq_out)
			f_valid_s_o.write(lab_out)
		else:
			f_train_lab.write(int_out)
			f_train_s_i.write(seq_out)
			f_train_s_o.write(lab_out)

		if len(train_seq) != len(train_lab):
			print 'error!!!!'
			sys.exit(0)

	for line in open(args.raw_test,'r'):
		data = line.strip().split()
		test_seq = data[:-1]
		test_lab = len(test_seq)*['O']
		seq_out = ' '.join(test_seq)+'\n'
		lab_out = ' '.join(test_lab)+'\n'
		int_out = 'atis_flight'+'\n'
		f_test_lab.write(int_out)
		f_test_s_i.write(seq_out)
		f_test_s_o.write(lab_out)





