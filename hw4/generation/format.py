import json 
import re
import os

def load_json_data(path):

	data = None
	with open(path, 'r') as f:
		data = json.load(f)

	return data

def load_test_data(path):

	data = []
	for line in open(path, 'r'):
		line = line.strip()
		data.append(line)

	return data

def parse_info(string):
	output = []
	info_format = []

	task = string.split('(')[0]
	task = task if task[0] != '?' else task[1:] # remove '?''
	info = re.findall(r"\((.*)?\)", string)[0]

	output.append(task)

	if len(re.findall(r'=', info)) > 0:
		info = info.split(';')
		for i in info:
			data = i.split('=')
			if len(data) == 2:
				token = '_'+data[0].upper()+'_'
				content = data[1] if data[1][0] != "'" and data[1][0] != "'" else data[1][1:-1]
				info_format.append((token, content))
				output.append(token)
			elif len(data) == 1:
				output.append(data[0])
	else:
		output.append(info)

	return ' '.join(output), info_format

def replace_str2token(input_str, info_format):

	for pair in info_format:
		input_str = input_str.replace(pair[1], pair[0])

	return input_str

def replace_token2str(input_str, info_format):

	for pair in info_format:
		input_str = input_str.replace(pair[0], pair[1])

	return input_str


if __name__ == '__main__':

	train_path = './NLG_data/train.json'
	valid_path = './NLG_data/valid.json'
	test_path = './NLG_data/test.txt'

	train_in = './data/train.in'
	train_out = './data/train.out'

	valid_in = './data/valid.in'
	valid_out = './data/valid.out'

	test_in = './data/test.in'

	if not os.path.exists('./data'):
		os.mkdir('./data')


	train_data = load_json_data(train_path)
	valid_data = load_json_data(valid_path)
	test_data = load_test_data(test_path)

	
	t_fi = open(train_in, 'w')
	t_fo = open(train_out, 'w')

	v_fi = open(valid_in, 'w')
	v_fo = open(valid_out, 'w')

	test_fi = open(test_in, 'w')

	for data in train_data:
		format_str, info_format = parse_info(data[0])
		output1 = replace_str2token(data[1], info_format)
		output2 = replace_str2token(data[2], info_format)
		t_fi.write(format_str+'\n')
		t_fi.write(format_str+'\n')
		t_fo.write(output1+'\n')
		t_fo.write(output2+'\n')

	for data in valid_data:
		format_str, info_format = parse_info(data[0])
		output1 = replace_str2token(data[1], info_format)
		output2 = replace_str2token(data[2], info_format)
		v_fi.write(format_str+'\n')
		v_fi.write(format_str+'\n')
		v_fo.write(output1+'\n')
		v_fo.write(output2+'\n')

	for data in test_data:
		format_str, info_format = parse_info(data)
		test_fi.write(format_str+'\n')




