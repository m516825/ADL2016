import json 
import re
import sys
import os

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
		input_str = input_str.replace(pair[0], pair[0][1:-1].lower())

	return input_str


if __name__ == '__main__':

	test_path = sys.argv[1]#'./NLG_data/test.txt'
	pred_out = sys.argv[2]
	answer = sys.argv[3] 

	
	test_data = load_test_data(test_path)
	pred_data = load_test_data(pred_out)

	test_fo = open(answer, 'w')
	for i, data in enumerate(test_data):
		format_str, info_format = parse_info(data)
		output = replace_token2str(pred_data[i], info_format)
		test_fo.write(output+'\n')




