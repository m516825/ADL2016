import sys

def main():

	pred = sys.argv[2]
	ans = sys.argv[1]

	p = []
	a = []

	with open(pred, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			p.append(int(line))

	with open(ans, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			a.append(int(line))

	error = 0.


	for i, a_i in enumerate(a):
		if a_i != p[i]:
			error += 1.

	print 'acc: {}'.format(1. - error/float(len(a)))

if __name__ == '__main__':
	main()