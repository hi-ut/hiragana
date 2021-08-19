from __future__ import print_function
import glob
import os
import numpy as np
from PIL import Image
import six

DIM = 48						# 48x48 pixel image
CHANNEL = 1						# grayscale image
n_class = 0						# number of classes (=characters)


def load_dirdata(class_, dir, fspec, testratio):
	label = os.path.split(dir)[1]
	files = glob.glob(os.path.join(dir, fspec))

	num = len(files)
	n_test = int(num * testratio)
	n_train = num - n_test
	print("{}: {} train={} test={}".format(class_, dir, n_train, n_test))
	d = np.zeros(num * CHANNEL * DIM * DIM, dtype=np.uint8)\
					.reshape((num, CHANNEL, DIM, DIM))
	t = np.zeros(num, dtype=np.uint8).reshape((num, ))
	t += class_

	for i, file in enumerate(files):
		im = Image.open(file).convert("L")
		d[i, 0] = np.asarray(im)

	perm = np.random.permutation(num)
	d_train = d[perm[:n_train]]
	d_test = d[perm[n_train:]]

	return d_train, d_test, np.split(t, [n_train]), label


def load_ndlkana(datadir, fspec, testratio):
	global n_class

	dirs = glob.glob(os.path.join(datadir, "*"))
	n_class = len(dirs)
	label = [''] * n_class
	testsize = 0

	for i, dir in enumerate(dirs):
		d_train, d_test, t, l = load_dirdata(i, dir, fspec, testratio)
		t_train, t_test = t
		label[i] = l
		testsize += d_test.shape[0]
		if i == 0:
			data_train = d_train
			data_test = d_test
			target_train = t_train
			target_test = t_test
		else:
			data_train = np.append(data_train, d_train, axis=0)
			data_test = np.append(data_test, d_test, axis=0)
			target_train = np.append(target_train, t_train, axis=0)
			target_test = np.append(target_test, t_test, axis=0)

	data = np.append(data_train, data_test, axis=0)
	target = np.append(target_train, target_test, axis=0)
	print("data={} target={} testsize={}".format(
			data.shape, target.shape, testsize))
	return data, target, label, testsize


def download_ndlkana_data(datadir, fspec, testratio):
	print('Converting data...')
	data, target, label, testsize = load_ndlkana(datadir, fspec, testratio)
	ndlkana = {'data': data, 'target': target,
				'label': label, 'testsize': testsize}
	print('Done')

	print('Saving data...')
	pickle = os.path.split(datadir)[1] + '.pkl'
	with open(pickle, 'wb') as output:
		six.moves.cPickle.dump(ndlkana, output, -1)
	print('Done')
	print('Convert completed')


def load_ndlkana_data(datadir, fspec, testratio):
	global n_class
	pickle = os.path.split(datadir)[1] + '.pkl'

	if not os.path.exists(pickle):
		download_ndlkana_data(datadir, fspec, testratio)

	with open(pickle, 'rb') as ndlkana_pickle:
		ndlkana = six.moves.cPickle.load(ndlkana_pickle)

	n_class = len(ndlkana['label'])	# referred from net.py
	return ndlkana
