import chainer
import chainer.functions as F
import chainer.links as L

import data


class NdlkanaCNN(chainer.Chain):

	"""An example of CNN for NDLKANA dataset.
	"""
	def __init__(self):
		super(NdlkanaCNN, self).__init__(
			conv1=L.Convolution2D(data.CHANNEL, 20, 11, stride=4),
			conv2=L.Convolution2D(20, 20,  5, pad=1),
			fc3=L.Linear( 80, 500),
			fc4=L.Linear(500,  data.n_class),
		)

	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(
			F.local_response_normalization(self.conv1(x))), 2, stride=2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
		h = F.relu(self.fc3(h))
		return self.fc4(h)
