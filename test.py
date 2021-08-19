from pathlib import Path
import sys
sys.path.append(str(Path('./example/ndlkana').absolute()))
import net
import data

import numpy as np
import random
from PIL import Image

import chainer
import chainer.links as L
from chainer import serializers

# Prepare dataset
print('load NDLKANA dataset')
ndlkana = data.load_ndlkana_data('hiragana', '*.jpg', 1.0/7.0)
ndlkana['data'] = ndlkana['data'].astype(np.float32)
ndlkana['data'] /= 255
n_test = ndlkana['testsize']
n_train = ndlkana['data'].shape[0] - n_test
print("n_train={} n_test={}".format(n_train, n_test))

_, x_test = np.split(ndlkana['data'], [n_train])

# Prepare CNN model, defined in net.py
model = L.Classifier(net.NdlkanaCNN())
serializers.load_npz('hiragana.model', model)

########

# 推測の対象画像をランダム選択
i = random.randrange(n_test)
img = x_test[i:i + 1]
print('--画像')
# display(Image.fromarray(np.uint8(img[0][0]*255)))

Image.fromarray(np.uint8(img[0][0]*255)).show()

# 推測
x = chainer.Variable(np.asarray(img))
with chainer.using_config('train', False):
    y = model.predictor(x)
hexcode = ndlkana['label'][y.data[0].argmax()]
# print('--候補一位の文字')
# print(chr(int(hexcode.replace('U', '0x'), 16)))
# print(hexcode)

# print('--候補ぜんぶ（ラベルのインデックス）')
b = y.data[0]

'''
print(b)

print("--------")
'''

c = np.argsort(b)

# print(c)

for i in c[-1:-6:-1]:
    print(i, ndlkana['label'][i], b[i])

'''
print("--------")

a = c[-1::-1]
print(a)
print('--確率ぜんぶ')
print(np.sort(y.data[0])[-1::-1])
'''