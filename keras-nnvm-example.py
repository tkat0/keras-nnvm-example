from time import time

from PIL import Image
from matplotlib import pyplot as plt

import nnvm
import tvm
from tvm.contrib import graph_runtime
import keras
import numpy as np

from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.nasnet import NASNetMobile

USE_GPU = True # True: GPU, False: CPU

model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
#model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
#model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

img = Image.open('cat.png').resize((224, 224))
#plt.imshow(img)
#plt.show()

# input preprocess
data = np.array(img)[np.newaxis, :].astype('float32')
data = preprocess_input(data).transpose([0, 3, 1, 2]) # NHWC -> NCHW
print('data', data.shape)

# convert the keras model(NHWC layout) to NNVM format(NCHW layout).
sym, params = nnvm.frontend.from_keras(model)

# compile the model
if USE_GPU:
    target = 'opencl'
else:
    target = 'llvm'
shape_dict = {'data': data.shape}
with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)

# set inputs
m.set_input('data', tvm.nd.array(data.astype('float32')))
m.set_input(**params)
# execute
m.run()
# get outputs
out_shape = (1000,)
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, 'float32')).asnumpy()
top1_tvm = np.argmax(tvm_out)

with open('imagenet1000_clsid_to_human.txt') as f:
    synset = eval(f.read())
print('NNVM top-1 id: {}, class name: {}'.format(top1_tvm, synset[top1_tvm]))

# confirm correctness with keras output
keras_out = model.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print('Keras top-1 id: {}, class name: {}'.format(top1_keras, synset[top1_keras]))

# Benchmark
num_iter = 100
ftimer = m.module.time_evaluator("run", ctx, num_iter)
prof_res = ftimer()
print(prof_res)

x = data.transpose([0, 2, 3, 1])
start = time()
for i in range(0, num_iter):
    model.predict(x)
end = time()
print((end-start)/num_iter)
