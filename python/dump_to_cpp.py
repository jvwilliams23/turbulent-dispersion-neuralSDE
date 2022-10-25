"""
convert Keras weights and architecture to files readable by cpp script
"""

import numpy as np
np.random.seed(1337)
import tensorflow as tf
from keras.models import Sequential, model_from_json
import json
import argparse

from keras.layers import (
    Dense,
    Conv1D, Conv2D,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Input,
    Activation,
    MaxPooling2D,
    LSTM,
    Embedding,
    BatchNormalization,
    )

LAYERS = (
    Dense,
    Conv1D, Conv2D,
    Input,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Activation,
    MaxPooling2D,
    LSTM,
    Embedding,
    BatchNormalization,
)

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='This is a simple script to dump Keras model into simple format suitable for porting into pure C++ model')

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-o', '--output', help="Output file name", required=True)
parser.add_argument('-v', '--verbose', help="Verbose", required=False)
args = parser.parse_args()

print ('Read architecture from', args.architecture)
print ('Read weights from', args.weights)
print ('Writing to', args.output)

arch = open(args.architecture).read()
model = model_from_json(arch, custom_objects={'leaky_relu': tf.nn.leaky_relu})
model.load_weights(args.weights)
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
arch = json.loads(arch)

with open(args.output, 'w') as fout:
    # initialise first row in file to say 0 layers. Will be overwritten at end
    num_layers = 0
    fout.write('layers ' + str(num_layers) + '\n')

    layers = []
    if args.verbose:
        print(arch["config"])
    # if input layer is skipped, index to write must have different ordering
    ind_write = 0
    for ind, (l, l_model) in enumerate(zip(arch["config"]["layers"], model.layers)):
        if args.verbose:
            print (ind, l)
        if type(l_model) ==  tf.keras.layers.InputLayer:
            print(f"Not including input layer in {args.output}")
            continue
        fout.write('layer ' + str(ind_write) + ' ' + l['class_name'] + '\n')

        if args.verbose:
            print (str(ind_write), l['class_name'])
        layers += [l['class_name']]
        if l['class_name'] == 'Convolution2D':
            W = model.layers[ind].get_weights()[0]
            if args.verbose:
                print (W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    for k in range(W.shape[2]):
                        fout.write(str(W[i,j,k]) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')

        if l['class_name'] == 'Activation':
            fout.write(l['config']['activation'] + '\n')
        if l['class_name'] == 'MaxPooling2D':
            fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
        if l['class_name'] == 'Dropout':
            fout.write(str(l['config']['rate']) + '\n')
        if l['class_name'] == 'Dense':
            W = model.layers[ind].get_weights()[0]
            if args.verbose:
                 print (W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')


            for w in W:
                fout.write(str(w) + '\n')
            if args.verbose:
                print(model.layers[ind].get_weights())
            # if only weights (i.e. usebias=False), then num_params=1
            num_params = len(model.layers[ind].get_weights())
            if num_params != 1:
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
            else:
                out_shape = model.layers[ind].get_weights()[0].shape[1]
                fout.write(str(np.zeros(out_shape)) + '\n')
            # if using dense layer, activation function can be included this way
            # skip linear activation as it is the default i.e. a(x) = x
            if l['config']['activation'] != 'linear':
              ind_write += 1
              fout.write('layer ' + str(ind_write) + ' Activation\n')
              fout.write(l['config']['activation'] + '\n')
        ind_write += 1

# write number of lines using ind_write value
with open(args.output, "r") as f:
  lines = f.readlines()
lines[0] = 'layers ' + str(ind_write) + '\n'


with open(args.output, 'w') as fout:
  fout.writelines(lines)
