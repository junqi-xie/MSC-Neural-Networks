# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
# Modified to load the boston house-prices dataset from scikit-learn for regression.

import numpy as np
from sklearn.datasets import load_boston

from HelperClass.NeuralNet import *

if __name__ == '__main__':
    # data
    reader = DataReader(load_boston)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()

    # net
    num_input = 13
    hp = HyperParameters(num_input, 1, eta = 0.01, max_epoch = 200, batch_size = 10, eps = 1e-5, \
        net_type = NetType.Fitting)

    net = NeuralNet(hp)
    net.train(reader, checkpoint = 0.1)

    # inference
    index = int(input("The index to test in range(506): "))
    x = reader.XRaw[index].reshape(1, 13)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("z =", z)
    Z_true = z * reader.Y_norm[0, 1] + reader.Y_norm[0, 0]
    print("Z_true =", Z_true)
    print("original =", reader.YRaw[index])
