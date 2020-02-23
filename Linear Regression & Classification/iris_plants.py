# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
# Modified to load the iris plants dataset from scikit-learn for classification.

import numpy as np
from sklearn.datasets import load_iris

from HelperClass.NeuralNet import *

if __name__ == '__main__':
    # data
    num_category = 3
    reader = DataReader(load_iris)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base = 1)

    # net
    num_input = 4
    hp = HyperParameters(num_input, num_category, eta = 0.1, max_epoch = 100, batch_size = 10, eps = 1e-3, \
        net_type = NetType.MultipleClassifier)
    net = NeuralNet(hp)
    net.train(reader, checkpoint = 1)

    # inference
    index = int(input("The index to test in range(150): "))
    x = reader.XRaw[index].reshape(1, 4)
    x_new = reader.NormalizePredicateData(x)
    output = net.inference(x_new)
    r = np.argmax(output, axis = 1) + 1
    print("output =", output)
    print("r =", r[0])
    print("original =", reader.YRaw[index])
