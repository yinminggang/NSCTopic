# -*- coding: UTF-8 -*-

import theano
import theano.tensor as T
import numpy
import cPickle

class EmbLayer(object):
    def __init__(self, em_path, rng, inp, n_voc, dim, name, prefix=None):
        self.input = inp
        self.name = name

        if prefix is None:
            f = file(em_path, 'r')
            line = f.readline().strip()
            vectors = []
            while line:
                if len(line) < 5:
                    break
                str_split = str(line[1 + line.find(" ", 0, 50):]).strip().split(" ")
                float_split = [float(x) for x in str_split if x]
                vectors.append(numpy.array(float_split, dtype=numpy.float32))
                line = f.readline().strip()
            f.close()
            W = numpy.array(vectors, dtype=numpy.float32)
            W = theano.shared(value=W, name='E', borrow=True)
        else:
            f = file(prefix + name + '.save', 'rb')
            W = cPickle.load(f)
            f.close()
        self.W = W

        self.output = self.W[inp.flatten()].reshape((inp.shape[0], inp.shape[1], dim))
        self.params = [self.W]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
