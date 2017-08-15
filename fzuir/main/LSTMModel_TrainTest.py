# -*- coding: UTF-8 -*-

from EmbLayer import EmbLayer
from LSTMLayer import LSTMLayer
from HiddenLayer import HiddenLayer
from Update import AdaUpdates
from PoolLayer import *
from SentenceSortLayer import *
import theano
import theano.tensor as T
import numpy
import random
import sys
import logging

theano.config.floatX = 'float32'

class LSTMModel(object):
    def __init__(self, n_voc, trainset, testset, em_path, classes, prefix):
        if prefix is not None:
            prefix += '/'
        self.trainset = trainset
        self.testset = testset

        docs = T.imatrix()
        label = T.ivector()
        length = T.fvector()
        sentencenum = T.fvector()
        wordmask = T.fmatrix()
        sentencemask = T.fmatrix()
        maxsentencenum = T.iscalar()
        isTrain = T.iscalar()

        rng = numpy.random

        layers = []
        layers.append(EmbLayer(em_path, rng, docs, n_voc, 200, "emblayer", prefix))
        layers.append(LSTMLayer(rng, layers[-1].output, wordmask, 200, 200, 'wordlstmlayer', prefix)) 
        layers.append(MeanPoolLayer(layers[-1].output, length))
        layers.append(SentenceSortLayer(layers[-1].output,maxsentencenum))
        layers.append(LSTMLayer(rng, layers[-1].output, sentencemask, 200, 200, 'sentencelstmlayer', prefix))
        layers.append(MeanPoolLayer(layers[-1].output, sentencenum))
        layers.append(HiddenLayer(rng, layers[-1].output, 200, 200, 'fulllayer', prefix))
        # 保存这层的输出  是numpy.ndarray类型
        feature_vector = layers[-1].output
        layers.append(HiddenLayer(rng, layers[-1].output, 200, int(classes), 'softmaxlayer',
                                  prefix, activation=T.nnet.softmax))
        # softmax这层输出是 n行2维向量（2分类），表示每个句子属于每个分类的概率，之和为1
        predict_probability = layers[-1].output
        self.layers = layers
        
        cost = -T.mean(T.log(layers[-1].output)[T.arange(label.shape[0]), label], acc_dtype='float32')
        correct = T.sum(T.eq(T.argmax(layers[-1].output, axis=1), label), acc_dtype='int32')
        err = T.argmax(layers[-1].output, axis=1) - label
        mse = T.sum(err * err)
        # 返回预测的各个文本的类别（0和1）
        predict_label = T.argmax(layers[-1].output, axis=1)

        params = []
        for layer in layers:
            params += layer.params
        L2_rate = numpy.float32(1e-5)
        for param in params[1:]:
            cost += T.sum(L2_rate * (param * param), acc_dtype='float32')
        gparams = [T.grad(cost, param) for param in params]

        updates = AdaUpdates(params, gparams, 0.95, 1e-6)

        self.train_model = theano.function(
            inputs=[docs, label, length, sentencenum, wordmask, sentencemask, maxsentencenum],
            outputs=[cost, feature_vector],
            updates=updates,
            mode='FAST_RUN'
        )
        self.test_model = theano.function(
            inputs=[docs, label, length, sentencenum, wordmask, sentencemask, maxsentencenum],
            outputs=[correct, mse, predict_label, feature_vector, predict_probability],
            mode='FAST_RUN'
        )

    def train(self, iters):
        # self.trainset.epoch 表示窗口多少个
        # 随意选择iters个窗口来训练
        # lst = numpy.random.randint(self.trainset.epoch, size=iters)
        # print 'epoch=',self.trainset.epoch
        # print lst
        n = 0
        all_features = []
        # lst表示第几个窗口，每个窗口大小为16(最初源码为32)
        # self.trainset.epoch是每个窗口都训练
        for i in xrange(self.trainset.epoch):
            n += 1
            out = self.train_model(self.trainset.docs[i], self.trainset.label[i], self.trainset.length[i],
                                   self.trainset.sentencenum[i], self.trainset.wordmask[i],
                                   self.trainset.sentencemask[i], self.trainset.maxsentencenum[i])
            logging.info('%s cost:%s' % (n, out[0]))
            all_features.append(out[1])

        return all_features
        
    def test(self, root_dir, old_f1, db_old_f1, max_auc):
        cor = 0
        tot = 0
        mis = 0
        predict1 = 0
        corr_predict1 = 0
        db_predict1 = 0
        sum1 = 0

        db_all_label_outputs = []
        all_label_outputs = []
        all_feature_outputs = []
        predict_probability = []
        predict_ids = self.testset.predict_ids
        topic_ids = self.testset.topic_ids

        for i in xrange(self.testset.epoch):
            outputs = self.test_model(self.testset.docs[i], self.testset.label[i], self.testset.length[i],
                                  self.testset.sentencenum[i], self.testset.wordmask[i],
                                  self.testset.sentencemask[i], self.testset.maxsentencenum[i])
            cor += outputs[0]
            mis += outputs[1]
            tot += len(self.testset.label[i])

            # outputs[3]是numpy.ndarray类型,是全连接层输出的特征向量
            all_feature_outputs.append(outputs[3])

            # 得到标注中有多少个1
            sum1 += sum(self.testset.label[i])
            j = -1
            # print outputs[2]
            for label in outputs[2]:
                j += 1
                # 将预测的概率保存
                predict_probability.append(round(outputs[4][j][1], 4))
                all_label_outputs.append(label)
                if label == 1:
                    # 预测到的1的个数
                    predict1 += 1
                    # 表示预测1正确,而且该预测的话题id在人工标注中是新兴话题
                    if self.testset.label[i][j] == 1:
                        corr_predict1 += 1
                        if predict_ids[self.testset.batch * i + j] in topic_ids:
                            db_predict1 += 1
                            db_all_label_outputs.append(1)
                        else:
                            db_all_label_outputs.append(0)
                    else:
                        db_all_label_outputs.append(0)
                else:
                    db_all_label_outputs.append(0)

        # cor表示预测正确的个数（预测为1，实际也为1；预测为0，实际也为0）
        # print 'Accuracy:', float(cor) / float(tot), 'RMSE:', numpy.sqrt(float(mis) / float(tot))
        logging.info('预测到1的个数为：%s，正确的1个数且是真新兴话题为：%s，标注中1的个数为：%s' % (predict1, corr_predict1, sum1))
        precision = 0
        db_precision = 0
        recall = 0
        if predict1 > 0:
            precision = float(corr_predict1) / float(predict1)
            db_precision = float(db_predict1) / float(predict1)
        if sum1 > 0:
            recall = float(corr_predict1) / float(sum1)
        db_new_f1 = float(2 * db_precision * recall) / float(0.0000001 + db_precision + recall)
        new_f1 = float(2 * precision * recall) / float(0.0000001 + precision + recall)
        test_label = []
        for labels in self.testset.label:
            for label in labels:
                test_label.append(label)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(test_label, predict_probability, pos_label=1, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        if max_auc < roc_auc:
            max_auc = roc_auc
        logging.info("当前roc_auc=%s,最大roc_auc=%s\n" % (roc_auc, max_auc))

        # 这里直接修改f1为预测的准确1的个数，因为现在只需要预测正确就好了
        # new_f1 = corr_predict1
        logging.info('预测为1的准确率=%s,  召回率为=%s' % (precision, recall))
        logging.info('new_f1=%s,  bigest_f1=%s' % (new_f1, old_f1))
        logging.info('加入了数据库中  预测为1的准确率=%s,  召回率为=%s' % (db_precision, recall))
        logging.info('new_f1=%s,  bigest_f1=%s' % (db_new_f1, db_old_f1))

        # 如果新的F1较大，说明本次预测更优，需要保存
        if new_f1 > old_f1:
            matlab_write = file(root_dir + 'matlab_test_feature.txt', mode='w+')
            predict_write = file(root_dir + 'best_predict.txt', mode='w+')
            db_predict_write = file(root_dir + 'db_best_predict_label.txt', mode='w+')
            # id_feature = self.testset.id_feature
            for out1 in db_all_label_outputs:
                db_predict_write.write(str(out1) + "\n")
            db_predict_write.close()

            for j in xrange(len(all_label_outputs)):
                predict_write.write(str(all_label_outputs[j]) + "\t" + str(predict_probability[j]) + "\n")
            predict_write.close()
            i = -1
            for out2 in all_feature_outputs:
                j = -1
                for feature in out2:
                    num = 8
                    fea_str = ""
                    j += 1
                    i += 1
                    for fea in feature:
                        num += 1
                        # fea_str += " " + str(num) + ":" + str(fea)
                        fea_str += " " + str(fea)
                    # matlab_write.write("-1 "+str(id_feature[i]).replace("\t", " ") + fea_str + "\n")
                    matlab_write.write(fea_str + "\n")
            matlab_write.close()
        return new_f1, db_new_f1, max_auc

    def save(self, prefix):
        prefix += '/'
        for layer in self.layers:
            layer.save(prefix)
