# -*- coding: UTF-8 -*-

import numpy
import copy
import theano
import random
import sys
import logging
sys.path.append("/home/fzuir/ymg/NSCTopic/NSC/fzuir/util")
sys.path.append("E:/DLTopicWorkspace/NSCTopic/NSC/fzuir/util")

theano.config.floatX = 'float32'

def genBatch(data):
    m =0 
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence)>m:
                m = len(sentence)
        for i in xrange(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = map(lambda doc: numpy.asarray(map(lambda sentence : sentence + [-1]*(m - len(sentence)), doc),
                                        dtype = numpy.int32).T, data)  #[-1]是加在最前面
    tmp = reduce(lambda doc,docs : numpy.concatenate((doc,docs),axis = 1),tmp)
    return tmp 
            
def genLenBatch(lengths,maxsentencenum):
    lengths = map(lambda length : numpy.asarray(length + [1.0]*(maxsentencenum-len(length)),
                                                dtype = numpy.float32)+numpy.float32(1e-4), lengths)
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = map(lambda x : map(lambda y : [1.0 ,0.0][y == -1],x), mask)
    mask = numpy.asarray(mask,dtype=numpy.float32)
    return mask

def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray(map(lambda num : [1.0]*num + [0.0]*(maxnum - num),sentencenum), dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb,maxbatch = 32, maxword = 500, test_path=None):
        lines = map(lambda x: x.split('\t\t'), open(filename).readlines())           
        label = numpy.asarray(map(lambda x: int(x[1]), lines), dtype = numpy.int32)
        self.batch = maxbatch
        self.id_feature = None
        # 得到话题id和特征
        self.id_feature = map(lambda x: str(x[2]).strip(), lines)
        self.topic_ids = []
        self.predict_ids = []
        time = len(label) * [1]
        if test_path is not None:
            import database_util
            time = []
            conn = database_util.ConnMysql("59.77.233.198", 3306, "root", "mysql_fzu_118", "HotTopic")
            conn.connectMysql()
            child_ids = conn.queryData("select b.childTopicid from topicmanual a,topicmanual_topic b "
                                       "where a.topicStatus=1 and a.id=b.topicid")
            if child_ids is None:
                logging.error("有热点话题在数据库中查询不到")
                conn.closeMysql()
                sys.exit(0)
            for id in child_ids:
                self.topic_ids.append(str(id[0]))
            conn.closeMysql()
            predict_read = file(test_path, 'r')
            line = predict_read.readline().strip()
            while line:
                self.predict_ids.append(line.split('\t')[1])
                time.append(line.split("\t")[0])
                line = predict_read.readline().strip()
            predict_read.close()

        docs = map(lambda x: x[0][0:len(x[0])], lines)
        docs = map(lambda x: x.split('<sssss>'), docs) 
        docs = map(lambda doc: map(lambda sentence: sentence.split(' '), doc), docs)
        docs = map(lambda doc: map(lambda sentence: filter(lambda wordid: wordid !=-1,
                                                           map(lambda word: emb.getID(word), sentence)), doc), docs)
        tmp = zip(docs, label, time)
        #random.shuffle(tmp)
        tmp.sort(lambda x, y: len(y[0]) - len(x[0]))  
        docs, label, time = zip(*tmp)

        sentencenum = map(lambda x : len(x),docs)
        length = map(lambda doc : map(lambda sentence : len(sentence), doc), docs)
        self.epoch = len(docs) / maxbatch                                      
        if len(docs) % maxbatch != 0:
            self.epoch += 1
        
        self.docs = []
        self.label = []
        self.length = []
        self.sentencenum = []
        self.wordmask = []
        self.sentencemask = []
        self.maxsentencenum = []

        for i in xrange(self.epoch):
            self.maxsentencenum.append(sentencenum[i*maxbatch])
            self.length.append(genLenBatch(length[i*maxbatch:(i+1)*maxbatch],sentencenum[i*maxbatch])) 
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))
            self.sentencenum.append(numpy.asarray(sentencenum[i*maxbatch:(i+1)*maxbatch],dtype = numpy.float32)+numpy.float32(1e-4))
            self.wordmask.append(genwordmask(docsbatch))
            self.sentencemask.append(gensentencemask(sentencenum[i*maxbatch:(i+1)*maxbatch]))

        if test_path is not None:
            test_label_write = file(test_path[:-18]+"testSorted_label.txt", 'w+')
            i = -1
            for labels in self.label:
                for label in labels:
                    i += 1
                    test_label_write.write(time[i]+"\t"+str(label)+"\n")
            test_label_write.close()

class Wordlist(object):
    def __init__(self, filename, max_words_num = 50000):
        lines = map(lambda x: x.split(" ")[0], open(filename).readlines()[:max_words_num])
        self.size = len(list(lines))
        voc = [(item[0], item[1]) for item in zip(lines, xrange(self.size))]
        self.voc = dict(voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1
