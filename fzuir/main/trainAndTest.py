# -*- coding: UTF-8 -*-

import sys
from LSTMModel_TrainTest import LSTMModel
from Dataset import *
import logging

classes = sys.argv[1]

root_dir = "../../sources/BoSongModel/"
logging.basicConfig(filename=root_dir+"TrainAndTest.log", format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

voc = Wordlist(root_dir+'word2vec.vocabulary')
logging.info('words loaded')

trainset = Dataset(root_dir+'train.txt', emb=voc, maxbatch=16)
logging.info('trainset loaded')

test_path = root_dir+'new_test_clusterid.txt'

devset = Dataset(root_dir+'test.txt', emb=voc, maxbatch=16, test_path=test_path)
logging.info('devset loaded')
logging.info('===========data loaded.================')

em_path = root_dir+'word2vec.vector'

model = LSTMModel(voc.size, trainset, devset, em_path, classes, None)

logging.info('****************************************************************************')
bigest_f1 = 0.0
db_bigest_f1 = 0.0
max_auc = 0.0

for i in xrange(1, 512):
	logging.info('test%s' % i)

	# 我们这里使用预测为1时的F1值作为优化参数
	# 每次训练都要返回最后一层的输出，以便写入最好训练集特征
	train_features = model.train(50)
	new_f1, db_new_f1, max_auc = model.test(root_dir=root_dir, old_f1=bigest_f1, db_old_f1=db_bigest_f1,
											max_auc=max_auc)

	if new_f1 > bigest_f1:
		matlab_write = file(root_dir+'matlab_train_feature.txt', mode='w+')
		# id_feature = model.trainset.id_feature
		train_label = model.trainset.label
		bigest_f1 = new_f1
		db_bigest_f1 = db_new_f1
		model.save(root_dir)
		i = -1
		j = -1
		for feature in train_features:
			j += 1
			for fea in feature:
				i += 1
				num = 8
				fea_str = ""
				# 遍历每个值，添加到训练集后面，并加上序号
				for f in fea:
					num += 1
					# fea_str += " " + str(num) + ":" + str(f)
					fea_str += " " + str(f)
				# matlab_write.write(str(train_label[j][i%model.trainset.batch])+ " " + str(id_feature[i]).replace("\t", " ") + fea_str + "\n")
				matlab_write.write(str(train_label[j][i % model.trainset.batch]) + " " + fea_str + "\n")
		matlab_write.close()
	logging.info('****************************************************************************\n\n')

logging.info('\n=========bestmodel saved!======End!!!=====')

