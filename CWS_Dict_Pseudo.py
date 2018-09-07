#coding=utf-8
import tensorflow as tf
import numpy as np
import collections
import os
class CWSDictPseudo(object):
    def __init__(self, config):
        self.vocab_size = config.vocab_size#0 MASK，1 UNK
        self.emb_dim = config.emb_dim
        self.output_dim = config.output_dim#label class num
        self.use_pretrain_emb = config.use_pretrain_emb
        self.pretrain_emb = config.pretrain_emb
        self.emb_trainable = config.emb_trainable
        self.batch_size = config.batch_size
        self.reg = config.reg
        self.maxlen = config.maxlen
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.clip_norm = config.clip_norm
        self.dropSet = config.dropSet
        self.config = config

        self.add_placeholders()

        self.add_embedding()

        self.ch_embs = self.get_embedding(self.input_sens)
        self.ch_embs = self.ch_embs * self.mask[:,:,None]

        self.add_model_variables()

        self.calc_batch_loss()
    def add_placeholders(self):
        self.input_sens = tf.placeholder(tf.int32, [None, None], name='input_sens')
        self.seqlen = tf.placeholder(tf.int32, [None], name='seqlen')
        self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
        self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_s = tf.placeholder(tf.int32, shape=[], name='batch_s')
        self.loss_weight = tf.placeholder(tf.float32, shape=[None], name='loss_weight')
    def add_embedding(self):
        with tf.variable_scope('Embed', regularizer=None):
            if self.use_pretrain_emb:
                char_embedding = tf.get_variable('char_embedding', shape=[self.vocab_size, self.emb_dim],
                                            initializer=tf.constant_initializer(self.pretrain_emb),
                                            trainable=self.emb_trainable, regularizer=None)
            else:
                char_embedding = tf.get_variable('char_embedding', shape=[self.vocab_size, self.emb_dim],
                                            initializer=tf.random_uniform_initializer(-1,1),
                                            trainable=self.emb_trainable, regularizer=None)
    def get_embedding(self, input_sens):
        with tf.variable_scope('Embed', regularizer=None, reuse=True):
            char_embedding = tf.get_variable('char_embedding', shape=[self.vocab_size, self.emb_dim])
            ch_embs = tf.nn.embedding_lookup(char_embedding, input_sens)
            return ch_embs
    def calc_wt_init(self, fan_in=300):#for xavier_initializer
        eps=1.0/np.sqrt(fan_in)
        return eps
    def add_model_variables(self):
        self.densedim = self.num_filters * len(self.filter_sizes)
        #elf.densedim = self.hidden_dim * 2
        with tf.variable_scope('Encode', regularizer=None):
            for ii in range(len(self.filter_sizes)):
                hsize = self.emb_dim
                conv_W = tf.get_variable(('conv_W%d' % ii), [self.filter_sizes[ii], hsize, 1, self.num_filters],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv_b = tf.get_variable(('conv_b%d' % ii), [self.num_filters],initializer=
                                             tf.constant_initializer(0.), regularizer=tf.contrib.layers.l2_regularizer(0.0))
        with tf.variable_scope('Decode', regularizer=None):
            out_W = tf.get_variable('out_W', [self.densedim, self.output_dim],
                                initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.densedim),self.calc_wt_init(self.densedim)))
            out_b = tf.get_variable('out_b', [self.output_dim],
                                   initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
    def calc_batch_loss(self):
        if 'emb' in self.dropSet and self.config.keep_prob < 1:
            self.ch_embs = tf.nn.dropout(self.ch_embs, self.keep_prob)

        cnn_outputs = []
        with tf.variable_scope('Encode', reuse=True):
            cnn_input = self.ch_embs
            encode_h = tf.expand_dims(cnn_input, -1)
            for ii in range(len(self.filter_sizes)):
                filter_size = self.filter_sizes[ii]
                hsize = self.emb_dim
                filter_shape = [filter_size, hsize, 1, self.num_filters]
                conv_W = tf.get_variable(('conv_W%d' % ii), filter_shape)
                conv_b = tf.get_variable(('conv_b%d' % ii), [self.num_filters])
                conv_in = tf.concat([tf.zeros([self.batch_s, filter_size//2, hsize, 1]), encode_h, tf.zeros([self.batch_s, filter_size-filter_size//2-1, hsize, 1])], axis=1)
                conv_out = tf.nn.conv2d(conv_in, conv_W, strides=[1, 1, 1, 1], padding='VALID')
                conv_out = tf.nn.bias_add(conv_out, conv_b)
                conv_out = tf.nn.relu(conv_out)
                #encode_h = tf.reshape(conv_out, [self.batch_s, tf.shape(self.ch_embs)[1], self.num_filters, 1])
                conv_out = tf.reshape(conv_out, [self.batch_s, tf.shape(self.ch_embs)[1], self.num_filters])
                cnn_outputs.append(conv_out)
        cnn_outputs = tf.concat(cnn_outputs, axis=2)
        print(tf.shape(cnn_outputs))

        if 'cnn' in self.dropSet and self.config.keep_prob < 1:
            cnn_outputs = tf.nn.dropout(cnn_outputs, self.keep_prob)

        h2dense = tf.reshape(cnn_outputs, [-1, self.densedim])
        with tf.variable_scope("Decode",reuse=True):
            out_W = tf.get_variable('out_W', [self.densedim, self.output_dim])
            out_b = tf.get_variable('out_b', [self.output_dim])
            self.unary_scores = tf.matmul(h2dense, out_W) + out_b
            self.unary_scores = tf.reshape(self.unary_scores, [self.batch_s, -1, self.output_dim])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.labels, self.seqlen)
        self.loss = tf.reduce_mean(-self.log_likelihood*self.loss_weight)
        #self.loss = tf.reduce_mean(tf.nn.relu(self.train_pre_score + self.label_pre_delta*self.ita - self.label_scores))
        self.train_step = tf.train.RMSPropOptimizer(self.config.lr).minimize(self.loss)
        #self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
        # self.optimizer = tf.train.AdamOptimizer(self.config.lr)
        # gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        # self.train_step = self.optimizer.apply_gradients(zip(gradients, variables))

    def train(self,
                trainsens,
                trainlabel,
                trainweight,
                validsens,
                validlabel,
                testsens,
                testlabel,
                sess,
                dispFreq=10,
                validFreq=500,
                savepath='./ckpt/model'):
        update = 0
        printloss = 0
        best_valid_loss = 1000
        best_valid_epoch = 0
        saver = tf.train.Saver()
        #saver.restore(sess, '%s_best.ckpt' % savepath)
        for i in range(self.config.max_epochs):
            r = np.random.permutation(len(trainsens))
            trainsens = [trainsens[ii] for ii in r]
            trainlabel = [trainlabel[ii] for ii in r]
            trainweight = [trainweight[ii] for ii in r]
            for ii in range(0, len(trainsens), self.batch_size):
                endidx = min(ii+self.batch_size, len(trainsens))
                if endidx <= ii:
                    break
                xx, ll, mm, seql = prepare_data(trainsens[ii:endidx], trainlabel[ii:endidx])
                lw = np.asarray(trainweight[ii:endidx])
                batch_ss = xx.shape[0]
                feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: self.config.keep_prob, self.batch_s: batch_ss, self.loss_weight: lw}
                result = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
                update += 1
                printloss += result[0]
                if update % dispFreq == 0:
                    print("Epoch:\t%d\tUPdate:\t%d\tloss:\t%f" % (i, update, printloss/dispFreq))
                    printloss = 0
            #valid
            print('valid begin!')
            print('valid:')
            validloss = self.evaluate(validsens, validlabel, sess)
            print('test begin!')
            print('test:')
            self.evaluate(testsens, testlabel, sess)
            if validloss < best_valid_loss:
                print('save the best model!')
                best_valid_loss = validloss
                best_valid_epoch = i
                saver.save(sess, '%s_best.ckpt' % savepath)

            if i - best_valid_epoch >= self.config.early_stopping:
                print('early stop!')
                break
        saver.save(sess,'%s_final.ckpt' % savepath)
        #saver.restore(sess, '%s_best.ckpt' % savepath)

    def evaluate(self, validsens, validlabel, sess):
        predicts = []
        validloss = []
        for ii in range(0, len(validsens), self.batch_size):
            endidx = min(ii+self.batch_size, len(validsens))
            if endidx <= ii:
                break
            xx, ll, mm, seql = prepare_data(validsens[ii:endidx], validlabel[ii:endidx])
            lw = np.asarray([1. for jj in range(endidx-ii)])
            batch_ss = xx.shape[0]
            feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: 1, self.batch_s: batch_ss, self.loss_weight: lw}
            loss, unary_scores, transition_params = sess.run([self.loss, self.unary_scores, self.transition_params], feed_dict=feed_dict)
            pp = []
            for us_, sl_ in zip(unary_scores, seql):
                us_ = us_[:sl_]
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(us_, transition_params)
                pp.append(viterbi_seq)
            predicts.extend(pp)
            validloss.append(loss)
        validloss = np.asarray(validloss)
        validloss = np.mean(validloss)
        accuracy, recall, precision, fscore = cal4metrics(predicts, validlabel)
        print("accuracy: %s\trecall: %s\nprecision: %s\tfscore: %s\n" % (accuracy, recall, precision, fscore))
        print('loss:\t%f' % validloss)
        print(transition_params)
        return validloss

    def giveTestResult(self, testsens, testlabel, sess, ofilepath='./result/result.txt'):
        predicts = []
        print('test begin!')
        for ii in range(0, len(testsens), self.batch_size):
            endidx = min(ii+self.batch_size, len(testsens))
            if endidx <= ii:
                break
            xx, ll, mm, seql = prepare_data(testsens[ii:endidx], testlabel[ii:endidx])
            lw = np.asarray([1. for jj in range(endidx-ii)])
            batch_ss = xx.shape[0]
            feed_dict = {self.input_sens: xx, self.mask: mm, self.seqlen: seql, self.labels: ll, self.keep_prob: 1, self.batch_s: batch_ss, self.loss_weight: lw}
            unary_scores, transition_params = sess.run([self.unary_scores, self.transition_params], feed_dict=feed_dict)
            pp = []
            for us_, sl_ in zip(unary_scores, seql):
                us_ = us_[:sl_]
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(us_, transition_params)
                pp.append(viterbi_seq)
            predicts.extend(pp)
        accuracy, recall, precision, fscore = cal4metrics(predicts, testlabel)
        print("test:\naccuracy: %s\trecall: %s\nprecision: %s\tfscore: %s\n" % (accuracy, recall, precision, fscore))
        with open(ofilepath, 'w') as f:
            for ii in range(len(testsens)):
                f.write('%d' % predicts[ii][0])
                for jj in range(1, len(testsens[ii])):
                    f.write(' %d' % predicts[ii][jj])
                f.write('\n')
        return predicts

def cal_metrics(predicts, labels):
    accuracy = 0.
    num = 0
    for ii in range(len(predicts)):
        for jj in range(len(predicts[ii])):
            if predicts[ii][jj] == labels[ii][jj]:
                accuracy += 1
            num += 1
    accuracy = accuracy / num
    return accuracy

def cal4metrics(predicts, labels):
    accuracy = cal_metrics(predicts, labels)
    predicts_bi = []
    labels_bi = []
    for ii in range(len(predicts)):
        pp = []
        ll = []
        for jj in range(len(predicts[ii])):
            if predicts[ii][jj] == 0 or predicts[ii][jj] == 3:
                pp.append(1)
            else:
                pp.append(0)
            if labels[ii][jj] == 0 or labels[ii][jj] == 3:
                ll.append(1)
            else:
                ll.append(0)
        predicts_bi.append(pp)
        labels_bi.append(ll)

    TP = 0.0
    FN = 0.0
    FP = 0.0
    for ii in range(len(labels_bi)):
        jj = 0
        while jj < len(labels_bi[ii]):
            if labels_bi[ii][jj] == 1:
                kk = jj + 1
                while kk < len(labels_bi[ii]) and labels_bi[ii][kk] == 0:
                    kk += 1
                temp1 = labels_bi[ii][jj:kk]
                temp2 = predicts_bi[ii][jj:kk]
                if all([temp1[zz]==temp2[zz] for zz in range(len(temp1))]) and (kk==len(labels_bi[ii]) or predicts_bi[ii][kk] != 0):
                    TP += 1
                else:
                    FN += 1
                jj = kk
            else:
                jj += 1
    for ii in range(len(predicts_bi)):
        jj = 0
        while jj < len(predicts_bi[ii]):
            if predicts_bi[ii][jj] == 1:
                kk = jj + 1
                while kk < len(predicts_bi[ii]) and predicts_bi[ii][kk] == 0:
                    kk += 1
                temp1 = labels_bi[ii][jj:kk]
                temp2 = predicts_bi[ii][jj:kk]
                #if any([temp1[zz]!=temp2[zz] for zz in range(len(temp2))]) or temp1[0] != 2:
                if any([temp1[zz]!=temp2[zz] for zz in range(len(temp2))]) or (kk < len(predicts_bi[ii]) and labels_bi[ii][kk] == 0):
                    FP += 1
                jj = kk
            else:
                jj += 1
    if TP == 0:
        recall = 0
        precise = 0
    else:
        recall = TP/(TP+FN)
        precise = TP/(TP+FP)
    if recall == 0 and precise == 0:
        fscore = 0
    else:
        fscore = 2*precise*recall/(precise+recall)

    return accuracy, recall, precise, fscore

def prepare_data(seqs_x, seqs_l, classesnum=4, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_l = [len(s) for s in seqs_l]

    if maxlen is not None:
        seqs_x = [s if len(s) <= maxlen else s[:maxlen] for s in seqs_x]
        seqs_l = [s if len(s) <= maxlen else s[:maxlen] for s in seqs_l]

        lengths_x = [len(s) for s in seqs_x]
        lengths_l = [len(s) for s in seqs_l]

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    maxlen_l = np.max(lengths_l)
    assert maxlen_x == maxlen_l

    x = np.zeros((n_samples, maxlen_x)).astype('int32')
    #y = np.zeros((n_samples, maxlen_l, classesnum)).astype('float32')
    y = np.zeros((n_samples, maxlen_l)).astype('int32')
    mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_l] in enumerate(zip(seqs_x, seqs_l)):
        x[idx, :lengths_x[idx]] = s_x
        mask[idx, :lengths_x[idx]] = 1.
        y[idx, :lengths_x[idx]] = s_l

    lengths_x = np.asarray(lengths_x).astype('int32')
    return x, y, mask, lengths_x


def gettraindata(filepath='./data/msr_training.utf8'):
    orisens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            orisens.append(line)
    print("ori sample num: %d" % len(orisens))
    labels = []
    char_sens = []
    word_sens = []
    for sen in orisens:
        words = sen.strip().split()
        if len(words) == 0:
            continue
        chars = []
        label = []
        for word in words:
            chars.extend([char for char in word])
            if len(word) == 1:
                label.append(3)
            else:
                label.append(0)
                for ii in range(len(word)-2):
                    label.append(1)
                label.append(2)
        assert len(label) == len(chars), sen
        labels.append(label)
        char_sens.append(chars)
        word_sens.append(words)
    print("%d samples after preprocess" % len(char_sens))

    return char_sens, labels

def gettestdata(filepath='./data/msr_test_gold.utf8'):
    orisens = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            orisens.append(line)
    print("test ori sample num: %d" % len(orisens))
    labels = []
    char_sens = []
    for sen in orisens:
        words = sen.strip().split()
        if len(words) == 0:
            continue
        chars = []
        label = []
        for word in words:
            chars.extend([char for char in word])
            if len(word) == 1:
                label.append(3)
            else:
                label.append(0)
                for ii in range(len(word)-2):
                    label.append(1)
                label.append(2)
        assert len(label) == len(chars), sen
        labels.append(label)
        char_sens.append(chars)
    print("%d test samples after preprocess" % len(char_sens))
    return char_sens, labels

def getDict(orisens, word_dictpath='./data/cidian.txt', min_count=0):
    chars = []
    for sen in orisens:
        chars.extend(sen)
    count = []
    count.extend(collections.Counter(chars).most_common())
    dictionary = dict()
    dictionary['<f/s/f>'] = 0#mask
    dictionary['<UNK>'] = 1#UNK
    for word, c in count:
        if c < min_count:
            break
        dictionary[word] = len(dictionary)
    r_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print('vocab size: %d' % len(dictionary))
    return dictionary

def getsensid(orisens, dictionary):
    senslen = [len(s) for s in orisens]
    maxlen = max(senslen)
    sensid = []
    for sen in orisens:
        temp = [dictionary[char] if char in dictionary else 1 for char in sen]
        sensid.append(temp)
    return sensid, maxlen



def data_split(orisens, oricwslabels, valid_bili=0.1):
    #r = np.random.permutation(len(orisens))
    #orisens = [orisens[ii] for ii in r]
    #orilabels = [orilabels[ii] for ii in r]

    split_at = int((1-valid_bili) * len(orisens))
    validsens = orisens[split_at:]
    validcwslabels = oricwslabels[split_at:]

    trainsens = orisens[:split_at]
    traincwslabels = oricwslabels[:split_at]

    return trainsens, traincwslabels, validsens, validcwslabels

def data_random_select(orisens, oricwslabels, train_bili=0.25, rpath='./rpath/rtrain0.npy'):
    if os.path.exists(rpath):
        r = np.load(rpath)
    else:
        r = np.random.permutation(len(orisens))
        np.save(rpath, r)
    orisens = [orisens[ii] for ii in r]
    oricwslabels = [oricwslabels[ii] for ii in r]

    split_at = int(train_bili * len(orisens))
    trainsens = orisens[:split_at]
    traincwslabels = oricwslabels[:split_at]

    return trainsens, traincwslabels

def output_CWS_result(predicts, testsens, r_dictionary, ofilepath='./result/segresult.txt'):
    seg_sens = []
    for ii in range(len(testsens)):
        sen = []
        begin = 0
        end = 1
        while end < len(testsens[ii]):
            if predicts[ii][end] == 0 or predicts[ii][end] == 3:
                sen.append(''.join([r_dictionary[c] for c in testsens[ii][begin:end]]))
                begin = end
            end += 1
        sen.append(''.join([r_dictionary[c] for c in testsens[ii][begin:end]]))
        seg_sens.append(' '.join(sen))

    with open(ofilepath, 'w', encoding='utf-8') as f:
        for sen in seg_sens:
            f.write(sen)
            f.write('\n')

def get_pretrain_emb(dictionary, pretrain_file='./data/charVecForCWS.txt', emb_size=200):
    vocab_size = len(dictionary)
    pre_emb = np.random.random([vocab_size, emb_size]) * 2 - 1
    with open(pretrain_file, 'r', encoding='utf-8') as f:
        temp = f.readline().strip().split()
        wordnum = int(temp[0])
        veclen = int(temp[1])
        assert emb_size == veclen
        pre_num = 0
        for line in f:
            temp = line.strip().split()
            if temp[0] in dictionary:
                index = dictionary[temp[0]]
                try:
                    prevecs = np.asarray([float(num) for num in temp[1:]])
                    pre_emb[index, :] = prevecs
                except:
                    print(temp[0])
                pre_num += 1
        print('pretrain word num: %d' % pre_num)
    return pre_emb

def getworddict(word_dictpath='./data/cidian.txt'):
    word_dict = set()
    with open(word_dictpath, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 1:
                word_dict.add(line.strip())
    p = np.asarray([1. / len(word_dict) for ii in range(len(word_dict))])
    p = p / np.sum(p)
    p = list(p)
    return word_dict, p

def genewordcandata(num, dictionary):
    wordlist, p = getworddict()
    sens = []
    labels = []
    for ii in range(num):
        wordnum = 30 + np.random.randint(-5, 6)
        sen = []
        label = []
        for jj in range(wordnum):
            idx = np.random.randint(0, len(wordlist))
            #idx = np.random.choice(len(wordlist), 1, replace=True, p=p)[0]
            word = wordlist[idx]
            assert len(word) > 1, word
            chs = [dictionary[char] if char in dictionary else 1 for char in word]
            sen.extend(chs)
            label.append(0)
            for ii in range(len(word)-2):
                label.append(1)
            label.append(2)
        assert len(sen) == len(label)
        sens.append(sen)
        labels.append(label)
    return sens, labels


class Config(object):
    dataset = 'msra'
    vocab_size = None
    emb_dim = 200
    output_dim = 4


    use_pretrain_emb = True
    #use_pretrain_emb = False
    pretrain_emb = None
    emb_trainable = True
    #emb_trainable = False

    filter_sizes = [3]
    num_filters = 400


    max_epochs = 100
    reduce_lr = 3
    early_stopping = 3
    keep_prob = 0.5
    lr = 0.001
    emb_lr = 0.01
    reg = 0.0001
    batch_size = 64
    maxlen = 70
    clip_norm = 5.0
    dropSet = set(['emb', 'cnn'])
    train_bili = 1


def main(saveid=0, dporate=0.5, dataset='msra', tbili=0.01, flb=0.05, fbei=2):
    print("save id %d" % saveid)
    print("dropout rate: %f" % dporate)
    config = Config()
    config.dataset = dataset
    config.train_bili = tbili

    filepath = './data/%s_training.utf8' % config.dataset
    orisens, cwslabels = gettraindata(filepath=filepath)
    filepath = './data/%s_test_gold.utf8' % config.dataset
    oritestsens, testcwslabels = gettestdata(filepath=filepath)
    saveprefix = config.dataset


    dictionary = getDict(orisens)
    r_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    valid_bili = 0.1
    oritrainsens, traincwslabels, orivalidsens, validcwslabels = data_split(orisens, cwslabels, valid_bili=valid_bili)

    if config.train_bili < 1:
        rid = saveid % 5
        rpath = './rpath/%s_rtrain%d.npy' % (config.dataset, rid)
        train_bili = config.train_bili
        oritrainsens, traincwslabels = data_random_select(oritrainsens, traincwslabels, train_bili=train_bili, rpath=rpath)

    trainsens, maxlen = getsensid(oritrainsens, dictionary)
    validsens, validmaxlen = getsensid(orivalidsens, dictionary)
    testsens, testmaxlen = getsensid(oritestsens, dictionary)
    maxlen = max(maxlen, validmaxlen, testmaxlen)
    config.maxlen = maxlen
    config.vocab_size = len(dictionary)
    if config.use_pretrain_emb:
        pretrain_file='./data/charVecForCWS.txt'
        config.emb_dim = 200
        pre_emb = get_pretrain_emb(dictionary, pretrain_file=pretrain_file, emb_size=config.emb_dim)
        config.pretrain_emb = pre_emb
        assert pre_emb.shape[0] == len(dictionary)

    wcdsens, wcdlabels = genewordcandata(len(trainsens)*fbei, dictionary)
    trainweight = [1. for ii in range(len(trainsens))] + [flb for ii in range(len(wcdsens))]
    trainsens = trainsens + wcdsens
    traincwslabels = traincwslabels + wcdlabels

    print("%d train samples" % len(trainsens))
    print("%d valid samples" % len(validsens))
    print("%d test samples" % len(testsens))

    config.keep_prob = dporate
    savesuffix = '%d' % (int(dporate*10))

    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    model = CWSDictPseudo(config)
    init = tf.initialize_all_variables()
    with tf.Session(config=tfconfig) as sess:
        sess.run(init)
        savepath = './CWSDictPseudo/ckpt/%scwspseudo_%s_%d' % (saveprefix, savesuffix, saveid)
        model.train(trainsens, traincwslabels, trainweight, validsens, validcwslabels, testsens, testcwslabels, sess, savepath=savepath)
        ofilepath = './CWSDictPseudo/result/%scwspseudo_%s_%d.txt' % (saveprefix, savesuffix, saveid)
        predicts = model.giveTestResult(testsens, testcwslabels, sess, ofilepath=ofilepath)
        ofilepath = './CWSDictPseudo/result/result_%scwspseudo_%s_%d.txt' % (saveprefix, savesuffix, saveid)
        output_CWS_result(predicts, testsens, r_dictionary, ofilepath=ofilepath)


if __name__=='__main__':
    main(0, dporate=0.7, dataset='msra', tbili=0.1, flb=0.01, fbei=1)
