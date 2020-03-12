#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell as GRU
import time
import os
import logging
from tqdm import trange
from utils.misc import prepare_input
from utils.model_helper import *
from utils.data_preprocessor import MAX_WORD_LEN

from scipy.integrate import quad
import numpy as np


class EpiReader:
    def __init__(self, n_layers, vocab_size, n_chars,
                 gru_size, embed_dim, train_emb, char_dim,
                 use_feat, gating_fn, save_attn=False, gama=0.04, lam=1, K=5, Nf=32):
        self.gru_size = gru_size
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.train_emb = train_emb
        self.char_dim = char_dim
        self.n_chars = n_chars
        self.use_feat = use_feat
        self.save_attn = save_attn
        self.gating_fn = gating_fn
        self.n_vocab = vocab_size
        self.use_chars = self.char_dim != 0
        self.gama = gama
        self.lam = lam
        self.K = K
        self.Nf = Nf
    def build_graph(self, grad_clip, embed_init):
        """
        define model variables
        """
        # word input
        self.doc = tf.placeholder(
            tf.int32, [None, None], name="doc")
        self.doc_sent = tf.placeholder(
            tf.int32, [None, None, None], name="doc")

        self.qry = tf.placeholder(
            tf.int32, [None, None], name="query")

        self.answer = tf.placeholder(
            tf.int32, [None], name="answer_pos")
        self.candidate = tf.placeholder(
            tf.int32, [None, None, None], name="candidate")
        self.candidate_mask = tf.placeholder(
            tf.int32, [None, None], name="candidate_mask")
        self.position = tf.placeholder(
            tf.int32, [None], name="cl_position")

        # word mask
        self.doc_mask = tf.placeholder(
            tf.int32, [None, None], name="doc_mask")
        self.doc_sent_mask = tf.placeholder(
            tf.int32, [None, None, None], name="doc_mask_sent")
        self.qry_mask = tf.placeholder(
            tf.int32, [None, None], name="query_mask")
        # extra features
        self.feat = tf.placeholder(
            tf.int32, [None, None], name="features")

        # model parameters
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

        # word embedding
        if embed_init is None:
            word_embedding = tf.get_variable(
                "word_embedding", [self.n_vocab, 384],
                initializer=tf.random_uniform_initializer(-0.05,0.05),
                trainable=self.train_emb)
        else:
            word_embedding = tf.Variable(embed_init, trainable=self.train_emb,
                                         name="word_embedding")
        doc_embed = tf.nn.embedding_lookup(word_embedding, self.doc)
        qry_embed = tf.nn.embedding_lookup(word_embedding, self.qry)

        # feature embedding
        feature_embedding = tf.get_variable(
            "feature_embedding", [2, 2],
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=self.train_emb)
        feat_embed = tf.nn.embedding_lookup(feature_embedding, self.feat)

        # answer
        # self.answer: [batch_size]  the index of the candidate
        # gather the batch_id with answer
        answer_idx = tf.stack([tf.range(tf.shape(self.answer)[0]), self.answer], -1)
        # find position in the text (reasoner part use)
        answer_text_idx = tf.map_fn(fn=lambda x: self.candidate[x[0], :, x[1]], elems=answer_idx)
        answer_text_idx = tf.arg_max(answer_text_idx, -1)

        # l2_reg
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        # cnn & rnn
        def cnn(_input, name, expand = False):
            _input = tf.expand_dims(_input, -1)
            # m = 3
            # self.Nf = 32
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                filter_shape = [3, self.embed_dim+2, 1, self.Nf] if expand else [3, self.embed_dim, 1, self.Nf]
                W = tf.get_variable(name+"weights", filter_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1),trainable=True,regularizer=regularizer)
                b = tf.get_variable(name+"biases", [self.Nf],
                                    initializer=tf.constant_initializer(0.0),trainable=True,regularizer=regularizer)
                conv = tf.nn.conv2d(
                    _input, W, strides=[1, 1, 1, 1], padding="VALID",
                    name=name+"conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name=name+"relu")  # B*s, L,1,100
                h = tf.squeeze(h,-2)
                # max_pooling
                pooled = tf.reduce_max(h, 1)  # B*s x 100

                return pooled

        def rnn(_input, mask, name, result=True, reuse= tf.AUTO_REUSE):

            with tf.variable_scope(name, reuse=reuse):
                fw_qry = GRU(self.gru_size, reuse=reuse)
                bk_qry = GRU(self.gru_size, reuse=reuse)
                #fw_qry = tf.nn.rnn_cell.DropoutWrapper(fw_qry,input_keep_prob=0.2,output_keep_prob=0.5)
                #bk_qry = tf.nn.rnn_cell.DropoutWrapper(bk_qry,input_keep_prob=0.2,output_keep_prob=0.5)
                seq_length = tf.reduce_sum(mask, axis=1)
                doc, doc_state = \
                    tf.nn.bidirectional_dynamic_rnn(
                        fw_qry, bk_qry, _input, sequence_length=seq_length,
                        dtype=tf.float32, scope=name)
                doc = tf.concat(doc, -1)
                doc_state = tf.concat(doc_state, -1)
                # doc use doc_state  && qry use doc
                return doc_state if result else doc

        # Extractor
        doc_rnn = rnn(doc_embed, self.doc_mask, name="doc_rnn", result=False)
        qry_rnn = rnn(qry_embed, self.qry_mask, name="qry_rnn")

        # extractor output
        k_idx, pk, cand_prob = self.comparison_ext(doc_rnn, qry_rnn, self.candidate, self.candidate_mask)
        
        self.test = tf.shape(cand_prob), cand_prob
        
        # loss
        loss_e = tf.reduce_mean(-tf.math.log(tf.clip_by_value(tf.gather_nd(cand_prob,answer_idx),1e-15,1)),-1)
        #self.loss =loss_e
        #self.equal = tf.cast(tf.equal(k_idx[:,0],self.answer),tf.float32)
        # Reasoner
        sent_num = tf.shape(self.doc_sent)[1]
        sent_word = tf.shape(self.doc_sent)[2]
        _doc_sent = tf.reshape(self.doc_sent, [-1, sent_word])
        doc_sent_embed = tf.nn.embedding_lookup(word_embedding, self.doc_sent)  # (batch, sentence, word, embed)

        # find out every candidate in the doc 
        # cand_embed  (K, batch, embed)
        _idx = tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(k_idx)[0]), -1), [1, self.K]), [-1])
        _k_idx = tf.stack([_idx, tf.reshape(k_idx, [-1])], -1)
        k_text_idx = tf.map_fn(fn=lambda x: self.candidate[x[0], :, x[1]], elems=_k_idx)
        k_text_idx = tf.arg_max(k_text_idx, -1)
        k_text = tf.stack([_idx, tf.cast(tf.reshape(k_text_idx, [-1]), tf.int32)], -1)
        k_text = tf.gather_nd(self.doc, k_text)
        cand_embed = tf.nn.embedding_lookup(word_embedding, tf.reshape(k_text,[self.K,-1]))

        # matrix M  (K, batch, sentence, word, 2)
        # inner product of word in the sentence with candidate
        m_1 = tf.matmul(doc_sent_embed,
                        tf.tile(tf.expand_dims(tf.transpose(cand_embed,[1,0,2]),1),[1,tf.shape(doc_sent_embed)[1],1,1]),
                        transpose_b=True)
        m_1 = tf.transpose(m_1,[3,0,1,2])
        # max inner product of each word in sentence with any word in qry
        m_2 = tf.matmul(doc_sent_embed,tf.tile(tf.expand_dims(qry_embed,1),[1,tf.shape(doc_sent_embed)[1],1,1]),
                        transpose_b=True)
        m_2 = tf.reduce_max(m_2, -1)
        m_2 = tf.tile(tf.expand_dims(m_2,0),[self.K,1,1,1])
        m = tf.stack([m_1,m_2],-1)

        # replace the @placeholder
        qry_rep = tf.tile(self.qry, [self.K, 1])  # size (K, qry_length)
        _candidate = tf.expand_dims(tf.reshape(k_text, [-1]), -1)
        qry_rep = tf.concat([qry_rep[:, :self.position[0]], _candidate,
                             qry_rep[:, self.position[0] + 1:]], axis=1)
        qry_rep = tf.reshape(qry_rep, [self.K,-1,tf.shape(qry_rep)[-1]])
        qry_rep_embed = tf.nn.embedding_lookup(word_embedding, qry_rep)
        qry_rep_embed = tf.reshape(qry_rep_embed, [-1,tf.shape(qry_rep_embed)[-2],tf.shape(qry_rep_embed)[-1]])

        _doc_sent_embed = tf.tile(tf.expand_dims(doc_sent_embed, 0), [self.K, 1, 1, 1, 1])
        _doc_expand_embed = tf.concat([_doc_sent_embed, m], -1)
        _doc_expand_embed = tf.reshape(_doc_expand_embed, [-1, sent_word, tf.shape(_doc_expand_embed)[-1]])
        #_doc_expand_embed: [K*batch_size*sentence, words, embed]
        #qry_rep_embed: [K*batch_size, words, embed]
        doc_cnn = cnn(_doc_expand_embed, 'doc', expand=True)
        doc_cnn = tf.reshape(doc_cnn, [-1, sent_num, self.Nf])
        _doc_cnn = tf.expand_dims(doc_cnn, -2)
        qry_cnn = cnn(qry_rep_embed, 'qry')
        _qry_cnn = tf.expand_dims(qry_cnn, 1)
        _qry_cnn = tf.tile(tf.expand_dims(_qry_cnn, -1), [1, sent_num, 1, 1])
        # _doc_cnn * R * _qry_cnn
        R = tf.get_variable('R', [self.Nf, self.Nf],
                            initializer=tf.truncated_normal_initializer(stddev=0.1),trainable=True,regularizer=regularizer)
        sigma = tf.matmul(_doc_cnn, R)
        sigma = tf.squeeze(tf.matmul(sigma, _qry_cnn), -1)
        x_ik = tf.concat([sigma, doc_cnn, tf.squeeze(_qry_cnn, -1)], -1)
        yk = self.comparison_rea(x_ik, self.doc_sent_mask, "compare_rea")
        # reasoner part output
        ek = tf.transpose(tf.nn.softmax(yk,0),[1,0])


        # total output
        pai_k = tf.add(tf.log(pk+1e-15),tf.log(ek+1e-15))
        # check if correct answer is in pai_k
        check = tf.map_fn(fn=lambda x:tf.equal(k_idx[x[0],:],x[1]), elems=answer_idx, dtype=tf.bool)
        # loss_r = gama - true(pai) + each         
        # expect correct answer and do not need negative float
        pai_star = tf.reduce_sum(tf.multiply(tf.cast(check,tf.float32),pai_k),-1)
        loss_r = tf.subtract(self.gama,pai_star)
        loss_r = tf.add(tf.tile(tf.expand_dims(loss_r,-1),[1,self.K]),pai_k)
        loss_r = tf.subtract(loss_r, tf.multiply(self.gama, tf.cast(check,tf.float32)))
        neg_check = tf.cast(tf.sign(1+tf.sign(loss_r)),tf.float32)
        loss_r = tf.reduce_mean(tf.reduce_sum(tf.multiply(loss_r,neg_check),-1))
        # le + 50 * lr
        total_loss = loss_e + self.lam * loss_r
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = total_loss + tf.add_n(reg_variables)

        equal = tf.arg_max(pai_k,-1)
        equal = tf.stack([tf.range(tf.shape(equal)[0]), tf.cast(tf.reshape(equal, [-1]),tf.int32)], -1)
        equal = tf.gather_nd(k_idx, equal)
        equal = tf.equal(tf.cast(equal,tf.int32),self.answer)
        self.equal = tf.cast(equal, tf.float32)
        self.accuracy = tf.reduce_sum(self.equal)

        vars_list = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr)
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_list), grad_clip)
        # for grad, var in zip(grads, vars_list):
        #     tf.summary.histogram(var.name + '/gradient', grad)
        self.updates = optimizer.apply_gradients(zip(grads, vars_list))

    def comparison_ext(self, d_rnn, q_rnn, c, c_mask):
        '''
           d_rnn: [batch_size, words, 2d]
           q_rnn: [batch_size, 2d]
           c: [batch, word_in_doc, candidate_number]  bool
           c_mask: candidate mask
        '''
        si = tf.nn.softmax(tf.squeeze(tf.matmul(d_rnn, tf.expand_dims(q_rnn, -1))))
        # do not need words that are not candidate
        asi = tf.multiply(si, tf.cast(c_mask, tf.float32))
        # sum the prob of same candidate 
        cand_prob = tf.transpose(tf.multiply(tf.transpose(asi,[1,0]),
                                        tf.cast(tf.transpose(c,[2,1,0]),tf.float32)),[2,1,0])
        cand_prob = tf.reduce_sum(cand_prob,1)
        # normalization (fix the softmax rate)
        total = tf.add(1e-15,tf.reduce_sum(cand_prob,-1,keepdims=True))
        _cand_prob = cand_prob / total
        # select top K candidate
        k_prob, k_idx = tf.nn.top_k(_cand_prob,self.K)
        return k_idx, k_prob, _cand_prob


    def comparison_rea(self, x_ik, _mask_d, scope=None):
        """
        x_ik: (K*B) x L x (2+Nf+Nf)
        _mask_d : doc_sent_mask
        """
        _input = x_ik
        # claim the shape
        _input.set_shape([None, None, 1+2*self.Nf])
        _mask_d = tf.reshape(tf.tile(tf.expand_dims(_mask_d,0),[self.K,1,1,1]),[-1,tf.shape(_mask_d)[-2],tf.shape(_mask_d)[-1]]) 
        fw_qry = GRU(self.gru_size, name=scope + "_fw__", reuse=tf.AUTO_REUSE)
        bk_qry = GRU(self.gru_size, name=scope + "_bk__", reuse=tf.AUTO_REUSE)
        seq_length = tf.reduce_sum(tf.sign(tf.reduce_sum(_mask_d, -1)), axis=-1)

        _, doc_state = \
            tf.nn.bidirectional_dynamic_rnn(
                fw_qry, bk_qry, _input, sequence_length=seq_length,
                dtype=tf.float32, scope="mixed_rnn")
        doc_state = tf.concat(doc_state, -1)
        # only need a scalar for output
        yk = tf.squeeze(tf.layers.dense(doc_state, 1, name=scope+'full_connect', reuse=tf.AUTO_REUSE), -1)
        yk = tf.reshape(yk,[self.K,-1])
        return yk

    def train(self, sess, dw, dsw, dt, qw, qt, nw, a, m_dw, m_dsw, m_qw, tt,
              tm, c, m_c, cl, fnames, dropout, learning_rate):
        """
        train model
        Args:
        - data: (object) containing training data
        """
        feed_dict = {self.doc_sent: dsw, self.qry: qw,
                     self.doc: dw, self.doc_mask: m_dw,
                     self.answer: a,
                     self.doc_sent_mask: m_dsw, self.qry_mask: m_qw,
                     self.candidate: c, self.candidate_mask: m_c, self.position: cl,
                     self.lr: learning_rate}
        if self.use_feat:
            feat = prepare_input(dw, qw)
            feed_dict += {self.feat: feat}
        loss, acc, _, test = \
            sess.run([self.loss, self.accuracy, self.updates, self.test], feed_dict)
        return loss, acc, test

    def validate(self, sess, data):
        """
        test the model
        """
        loss = acc_p = acc_n = n_exmple = 0
        tr = trange(
            len(data),
            desc="loss: {:.3f}, acc: {:.3f}".format(0.0, 0.0),
            leave=False)
        start = time.time()
        for dw, dsw, dt, qw, qt, nw, a, m_dw, m_dsw, m_qw, tt, \
            tm, c, m_c, cl, fnames in data:
            feed_dict = {self.doc_sent: dsw, self.qry: qw,
                     	 self.doc: dw, self.doc_mask: m_dw,
                     	 self.answer: a,
                     	 self.doc_sent_mask: m_dsw, self.qry_mask: m_qw,
                     	 self.candidate: c, self.candidate_mask: m_c, self.position: cl,
                         self.lr: 0.0}
            if self.use_feat:
                feat = prepare_input(dw, qw)
                feed_dict += {self.feat: feat}
            _acc = sess.run(self.accuracy, feed_dict)
            # _acc = sess.run(self.seq_length, feed_dict)
            n_exmple += dw.shape[0]
            acc_p += _acc
            tr.set_description("loss: {:.3f}, acc: {:.3f}".
                               format(loss, _acc / dw.shape[0]))
            tr.update()
        tr.close()
        '''
        tr = trange(
            len(data),
            desc="loss: {:.3f}, acc: {:.3f}".format(0.0, 0.0),
            leave=False)
        start = time.time()
        for dw, dsw, dt, qw, qt, nw, a, m_dw, m_dsw, m_qw, tt, \
            tm, c, m_c, cl, fnames in data:
            feed_dict = {self.doc_sent: dsw, self.qry: qw
                     	 self.doc: dw, self.doc_mask: m_dw,
                     	 self.answer: a,
                     	 self.doc_sent_mask: m_dsw, self.qry_mask: m_qw, self.neg_mask: m_qw,
                     	 self.candidate: c, self.candidate_mask: m_c, self.position: cl,
                         self.keep_prob: 1.0,
                         self.lr: 0.0}
            if self.use_feat:
                feat = prepare_input(dw, qw)
                feed_dict += {self.feat: feat}
            _acc = sess.run(self.accuracy, feed_dict)
            acc_n += _acc
            tr.set_description("loss: {:.3f}, acc: {:.3f}".
                               format(loss, _acc / dw.shape[0]))
            tr.update()
        tr.close()
        '''
        loss /= n_exmple
        acc_p /= n_exmple
        #acc_n /= n_exmple
        spend = (time.time() - start) / 60
        statement = "\nloss: {:.3f}, acc_p: {:.3f}\n" \
            .format(loss, acc_p)

        logging.info(statement)

        return loss, acc_p
#!/usr/bin/env python
# -*- coding: utf-8 -*-
