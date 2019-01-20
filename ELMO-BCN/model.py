# This file makes use of the InferSent, SentEval and CoVe libraries, and may contain adapted code from the repositories
# containing these libraries. Their licenses can be found in <this-repository>/Licenses.
#
# CoVe:
#   Copyright (c) 2017, Salesforce.com, Inc. All rights reserved.
#   Repository: https://github.com/salesforce/cove
#   Reference: McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation:
#              Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp, 6297-6308.
#              Curran Associates, Inc., 2017.
#
# This code also makes use of TensorFlow: Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig
# Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp,
# Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan
# Mane, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya
# Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden,
# Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on
# heterogeneous systems, 2015. Software available from tensorflow.org.
#

import sys
import os
import timeit
import gc

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Helper function for asserting that dimensions are correct and allowing for "None" dimensions
def dimensions_equal(dim1, dim2):
    return all([d1 == d2 or (d1 == d2) is None for d1, d2 in zip(dim1, dim2)])


class ELMOBCN:
    def __init__(self, params, n_classes, max_sent_len, outputdir, weight_init=0.01, bias_init=0.01):
        self.params = params
        self.n_classes = n_classes
        self.max_sent_len = max_sent_len
        self.outputdir = outputdir
        self.W_init = weight_init
        self.b_init = bias_init
        self.embed_dim = 1024

    def create_model(self):
        print("\nCreating BCN model...")
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        textinputs1 = tf.placeholder(tf.string, shape=[None, self.max_sent_len])
        inputs1length = tf.placeholder(tf.int32, shape=[None])
        textinputs2 = tf.placeholder(tf.string, shape=[None, self.max_sent_len])
        inputs2length = tf.placeholder(tf.int32, shape=[None])
        # Takes 2 input sequences, each of the form [GloVe(w); CoVe(w)] (duplicated if only one input sequence is needed)
        inputs1 = elmo(
                        inputs={
                        "tokens": textinputs1,
                        "sequence_len": inputs1length
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]
        inputs2 = elmo(
                        inputs={
                        "tokens": textinputs2,
                        "sequence_len": inputs2length
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]
        labels = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool)
        assert dimensions_equal(inputs1.shape, (self.params['batch_size'], self.max_sent_len, self.embed_dim))
        assert dimensions_equal(inputs2.shape, (self.params['batch_size'], self.max_sent_len, self.embed_dim))

        # Feedforward network with ReLU activation, applied to each word embedding (word) in the sequence (sentence)
        feedforward_weight = tf.get_variable("feedforward_weight", shape=[self.embed_dim, self.embed_dim],
                                             initializer=tf.random_uniform_initializer(-self.W_init, self.W_init))
        feedforward_bias = tf.get_variable("feedforward_bias", shape=[self.embed_dim],
                                           initializer=tf.constant_initializer(self.b_init))
        with tf.variable_scope("feedforward"):
            feedforward_inputs1 = tf.layers.dropout(inputs1, rate=self.params['dropout_ratio'], training=is_training)
            feedforward_inputs2 = tf.layers.dropout(inputs2, rate=self.params['dropout_ratio'], training=is_training)
            def feedforward(feedforward_input):
                return tf.nn.relu6(tf.matmul(feedforward_input, feedforward_weight) + feedforward_bias)
            feedforward_outputs1 = tf.map_fn(feedforward, feedforward_inputs1)
            feedforward_outputs2 = tf.map_fn(feedforward, feedforward_inputs2)
            assert dimensions_equal(feedforward_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.embed_dim))
            assert dimensions_equal(feedforward_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.embed_dim))

        # BiLSTM processes the resulting sequences
        # The BCN is symmetrical - not sure whether to use same BiLSTM or or two separate BiLSTMs (one for each side).
        # Therefore implemented both, e.g. For SNLI you might want to use different models for inputs1 and inputs2,
        # whereas for sentiment analysis you might want to use the same model
        if self.params['same_bilstm_for_encoder']:
            with tf.variable_scope("bilstm_encoder_scope"):
                encoder_fw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'],
                                                          forget_bias=self.params['bilstm_encoder_forget_bias'])
                encoder_bw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'],
                                                          forget_bias=self.params['bilstm_encoder_forget_bias'])
                encoder_inputs = tf.concat((feedforward_outputs1, feedforward_outputs2), 0)
                encoder_raw_outputs, _ = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell, encoder_bw_cell,
                                                                         encoder_inputs, dtype=tf.float32)
                emcoder_outputs1, encoder_outputs2 = tf.split(tf.concat([encoder_raw_outputs[0], encoder_raw_outputs[-1]], 2), 2, axis=0)
        else:
            with tf.variable_scope("bilstm_encoder_scope1"):
                encoder_fw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'],
                                                          forget_bias=self.params['bilstm_encoder_forget_bias'])
                encoder_bw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'],
                                                          forget_bias=self.params['bilstm_encoder_forget_bias'])
                encoder_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell, encoder_bw_cell,
                                                                          feedforward_outputs1, dtype=tf.float32)
                emcoder_outputs1 = tf.concat([encoder_raw_outputs1[0], encoder_raw_outputs1[-1]], 2)
            with tf.variable_scope("bilstm_encoder_scope2"):
                encoder_fw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'],
                                                           forget_bias=self.params['bilstm_encoder_forget_bias'])
                encoder_bw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'],
                                                           forget_bias=self.params['bilstm_encoder_forget_bias'])
                encoder_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell2, encoder_bw_cell2,
                                                                          feedforward_outputs2, dtype=tf.float32)
                encoder_outputs2 = tf.concat([encoder_raw_outputs2[0], encoder_raw_outputs2[-1]], 2)
        assert dimensions_equal(emcoder_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))
        assert dimensions_equal(encoder_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))

        # Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
        def biattention(biattention_input):
            X = biattention_input[0]
            Y = biattention_input[1]

            # Affinity matrix A=XY^T
            A = tf.matmul(X, Y, transpose_b=True)
            assert dimensions_equal(A.shape, (self.max_sent_len, self.max_sent_len))

            # Column-wise normalisation to extract attention weights
            Ax = tf.nn.softmax(A)
            Ay = tf.nn.softmax(tf.transpose(A))
            assert dimensions_equal(Ax.shape, (self.max_sent_len, self.max_sent_len))
            assert dimensions_equal(Ay.shape, (self.max_sent_len, self.max_sent_len))

            # Context summaries
            Cx = tf.matmul(Ax, X, transpose_a=True)
            Cy = tf.matmul(Ay, X, transpose_a=True)
            assert dimensions_equal(Cx.shape, (self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))
            assert dimensions_equal(Cy.shape, (self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))

            biattention_output1 = tf.concat([X, X - Cy, tf.multiply(X, Cy)], 1)
            biattention_output2 = tf.concat([Y, Y - Cx, tf.multiply(Y, Cx)], 1)
            return biattention_output1, biattention_output2
        biattention_outputs1, biattention_outputs2 = tf.map_fn(biattention, (emcoder_outputs1, encoder_outputs2))
        assert dimensions_equal(biattention_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2*3))
        assert dimensions_equal(biattention_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2*3))

        # Integrate with two separate one-layer BiLSTMs
        with tf.variable_scope("bilstm_integrate_scope1"):
            integrate_fw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'],
                                                        forget_bias=self.params['bilstm_integrate_forget_bias'])
            integrate_bw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'],
                                                        forget_bias=self.params['bilstm_integrate_forget_bias'])
            integrate_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(integrate_fw_cell, integrate_bw_cell,
                                                                        biattention_outputs1, dtype=tf.float32)
            integrate_outputs1 = tf.concat([integrate_raw_outputs1[0], integrate_raw_outputs1[-1]], 2)
        with tf.variable_scope("bilstm_integrate_scope2"):
            integrate_fw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'],
                                                         forget_bias=self.params['bilstm_integrate_forget_bias'])
            integrate_bw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'],
                                                         forget_bias=self.params['bilstm_integrate_forget_bias'])
            integrate_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(integrate_fw_cell2, integrate_bw_cell2,
                                                                        biattention_outputs2, dtype=tf.float32)
            integrate_outputs2 = tf.concat([integrate_raw_outputs2[0], integrate_raw_outputs2[-1]], 2)
        assert dimensions_equal(integrate_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2))
        assert dimensions_equal(integrate_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2))

        # Max, mean, min and self-attentive pooling
        with tf.variable_scope("pool"):
            self_pool_weight1 = tf.get_variable("self_pool_weight1", shape=[self.params['bilstm_integrate_n_hidden']*2, 1],
                                                initializer=tf.random_uniform_initializer(-self.W_init, self.W_init))
            self_pool_bias1 = tf.get_variable("self_pool_bias1", shape=[1],
                                              initializer=tf.constant_initializer(self.b_init))
            self_pool_weight2 = tf.get_variable("self_pool_weight2", shape=[self.params['bilstm_integrate_n_hidden']*2, 1],
                                                initializer=tf.random_uniform_initializer(-self.W_init, self.W_init))
            self_pool_bias2 = tf.get_variable("self_pool_bias2", shape=[1],
                                              initializer=tf.constant_initializer(self.b_init))
            def pool(pool_input):
                Xy = pool_input[0]
                Yx = pool_input[1]
                assert dimensions_equal(Xy.shape, (self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(Xy.shape, (self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2,))

                # Max pooling - just take the (max_sent_len) "columns" in the matrix and get the max in each of them.
                max_Xy = tf.reduce_max(Xy, axis=0)
                max_Yx = tf.reduce_max(Yx, axis=0)
                assert dimensions_equal(max_Xy.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(max_Yx.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Mean pooling - just take the (max_sent_len) "columns" in the matrix and get the mean in each of them.
                mean_Xy = tf.reduce_mean(Xy, axis=0)
                mean_Yx = tf.reduce_mean(Yx, axis=0)
                assert dimensions_equal(mean_Xy.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(mean_Yx.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Min pooling - just take the (max_sent_len) "columns" in the matrix and get the min in each of them.
                min_Xy = tf.reduce_min(Xy, axis=0)
                min_Yx = tf.reduce_min(Yx, axis=0)
                assert dimensions_equal(min_Xy.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(min_Yx.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Self-attentive pooling
                Bx = tf.nn.softmax((tf.matmul(Xy, self_pool_weight1)) + self_pool_bias1)
                By = tf.nn.softmax((tf.matmul(Yx, self_pool_weight2)) + self_pool_bias2)
                assert dimensions_equal(Bx.shape, (self.max_sent_len, 1))
                assert dimensions_equal(By.shape, (self.max_sent_len, 1))
                x_self = tf.squeeze(tf.matmul(Xy, Bx, transpose_a=True))
                y_self = tf.squeeze(tf.matmul(Yx, By, transpose_a=True))
                assert dimensions_equal(x_self.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(y_self.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Combine pooled representations
                pool_output1 = tf.concat([max_Xy, mean_Xy, min_Xy, x_self], 0)
                pool_output2 = tf.concat([max_Yx, mean_Yx, min_Yx, y_self], 0)
                return pool_output1, pool_output2
            pool_outputs1, pool_outputs2 = tf.map_fn(pool, (integrate_outputs1, integrate_outputs2))
            assert dimensions_equal(pool_outputs1.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4))
            assert dimensions_equal(pool_outputs2.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4))

        # FeedForward network (2 relu layers, followed by a softmax)
        with tf.variable_scope("output"):
            joined = tf.concat([pool_outputs1, pool_outputs2], 1)
            assert dimensions_equal(joined.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))



            # Output layer 1: Dropout, fully-connected layer, D->(D/(2 or 4 or 8)) relu
            dp1 = tf.layers.dropout(joined, rate=self.params['dropout_ratio'], training=is_training)
            assert dimensions_equal(dp1.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))
            output_dim1 = int((self.params['bilstm_integrate_n_hidden'] * 2 * 4 * 2) / self.params['maxout_reduction'])
            output_weight1 = tf.get_variable("output_weight1", shape=[self.params['bilstm_integrate_n_hidden']*2*4*2,
                                                                      output_dim1],
                                             initializer=tf.random_uniform_initializer(-self.W_init, self.W_init))
            output_bias1 = tf.get_variable("output_bias1", shape=[output_dim1],
                                           initializer=tf.constant_initializer(self.b_init))

            output_outputs1 = tf.nn.relu6((tf.matmul(dp1, output_weight1) + output_bias1))
            assert dimensions_equal(output_outputs1.shape, (self.params['batch_size'], output_dim1))

            # Output layer 2: Dropout, fully-connected layer, D->(D/2) relu
            dp2 = tf.layers.dropout(output_outputs1, rate=self.params['dropout_ratio'], training=is_training)
            assert dimensions_equal(dp2.shape, (self.params['batch_size'], output_dim1))
            output_dim2 = int(output_dim1 / 2)
            output_weight2 = tf.get_variable("output_weight2", shape=[output_dim1, output_dim2],
                                             initializer=tf.random_uniform_initializer(-self.W_init, self.W_init))
            output_bias2 = tf.get_variable("output_bias2", shape=[output_dim2],
                                           initializer=tf.constant_initializer(self.b_init))
            output_outputs2 = tf.nn.relu6((tf.matmul(dp2, output_weight2) + output_bias2))
            assert dimensions_equal(output_outputs2.shape, (self.params['batch_size'], output_dim2))

            # SoftMax layer: Dropout, D->n_classes fully-connected layer, SoftMax
            dp3 = tf.layers.dropout(output_outputs2, rate=self.params['dropout_ratio'], training=is_training)
            assert dimensions_equal(dp3.shape, (self.params['batch_size'], output_dim2))


            softmax_weight = tf.get_variable("softmax_weight", shape=[output_dim2, self.n_classes],
                                             initializer=tf.random_uniform_initializer(-self.W_init, self.W_init))
            softmax_bias = tf.get_variable("softmax_bias", shape=[self.n_classes],
                                           initializer=tf.constant_initializer(self.b_init))


            logits = (tf.matmul(dp3, softmax_weight) + softmax_bias)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            cost = tf.reduce_mean(cross_entropy)

            if self.params['optimizer'] == "adam":
                train_step = tf.train.AdamOptimizer(self.params['learning_rate'],
                                                    beta1=self.params['adam_beta1'],
                                                    beta2=self.params['adam_beta2'],
                                                    epsilon=self.params['adam_epsilon']).minimize(cost)
            elif self.params['optimizer'] == "gradientdescent":
                train_step = tf.train.GradientDescentOptimizer(self.params['learning_rate']).minimize(cost)
            else:
                print("ERROR: Invalid optimizer: \"" + self.params['optimizer'] + "\".")
                sys.exit(1)

            predict = tf.argmax(tf.nn.softmax(logits), axis=1)

        print("Successfully created BCN model.")
        return textinputs1, inputs1length, textinputs2, inputs2length, labels, is_training, predict, cost, train_step

    def dry_run(self):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            return self.create_model()

    def train(self, dataset):
        best_dev_accuracy = -1
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            textinputs1, inputs1length, textinputs2, inputs2length, labels, is_training, predict, loss_op, train_op = self.create_model()
        with tf.Session(graph=graph) as sess:
            print("\nTraining model...")
            sess.run(tf.global_variables_initializer())
            train_data_len = dataset.get_total_samples("train")
            total_train_batches = train_data_len // self.params['batch_size']
            train_milestones = {int(total_train_batches * 0.1): "10%", int(total_train_batches * 0.2): "20%",
                                int(total_train_batches * 0.3): "30%", int(total_train_batches * 0.4): "40%",
                                int(total_train_batches * 0.5): "50%", int(total_train_batches * 0.6): "60%",
                                int(total_train_batches * 0.7): "70%", int(total_train_batches * 0.8): "80%",
                                int(total_train_batches * 0.9): "90%", total_train_batches: "100%"}
            best_epoch_number = 0
            epochs_since_last_save = 0
            for epoch in range(self.params['n_epochs']):
                print("  ============== Epoch " + str(epoch + 1) + " of " + str(self.params['n_epochs']) + " ==============")
                epoch_start_time = timeit.default_timer()
                done = 0
                average_loss = 0
                indexes = np.random.permutation(train_data_len)
                for i in range(total_train_batches):
                    batch_indexes = indexes[i * self.params['batch_size']: (i + 1) * self.params['batch_size']]
                    batch_X1, batchX1length, batch_X2, batch_X2length, batch_y = dataset.get_batch('train', batch_indexes)
                    _, loss = sess.run([train_op, loss_op], feed_dict={textinputs1: batch_X1, textinputs2: batch_X2,
                                                                       inputs1length: batchX1length,
                                                                       inputs2length: batch_X2length,
                                                                       labels: batch_y, is_training: True})
                    average_loss += (loss / total_train_batches)
                    done += 1
                    if done in train_milestones:
                        print("    " + train_milestones[done])
                print("    Loss: " + str(average_loss))
                print("    Computing dev accuracy...")
                dev_accuracy = self.calculate_accuracy(dataset, sess, textinputs1, inputs1length,
                                                       textinputs2, inputs2length, labels, is_training,
                                                       predict, set_name="dev")
                print("      Dev accuracy:" + str(dev_accuracy))
                print("    Epoch took %s seconds" % (timeit.default_timer() - epoch_start_time))
                if dev_accuracy > best_dev_accuracy:
                    # If dev accuracy improved, save the model after this epoch
                    best_dev_accuracy = dev_accuracy
                    best_epoch_number = epoch
                    epochs_since_last_save = 0
                    tf.train.Saver().save(sess, os.path.join(self.outputdir, 'model'))
                else:
                    # If dev accuracy got worse, don't save
                    epochs_since_last_save += 1
                gc.collect()
                if epochs_since_last_save >= 7:
                    # If dev accuracy keeps getting worse, stop training (early stopping)
                    break
            print("Finished training model after " + str(best_epoch_number + 1) + " epochs. Model is saved in: " + self.outputdir)
            print("Best dev accuracy: " + str(best_dev_accuracy))
        return best_dev_accuracy

    def calculate_accuracy(self, dataset, sess, textinputs1, inputs1length, textinputs2, inputs2length, labels, is_training, predict, set_name="test", verbose=False):
        test_data_len = dataset.get_total_samples(set_name)
        total_test_batches = test_data_len // self.params['batch_size']
        test_milestones = {int(total_test_batches * 0.1): "10%", int(total_test_batches * 0.2): "20%",
                           int(total_test_batches * 0.3): "30%", int(total_test_batches * 0.4): "40%",
                           int(total_test_batches * 0.5): "50%", int(total_test_batches * 0.6): "60%",
                           int(total_test_batches * 0.7): "70%", int(total_test_batches * 0.8): "80%",
                           int(total_test_batches * 0.9): "90%", total_test_batches: "100%"}
        done = 0
        test_y = []
        predicted = []
        indexes = np.arange(test_data_len)
        for i in range(total_test_batches):
            batch_indexes = indexes[i * self.params['batch_size']: (i + 1) * self.params['batch_size']]
            batch_X1, batchX1length, batch_X2, batch_X2length, batch_y = dataset.get_batch(set_name, batch_indexes)
            for item in batch_y:
                test_y.append(item)
            batch_pred = list(sess.run(predict, feed_dict={textinputs1: batch_X1, textinputs2: batch_X2,
                                                           inputs1length: batchX1length,
                                                           inputs2length: batch_X2length,
                                                           labels: batch_y, is_training: False}))
            for item in batch_pred:
                predicted.append(item)
            done += 1
            if verbose and done in test_milestones:
                print("  " + test_milestones[done])
        return sum([p == a for p, a in zip(predicted, test_y)]) / float(test_data_len)

    def test(self, dataset):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            textinputs1, inputs1length, textinputs2, inputs2length, labels, is_training, predict, _, _ = self.create_model()
        with tf.Session(graph=graph) as sess:
            print("\nComputing test accuracy...")
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, os.path.join(self.outputdir, 'model'))
            accuracy = self.calculate_accuracy(dataset, sess, textinputs1, inputs1length,
                                               textinputs2, inputs2length,
                                               labels, is_training, predict, verbose=True)
            print("Test accuracy:    " + str(accuracy))
        return accuracy
