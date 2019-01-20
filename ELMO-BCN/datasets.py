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
# This code also makes use of the Stanford SST dataset: Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang,
# Christopher Manning, Andrew Ng and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a
# Sentiment Treebank. In Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).
#
# This code also makes use of the TREC dataset:
#   Xin Li, Dan Roth, Learning Question Classifiers. COLING'02, Aug., 2002.
#   E. M. Voorhees and D. M. Tice. The TREC-8 question answering track evaluation. In TREC, volume 1999, page 82, 1999.

import os
import io

import numpy as np


class SSTDataset(object):
    def __init__(self, n_classes, folder, data_dir, dry_run=False):
        self.n_classes = n_classes
        if dry_run:
            train = self.load_file(os.path.join(data_dir, folder, 'dev.txt'))
        else:
            train = self.load_file(os.path.join(data_dir, folder, 'train.txt'))
        dev = self.load_file(os.path.join(data_dir, folder, 'dev.txt'))
        test = self.load_file(os.path.join(data_dir, folder, 'test.txt'))
        textual_data = {'train': train, 'dev': dev, 'test': test}
        self.total_sentences = 0
        self.max_sent_len = 0
        for key in textual_data:
            for tokenized_sentence in textual_data[key]['X']:
                self.total_sentences += 1
                if self.max_sent_len < len(tokenized_sentence):
                    self.max_sent_len = len(tokenized_sentence)
        print("Successfully loaded dataset (classes: " + str(self.n_classes) + ").")
        self.data = self.get_data(textual_data)

    def load_file(self, fpath):
        textual_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                sample = line.strip().split(' ', 1)
                textual_data['y'].append(int(sample[0]))
                textual_data['X'].append(sample[1].split())
        assert max(textual_data['y']) == self.n_classes - 1
        return textual_data

    def get_data(self, textual_data):
        print("\nGenerating sentence embeddings,,,")
        data = dict()
        done = 0
        milestones = {int(self.total_sentences * 0.1): "10%", int(self.total_sentences * 0.2): "20%",
                      int(self.total_sentences * 0.3): "30%", int(self.total_sentences * 0.4): "40%",
                      int(self.total_sentences * 0.5): "50%", int(self.total_sentences * 0.6): "60%",
                      int(self.total_sentences * 0.7): "70%", int(self.total_sentences * 0.8): "80%",
                      int(self.total_sentences * 0.9): "90%", self.total_sentences: "100%"}
        for key in textual_data:
            data[key] = {}
            data[key]['X1length'] = []
            data[key]['X1'] = []
            for tokenized_sentence in textual_data[key]['X']:
                data[key]['X1length'].append(len(tokenized_sentence))
                data[key]['X1'].append(self.pad(tokenized_sentence))
                done += 1
                if done in milestones:
                    print("  " + milestones[done])
            data[key]['X1length'] = np.array(data[key]['X1length'])
            data[key]['X2'] = data[key]['X1'] # Only one input sentence is needed for SSTBinary so X1 is duplicated
            data[key]['X2length'] = data[key]['X1length']  # Only one input sentence is needed for SSTBinary so X1 is duplicated
            data[key]['y'] = np.array(textual_data[key]['y'])
        print("Successfully generated sentence embeddings,")
        return data

    def get_total_samples(self, key):
        return len(self.data[key]['y'])

    def pad(self, sentence):
        empties = [''] * (self.max_sent_len - len(sentence))
        sentence.extend(empties)
        return sentence

    def get_batch(self, key, indexes=None):
        if indexes is None:
            return self.data[key]['X1'], self.data[key]['X2'], self.data[key]['y']
        X1 = np.take(self.data[key]['X1'], indexes, axis=0)
        X1length = np.take(self.data[key]['X1length'], indexes, axis=0)
        X2 = np.take(self.data[key]['X2'], indexes, axis=0)
        X2length = np.take(self.data[key]['X2length'], indexes, axis=0)
        y = np.take(self.data[key]['y'], indexes, axis=0)
        return X1, X1length, X2, X2length, y

    def get_n_classes(self):
        return self.n_classes

    def get_max_sent_len(self):
        return self.max_sent_len


class SSTFineDataset(SSTDataset):
    def __init__(self, data_dir, dry_run=False):
        print("\nLoading SST Fine dataset...")
        super(SSTFineDataset, self).__init__(5, "SSTFine", data_dir, dry_run=dry_run)

class SSTFineLowerDataset(SSTDataset):
    def __init__(self, data_dir, dry_run=False):
        print("\nLoading SST Fine (lowercase) dataset...")
        super(SSTFineLowerDataset, self).__init__(5, "SSTFine_lower", data_dir, dry_run=dry_run)

