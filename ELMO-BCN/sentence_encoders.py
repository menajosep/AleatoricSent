# This file makes use of the InferSent, SentEval and CoVe libraries, and may contain adapted code from the repositories
# containing these libraries. Their licenses can be found in <this-repository>/Licenses.
#
# InferSent and SentEval:
#   Copyright (c) 2017-present, Facebook, Inc. All rights reserved.
#   InferSent repository: https://github.com/facebookresearch/InferSent
#   SentEval repository: https://github.com/facebookresearch/SentEval
#   Reference: Conneau, Alexis, Kiela, Douwe, Schwenk, Holger, Barrault, Loic, and Bordes, Antoine. Supervised learning
#              of universal sentence representations from natural language inference data. In Proceedings of the 2017
#              Conference on Empirical Methods in Natural Language Processing, pp. 670-680. Association for
#              Computational Linguistics, 2017.
#
# CoVe:
#   Copyright (c) 2017, Salesforce.com, Inc. All rights reserved.
#   Repository: https://github.com/salesforce/cove
#   Reference: McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation:
#              Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp, 6297-6308.
#              Curran Associates, Inc., 2017.
#
# This code also makes use of GloVe: Pennington, Jeffrey, Socher, Richard, and Manning, Christopher D. Glove: Global
# vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543, 2014.
#
# This code also makes use of Keras: Keras: Chollet, Francois et al. Keras. https://github.com/keras-team/keras, 2015
#
# This code also makes use of PyTorch: Pytorch. [online]. Available at: https://github.com/pytorch/pytorch
#

import sys

import numpy as np

class GloVeEncoder:
    def __init__(self, glove_path, ignore_glove_header=False):
        self.glove_path = glove_path
        self.ignore_glove_header = ignore_glove_header
        self.glove_dim = -1
        self.glove_embeddings_dict = dict()
        self.max_sent_len = 0
        # load() must be called before sentence embeddings can be generated - see below

    def load(self, samples):
        print("\nLoading GloVe embeddings...")
        f = open(self.glove_path)
        first_line = True
        for line in f:
            if first_line and self.ignore_glove_header:
                first_line = False
                continue
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            if self.glove_dim == -1:
                self.glove_dim = len(embedding)
            assert self.glove_dim == len(embedding)
            if word in samples:
                self.glove_embeddings_dict[word] = embedding
        f.close()
        if len(self.glove_embeddings_dict) == 0 or self.glove_dim == -1:
            print("ERROR: Failed to load GloVe embeddings.")
            sys.exit(1)
        print("Successfully loaded GloVe embeddings (vocab size: " + str(
            len(self.glove_embeddings_dict)) + ", dimensions: " + str(self.glove_dim) + ").")

    # Input sequence (tokenized sentence) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
    def encode_sentence(self, tokenized_sentence):
        if self.glove_dim == -1:
            print("ERROR: load() has not been called on encoder.")
            sys.exit(1)
        glove_embeddings = []
        for word in tokenized_sentence:
            try:
                glove_embedding = np.array(self.glove_embeddings_dict[word])
            except KeyError:
                # 1e-10 for unknown words, as recommended on https://github.com/rgsachin/CoVe
                glove_embedding = np.full(self.glove_dim, 1e-10)
            assert glove_embedding.shape == (self.glove_dim,)
            glove_embeddings.append(glove_embedding)
        glove = np.array([glove_embeddings])
        glove = glove[0]
        assert glove.shape == (len(tokenized_sentence), self.glove_dim)
        if glove.shape[0] > self.max_sent_len:
            self.max_sent_len = glove.shape[0]
        return glove

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_embed_dim(self):
        return self.glove_dim

class CoVeEncoder:
    def __init__(self, glove_path, cove_path, ignore_glove_header=False, cove_dim=900):
        self.glove_path = glove_path
        self.cove_path = cove_path
        self.ignore_glove_header = ignore_glove_header
        self.cove_dim = cove_dim
        self.glove_dim = -1
        self.glove_cove_dim = -1
        self.glove_embeddings_dict = dict()
        self.cove_model = None
        self.max_sent_len = 0
        # load() must be called before sentence embeddings can be generated - see below

    def load(self, samples):
        print("\nLoading GloVe embeddings...")
        f = open(self.glove_path)
        first_line = True
        for line in f:
            if first_line and self.ignore_glove_header:
                first_line = False
                continue
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            if self.glove_dim == -1:
                self.glove_dim = len(embedding)
            assert self.glove_dim == len(embedding)
            if word in samples:
                self.glove_embeddings_dict[word] = embedding
        f.close()
        if len(self.glove_embeddings_dict) == 0 or self.glove_dim == -1:
            print("ERROR: Failed to load GloVe embeddings.")
            sys.exit(1)
        print("Successfully loaded GloVe embeddings (vocab size: " + str(
            len(self.glove_embeddings_dict)) + ", dimensions: " + str(self.glove_dim) + ").")

        # Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
        # - Input: GloVe vectors of dimension - (<batch_size>, <sentence_len>, <self.glove_dim>)
        # - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, <self.cove_dim>)
        # - Example: self.cove_model.predict(np.random.rand(1, 10, self.glove_dime))
        # - For unknown words, use a dummy value different to the one used for padding - small non-zero value e.g. 1e-10
        print("\nLoading CoVe model...")
        from keras.models import load_model
        self.glove_cove_dim = self.glove_dim + self.cove_dim
        self.cove_model = load_model(self.cove_path)
        test = self.cove_model.predict(np.random.rand(1, 10, 300))
        assert test.shape == (1, 10, self.cove_dim)
        print("Successfully loaded CoVe model.")

    # Input sequence (tokenized sentence) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
    def encode_sentence(self, tokenized_sentence):
        if self.glove_dim == -1 or self.glove_cove_dim == -1:
            print("ERROR: load() has not been called on encoder.")
            sys.exit(1)
        glove_embeddings = []
        for word in tokenized_sentence:
            try:
                glove_embedding = np.array(self.glove_embeddings_dict[word])
            except KeyError:
                # 1e-10 for unknown words, as recommended on https://github.com/rgsachin/CoVe
                glove_embedding = np.full(self.glove_dim, 1e-10)
            assert glove_embedding.shape == (self.glove_dim,)
            glove_embeddings.append(glove_embedding)
        glove = np.array([glove_embeddings])
        assert glove.shape == (1, len(tokenized_sentence), self.glove_dim)
        cove = self.cove_model.predict(glove)
        cove = cove[0]
        assert cove.shape == (len(tokenized_sentence), self.cove_dim)
        if cove.shape[0] > self.max_sent_len:
            self.max_sent_len = cove.shape[0]
        return cove

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_embed_dim(self):
        return self.cove_dim

class GloVeCoVeEncoder:
    def __init__(self, glove_path, cove_path, ignore_glove_header=False, cove_dim=900):
        self.glove_path = glove_path
        self.cove_path = cove_path
        self.ignore_glove_header = ignore_glove_header
        self.cove_dim = cove_dim
        self.glove_dim = -1
        self.glove_cove_dim = -1
        self.glove_embeddings_dict = dict()
        self.cove_model = None
        self.max_sent_len = 0
        # load() must be called before sentence embeddings can be generated - see below

    def load(self, samples):
        print("\nLoading GloVe embeddings...")
        f = open(self.glove_path)
        first_line = True
        for line in f:
            if first_line and self.ignore_glove_header:
                first_line = False
                continue
            values = line.split(sep=' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            if self.glove_dim == -1:
                self.glove_dim = len(embedding)
            assert self.glove_dim == len(embedding)
            if word in samples:
                self.glove_embeddings_dict[word] = embedding
        f.close()
        if len(self.glove_embeddings_dict) == 0 or self.glove_dim == -1:
            print("ERROR: Failed to load GloVe embeddings.")
            sys.exit(1)
        print("Successfully loaded GloVe embeddings (vocab size: " + str(
            len(self.glove_embeddings_dict)) + ", dimensions: " + str(self.glove_dim) + ").")

        # Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
        # - Input: GloVe vectors of dimension - (<batch_size>, <sentence_len>, <self.glove_dim>)
        # - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, <self.cove_dim>)
        # - Example: self.cove_model.predict(np.random.rand(1, 10, self.glove_dime))
        # - For unknown words, use a dummy value different to the one used for padding - small non-zero value e.g. 1e-10
        print("\nLoading CoVe model...")
        from keras.models import load_model
        self.glove_cove_dim = self.glove_dim + self.cove_dim
        self.cove_model = load_model(self.cove_path)
        test = self.cove_model.predict(np.random.rand(1, 10, 300))
        assert test.shape == (1, 10, self.cove_dim)
        print("Successfully loaded CoVe model.")

    # Input sequence (tokenized sentence) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
    def encode_sentence(self, tokenized_sentence):
        if self.glove_dim == -1 or self.glove_cove_dim == -1:
            print("ERROR: load() has not been called on encoder.")
            sys.exit(1)
        glove_embeddings = []
        for word in tokenized_sentence:
            try:
                glove_embedding = np.array(self.glove_embeddings_dict[word])
            except KeyError:
                # 1e-10 for unknown words, as recommended on https://github.com/rgsachin/CoVe
                glove_embedding = np.full(self.glove_dim, 1e-10)
            assert glove_embedding.shape == (self.glove_dim,)
            glove_embeddings.append(glove_embedding)
        glove = np.array([glove_embeddings])
        assert glove.shape == (1, len(tokenized_sentence), self.glove_dim)
        cove = self.cove_model.predict(glove)
        assert cove.shape == (1, len(tokenized_sentence), self.cove_dim)
        glove_cove = np.concatenate([glove[0], cove[0]], axis=1)
        assert glove_cove.shape == (len(tokenized_sentence), self.glove_dim + self.cove_dim)
        if glove_cove.shape[0] > self.max_sent_len:
            self.max_sent_len = glove_cove.shape[0]
        return glove_cove

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_embed_dim(self):
        return self.glove_cove_dim

class GloVeInferSentEncoder:
    def __init__(self, glove_path, infersent_path, infersent_dim=900, ignore_glove_header=False):
        self.glove_path = glove_path
        self.infersent_path = infersent_path
        self.infersent_dim = infersent_dim
        self.infersent_model = None
        self.glove_dim = -1
        self.max_sent_len = 0
        self.glove_embeddings_dict = dict()
        self.ignore_glove_header = ignore_glove_header
        # load() must be called before sentence embeddings can be generated - see below

    def load(self, samples):
        print("\nLoading GloVe embeddings...")
        f = open(self.glove_path)
        first_line = True
        for line in f:
            if first_line and self.ignore_glove_header:
                first_line = False
                continue
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            if self.glove_dim == -1:
                self.glove_dim = len(embedding)
            assert self.glove_dim == len(embedding)
            if word in samples:
                self.glove_embeddings_dict[word] = embedding
        f.close()
        if len(self.glove_embeddings_dict) == 0 or self.glove_dim == -1:
            print("ERROR: Failed to load GloVe embeddings.")
            sys.exit(1)
        print("Successfully loaded GloVe embeddings (vocab size: " + str(
            len(self.glove_embeddings_dict)) + ", dimensions: " + str(self.glove_dim) + ").")
        self.glove_infersent_dim = self.glove_dim + self.infersent_dim

        print("\nLoading InferSent model...")
        import torch
        self.infersent_model = torch.load(self.infersent_path)
        self.infersent_model.set_glove_path(self.glove_path)
        self.infersent_model.build_vocab(samples, tokenize=False)
        print("Successfully loaded InferSent model.")

    def encode_sentence(self, tokenized_sentence):
        if self.infersent_model is None or self.glove_infersent_dim == -1:
            print("ERROR: load() has not been called on encoder.")
            sys.exit(1)
        infersent_emb = self.infersent_model.encode([' '.join(tokenized_sentence)], bsize=1, tokenize=False)[0]
        assert infersent_emb.shape[0] <= len(tokenized_sentence) + 2
        assert infersent_emb.shape[1] == self.infersent_dim
        assert len(infersent_emb.shape) == 2
        if infersent_emb.shape[0] > self.max_sent_len:
            self.max_sent_len = infersent_emb.shape[0]

        glove_embeddings = []
        glove_embeddings.append(np.full(self.glove_dim, 0)) # For <w> tag
        for word in tokenized_sentence:
            try:
                glove_embedding = np.array(self.glove_embeddings_dict[word])
                assert glove_embedding.shape == (self.glove_dim,)
                glove_embeddings.append(glove_embedding)
            except KeyError:
                pass # InferSent ignores unknown words
        glove_embeddings.append(np.full(self.glove_dim, 0)) # For </w> tag
        glove_emb = np.array([glove_embeddings])
        glove_emb = glove_emb[0]
        
        #print(infersent_emb.shape)
        #print(glove_emb.shape)

        assert glove_emb.shape[0] == infersent_emb.shape[0]
        assert glove_emb.shape[1] == self.glove_dim

        glove_infersent = np.concatenate([glove_emb, infersent_emb], axis=1)
        assert glove_infersent.shape == (infersent_emb.shape[0], self.glove_dim + self.infersent_dim)
        return glove_infersent

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_embed_dim(self):
        return self.glove_infersent_dim

class InferSentEncoder:
    def __init__(self, glove_path, infersent_path, infersent_dim=900):
        self.glove_path = glove_path
        self.infersent_path = infersent_path
        self.infersent_dim = infersent_dim
        self.infersent_model = None
        self.max_sent_len = 0
        # load() must be called before sentence embeddings can be generated - see below

    def load(self, samples):
        print("\nLoading InferSent model...")
        import torch
        self.infersent_model = torch.load(self.infersent_path)
        self.infersent_model.set_glove_path(self.glove_path)
        self.infersent_model.build_vocab(samples, tokenize=False)
        print("Successfully loaded InferSent model.")

    def encode_sentence(self, tokenized_sentence):
        if self.infersent_model is None:
            print("ERROR: load() has not been called on encoder.")
            sys.exit(1)
        embeddings = self.infersent_model.encode([' '.join(tokenized_sentence)], bsize=1, tokenize=False)[0]
        assert embeddings.shape[0] <= len(tokenized_sentence) + 2
        assert embeddings.shape[1] == self.infersent_dim
        assert len(embeddings.shape) == 2
        if embeddings.shape[0] > self.max_sent_len:
            self.max_sent_len = embeddings.shape[0]
        return embeddings

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_embed_dim(self):
        return self.infersent_dim
