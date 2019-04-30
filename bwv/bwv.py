import numpy as np
import pandas as pd
import re, random
from tqdm import tqdm
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

class WordData:
    def __init__(self, text, m=50):

        self.text = text
        self.vector_size = m

        self.covariance_u = np.identity(m)
        self.covariance_v = np.identity(m)
        self.mean_u = np.random.randn(m,1)
        self.mean_v = np.random.randn(m,1)

        self.P_u = np.identity(m)
        self.P_v = np.identity(m)
        self.R_u = np.zeros((m,1))
        self.R_v = np.zeros((m,1))

        self.P_u_new = np.zeros((m,m))
        self.P_v_new = np.zeros((m,m))
        self.R_u_new = np.zeros((m,1))
        self.R_v_new = np.zeros((m,1))

    def u_parameter_update(self, beta):

        expr = lambda x, y: beta * x + (1-beta) * y

        # update.
        self.R_u = expr(self.R_u_new, self.R_u)
        self.P_u = expr(self.P_u_new, self.P_u)

        # u
        self.covariance_u = np.linalg.inv(self.P_u)
        self.mean_u = np.matmul(self.covariance_u, self.R_u)
        self.covariance_u = np.diag(np.diagonal(self.covariance_u))

        # clear new values.
        self.R_u_new = np.zeros((self.vector_size,1))
        self.P_u_new = np.zeros((self.vector_size,self.vector_size))

    def v_parameter_update(self, beta):

        expr = lambda x, y: beta * x + (1-beta) * y

        # update.
        self.R_v = expr(self.R_v_new, self.R_v)
        self.P_v = expr(self.P_v_new, self.P_v)

        # v
        self.covariance_v = np.linalg.inv(self.P_v)
        self.mean_v = np.matmul(self.covariance_v, self.R_v)
        self.covariance_v = np.diag(np.diagonal(self.covariance_v))

        # clear new values.
        self.R_v_new = np.zeros((self.vector_size,1))
        self.P_v_new = np.zeros((self.vector_size,self.vector_size))

class BWV:
    def __init__(self, corpus, m=50, tau=1.0, gamma=0.7, n_without_stochastic_update=5, vocab_size=20000, sample=0.001):

        self.m = m
        self.tau = tau * np.identity(m)
        self.gamma = gamma
        self.n_without_stochastic_update = n_without_stochastic_update
        self.epoch = 0
        self.vocab_size = vocab_size
        self.sample = sample

        self.corpus, self.vocab_id, self.id_vocab, self.vocab_discard_prob, self.vocab_negative_sample = self._init_corpus(corpus)
        self.words = [WordData(self.id_vocab[n], m=m) for n in range(len(self.id_vocab))]

    def _init_corpus(self, c, threshold=1):

        def process_doc(txt):
            txt = txt.replace('\n', ' ').strip().lower()
            txt = txt.replace('.', ' . ')
            txt = txt.replace('  ', ' ')
            txt = re.sub(r'[^a-z0-9 \-\.]', '', txt)
            txt = txt.split('.')
            txt = [i.split(' ') for i in txt]
            txt = [[j for j in i if j != ''] for i in txt if len(i) > 1]
            text = []
            for sentence in txt:
                s = []
                for word in sentence:
                    try:
                        int(word)
                    except:
                        s.append(word)
                text.append(s)
            return text

        docs = [process_doc(d) for d in c]
        vocab_freq = {}
        for doc in docs:
            for sentence in doc:
                for word in sentence:
                    vocab_freq[word] = vocab_freq.get(word, 0) + 1

        vocab_freq = list(vocab_freq.items())
        vocab_freq.sort(key=lambda x: x[1], reverse=True)
        vocab_freq = vocab_freq[:self.vocab_size]
        vocab_freq = dict(vocab_freq)

        vocab_ix = {}
        ix_vocab = {}
        ix = 0
        for v in vocab_freq.keys():
            vocab_ix[v] = ix
            ix_vocab[ix] = v
            ix += 1

        # remove uncommon words from corpus.
        docs = [[[j for j in i if j in vocab_ix] for i in doc] for doc in docs]
        docs = [[i for i in doc if len(i) > 1] for doc in docs]

        # unigram probabilities
        s = sum([sum([len(sentence) for sentence in doc]) for doc in docs])
        vocab_freq = {k:v / s for k,v in vocab_freq.items()}

        disc_prob = lambda x: (np.sqrt(x / self.sample) + 1) * self.sample / x
        vocab_discard_prob = {k:disc_prob(v) for k,v in vocab_freq.items()}

        prob = lambda x: x ** 0.75
        vocab_negative_sample = {k:prob(v) for k,v in vocab_freq.items()}
        s = sum(vocab_negative_sample.values())
        vocab_negative_sample = {k: v/s for k,v in vocab_negative_sample.items()}

        return docs, vocab_ix, ix_vocab, vocab_discard_prob, vocab_negative_sample

    def get_training_set(self, window_size=4, neg_pos_ratio=1):

        positive_examples = {}
        for doc in self.corpus:
            for sentence in doc:

                # randomly discard common words
                sw = list(set(sentence))
                sample = np.random.uniform(0.0, 1.0, size=(len(sw),))
                for w, p in zip(sw,sample):
                    if p > self.vocab_discard_prob[w]:
                        sentence = [s for s in sentence if s != w]

                # compute positive examples
                for i,word in enumerate(sentence):
                    if word not in positive_examples:
                        positive_examples[word] = {}

                    start = max(0, i-window_size)
                    end = min(len(sentence), i+window_size+1)

                    for j in range(start, end):
                        if j != i:
                            positive_examples[word][sentence[j]] = positive_examples[word].get(sentence[j], 0) + 1

        vocab_set = list(self.vocab_id.keys())
        for word,value in dict(positive_examples).items():
            # pos_set = set(value.keys())
            # neg_set = list(vocab_set.difference(pos_set))
            # # the comment code slows sampling down but is supposed to improve
            # # quality over uniform random number selection.
            # p = [self.vocab_negative_sample[w] for w in neg_set]
            # s = sum(p)
            # p = [i/s for i in p]
            samples = np.random.choice(a=vocab_set, size=int(sum(value.values())*neg_pos_ratio)) # p=p

            for w in samples:
                positive_examples[word][w] = positive_examples[word].get(w, 0) - 1

        return positive_examples

    def train(self, window_size=4):
        beta = 1
        if self.epoch > self.n_without_stochastic_update: beta = (self.epoch-self.n_without_stochastic_update) ** (-1 * self.gamma)

        training_data = self.get_training_set(window_size=window_size)
        total_change = 0

        for i,j_dict in tqdm(training_data.items()):
            i = self.vocab_id[i]
            wi = self.words[i]

            var_wiu = np.expand_dims(np.diagonal(wi.covariance_u), axis=1)
            var_wiv = np.expand_dims(np.diagonal(wi.covariance_v), axis=1)
            xi_ui = ((var_wiu) + np.square(wi.mean_u))
            xi_vi = ((var_wiv) + np.square(wi.mean_v))

            for j, d in j_dict.items():
                j = self.vocab_id[j]
                wj = self.words[j]

                # for u
                var_wjv = np.expand_dims(np.diagonal(wj.covariance_v), axis=1)
                xi = np.matmul(xi_ui.T, (var_wjv + np.square(wj.mean_v)))
                xi = np.sqrt(xi)
                lambda_xi = float((0.5 / xi) * (sigmoid(xi) - 0.5))

                eq = wj.covariance_v + np.matmul(wj.mean_v, wj.mean_v.T)
                wi.P_u_new += abs(d) * (2 * lambda_xi * eq)
                wi.R_u_new += 0.5 * d * wj.mean_v

                # for v
                var_wju = np.expand_dims(np.diagonal(wj.covariance_u), axis=1)
                xi = np.matmul(xi_vi.T, (var_wju + np.square(wj.mean_u)))
                xi = np.sqrt(xi)
                lambda_xi = (0.5 / xi) * (sigmoid(xi) - 0.5)

                eq = wj.covariance_u + np.matmul(wj.mean_u, wj.mean_u.T)
                wi.P_v_new += abs(d) * (2 * lambda_xi * eq)
                wi.R_v_new += 0.5 * d * wj.mean_u

            total_change += np.linalg.norm(wi.R_u_new - wi.R_u)

            wi.P_u_new += self.tau
            wi.P_v_new += self.tau
            wi.u_parameter_update(beta)
            wi.v_parameter_update(beta)

        print(total_change / len(self.words))
        self.most_similar(2, prnt=5)
        self.epoch += 1
        time.sleep(0.1)

    def cosine_similarity(self, i, j):
        # calculate cosine similarities.
        m_y = np.matmul(i.mean_u.T, j.mean_u)
        m_y = m_y / (np.linalg.norm(i.mean_u) * np.linalg.norm(j.mean_u))
        var_y = np.matrix.trace(np.matmul(i.covariance_u, j.covariance_u))
        var_y += np.matmul(np.matmul(i.mean_u.T, i.covariance_u), i.mean_u)
        var_y += np.matmul(np.matmul(j.mean_u.T, j.covariance_u), j.mean_u)
        var_y = var_y / (np.linalg.norm(i.mean_u) * np.linalg.norm(j.mean_u))
        return float(m_y), float(var_y)

    def most_similar(self, i, prnt=None):
        wi = self.words[i]
        if prnt: print(wi.text)
        info = []
        for wj in self.words:
            if wi != wj:
                info.append((wj.text, self.cosine_similarity(wi,wj)))
        info.sort(key=lambda x: x[1][0], reverse=True)

        if prnt:
            for i in info[:prnt]:
                print(i)
        else:
            return info
