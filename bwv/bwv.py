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

        self.mean_u = np.random.randn(m,1)
        self.mean_v = np.random.randn(m,1)
        self.covariance_u = np.identity(m)
        self.covariance_v = np.identity(m)

        self.P_u = np.identity(m)
        self.P_v = np.identity(m)
        self.P_u_new = np.zeros((m,m))
        self.P_v_new = np.zeros((m,m))

        self.R_u = np.zeros((m,1))
        self.R_v = np.zeros((m,1))
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
        self.R_u_new = np.zeros((m,1))
        self.P_u_new = np.zeros((m,m))

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
        self.R_v_new = np.zeros((m,1))
        self.P_v_new = np.zeros((m,m))

class BWV:
    def __init__(self, corpus, m=40, tau=0.0, gamma=0.7, n_without_stochastic_update=5):

        self.m = m
        self.tau = tau * np.identity(m)
        self.gamma = gamma
        self.n_without_stochastic_update = n_without_stochastic_update
        self.epoch = 0

        self.corpus, self.vocab_id, self.id_vocab = self._init_corpus(corpus)
        self.words = [WordData(v,m=m) for v in vocab_id.items()]

    def _init_corpus(self, c):

        def clean(t):
            t = re.sub(r'\s+', ' ', t) # remove extra spacing
            t = t.lower() # lowercase
            t = re.sub(r'[^a-zA-Z\d\s-]', '', t) # keep al-num characters and hyphens
            t = t.strip()
            t = t.split()
            return t

        # c should be a list of lists at this point.
        def gen_vocab(c):
            vocab_id = {}
            id_vocab = {}
            index = 0
            for word in raw_text:
                if word not in vocab_id:
                    vocab_id[word] = index
                    id_vocab[index] = word
                    index += 1
            return vocab_id, id_vocab

        c = [clean(i) for i in c]
        vocab_id, id_vocab = gen_vocab(c)
        return c, vocab_id, id_vocab

    def get_training_set(self, window_size=4, neg_pos_ratio=1):

        training_data = {}

        # positive examples
        for text_i, word in enumerate(text):
            word_i = self.vocab_id[word]
            if word_i not in training_data:
                training_data[word_i] = {}

            start_window = max(0, text_i - window_size)
            end_window = min(len(text), text_i + window_size + 1)

            for text_j in range(start_window, end_window):
                word_j = self.vocab_id[text[text_j]]
                if text_i != text_j:
                    training_data[word_i][word_j] = training_data[word_i].get(word_j, 0) + 1

        # negative_examples
        for word_i in tqdm(training_data.keys()):

            found = 0
            positive_samples = sum(training_data[word_i].values())

            while found < neg_pos_ratio * positive_samples:
                neg_w = random.choice(text)
                neg_i = self.vocab_id[neg_w]
                if (neg_i not in training_data[word_i]) or (training_data[word_i][neg_i] < 0):
                    training_data[word_i][neg_i] = training_data[word_i].get(neg_i, 0) - 1
                    found += 1

        return training_data

    def train(self):
        beta = 1
        if self.epoch > self.n_without_stochastic_update: beta = (self.epoch-self.n_without_stochastic_update) ** (-1 * self.gamma)

        training_data = get_data()
        error = 0

        for i,j_dict in tqdm(training_data.items()):

            wi = self.words[i]

            var_wiu = np.expand_dims(np.diagonal(wi.covariance_u), axis=1)
            var_wiv = np.expand_dims(np.diagonal(wi.covariance_v), axis=1)
            xi_ui = ((var_wiu) + np.square(wi.mean_u))
            xi_vi = ((var_wiv) + np.square(wi.mean_v))

            for j, d in j_dict.items():
                wj = self.words[j]

                # for u
                var_wjv = np.expand_dims(np.diagonal(wj.covariance_v), axis=1)
                xi = np.matmul(xi_ui.T, (var_wjv + np.square(wj.mean_v)))
                xi = np.sqrt(xi)
                lambda_xi = (0.5 / xi) * (sigmoid(xi) - 0.5)

                eq = wj.covariance_v + np.matmul(wj.mean_v, wj.mean_v.T)
                wi.P_u_new += abs(d) * (2 * lambda_xi * eq + tau)
                wi.R_u_new += 0.5 * d * wj.mean_v

                # for v
                var_wju = np.expand_dims(np.diagonal(wj.covariance_u), axis=1)
                xi = np.matmul(xi_vi.T, (var_wju + np.square(wj.mean_u)))
                xi = np.sqrt(xi)
                lambda_xi = (0.5 / xi) * (sigmoid(xi) - 0.5)

                eq = wj.covariance_u + np.matmul(wj.mean_u, wj.mean_u.T)
                wi.P_v_new += abs(d) * (2 * lambda_xi * eq + tau)
                wi.R_v_new += 0.5 * d * wj.mean_u

            e += np.linalg.norm(wi.R_u_new - wi.R_u)
            wi.u_parameter_update(beta)
            wi.v_parameter_update(beta)

        print(e / len(words))
        self.most_similar(2, prnt=5)
        time.sleep(0.1)

    def cosine_similarity(self, i, j):
        i = self.words[i]
        j = self.words[j]
        # calculate cosine similarities.
        m_y = np.matmul(i.mean_u.T, j.mean_u)
        m_y = m_y / (np.linalg.norm(i.mean_u) * np.linalg.norm(j.mean_u))
        var_y = np.matrix.trace(np.matmul(i.covariance_u, j.covariance_u))
        var_y += np.matmul(np.matmul(i.mean_u.T, i.covariance_u), i.mean_u)
        var_y += np.matmul(np.matmul(j.mean_u.T, j.covariance_u), j.mean_u)
        # var_y = var_y / (np.linalg.norm(i.mean_u) * np.linalg.norm(j.mean_u))
        return float(m_y), float(var_y)

    def most_similar(self, i, prnt=None):
        wi = self.words[i]
        if prnt: print(wi.text)
        info = []
        for wj in self.words:
            if wi != wj:
                info.append((wj.text, cosine_similarity(wi,wj)))
        info.sort(key=lambda x: x[1][0], reverse=True)

        if prnt:
            for i in info[:prnt]:
                print(i)
        else:
            return info
