#!/usr/bin/env python
# coding:utf-8

from collections import Counter
from utils.common import *
import tqdm
import os
import json

class Vocab(object):
    def __init__(self, config, min_freq=1, max_size=None):
        """
        vocabulary class for text classification, initialized from pretrained embedding file
        and update based on minimum frequency and maximum size
        :param config: helper.configure, Configure Object
        :param min_freq: int, the minimum frequency of tokens
        :param special_token: List[Str], e.g. padding and out-of-vocabulary
        :param max_size: int, maximum size of the overall vocabulary
        """
        print('Building Vocabulary....')
        self.corpus_files = {"TRAIN": config.path.train,
                             "VAL": config.path.dev}

        counter = Counter()
        self.config = config
        # counter for tokens
        self.freqs = {'token': counter.copy(), 'label': counter.copy()}
        # vocab to index
        self.v2i = {'token': {"[PAD]":0, "[UNK]":1}, 'label': {"[PAD]":0,"(":1,")":2,"[END]":3}}
        # index to vocab
        self.i2v = {'token': {0:"[PAD]", 1:"[UNK]"}, 'label': {0:"[PAD]",1:"(",2:")",3:"[END]"}}

        self.min_freq = max(min_freq, 1)
        token_dir = config.path.vocab
        label_dir = config.path.hierachy_node
        vocab_dir = {'token': token_dir, 'label': label_dir}

        with open(label_dir, 'r') as f_in:
            for i, line in enumerate(f_in):
                data = line.rstrip().split('\t')
                assert len(data) == 2
                self.v2i['label'][data[0]] = i+4
                self.i2v['label'][i+4] = data[0]
        if os.path.isfile(token_dir):
            print('Loading Vocabulary from Cached Dictionary...')
            with open(token_dir, 'r') as f_in:
                for i, line in enumerate(f_in):
                    data = line.rstrip().split('\t')
                    assert len(data) == 2
                    self.v2i['token'][data[0]] = i+2
                    self.i2v['token'][i+2] = data[0]
            for vocab in self.v2i.keys():
                print('Vocabulary of ' + vocab + ' ' + str(len(self.v2i[vocab])))
        else:
            print('Generating Vocabulary from Corpus...')
            if self.config.path.embedding != "":
                self._load_pretrained_embedding_vocab()
            self._count_vocab_from_corpus()
            print('Vocabulary of ' + "token" + ' ' + str(len(self.freqs["token"])))

            self._shrink_vocab('token', max_size)

            temp_vocab_list = list(self.freqs["token"].keys())
            for i, k in enumerate(temp_vocab_list):
                self.v2i["token"][k] = i+2
                self.i2v["token"][i+2] = k
            print('Vocabulary of ' + "token" + ' with the size of ' + str(len(self.v2i["token"].keys())))
            with open(vocab_dir["token"], 'w') as f_out:
                for k in list(self.v2i["token"].keys())[2:]:
                    f_out.write(k + '\t' + str(self.freqs["token"][k]) + '\n')
            print('Save Vocabulary in ' + vocab_dir["token"])
        self.padding_index = 0
        self.oov_index = 1

    def _load_pretrained_embedding_vocab(self):
        """
        initialize counter for word in pre-trained word embedding
        """
        pretrained_file_dir = self.config.path.embedding
        with open(pretrained_file_dir, 'r', encoding='utf8') as f_in:
            print('Loading vocabulary from pretrained embedding...')
            for line in tqdm.tqdm(f_in):
                data = line.rstrip('\n').split(' ')
                if len(data) == 2:
                    # first line in pretrained embedding
                    continue
                v = data[0]
                self.freqs['token'][v] += self.min_freq + 1

    def _count_vocab_from_corpus(self):
        """
        count the frequency of tokens in the specified corpus
        """
        for corpus in self.corpus_files.keys():
            with open(self.corpus_files[corpus], 'r') as f_in:
                print('Loading ' + corpus + ' subset...')
                for line in tqdm.tqdm(f_in):
                    data = json.loads(line.rstrip())
                    self._count_vocab_from_sample(data)

    def _count_vocab_from_sample(self, line_dict):
        """
        update the frequency from the current sample
        :param line_dict: Dict{'token': List[Str], 'label': List[Str]}
        """
        for t in line_dict['token']:
            self.freqs['token'][t] += 1

    def _shrink_vocab(self, k, max_size=None):
        """
        shrink the vocabulary
        :param k: Str, field <- 'token', 'label'
        :param max_size: int, the maximum number of vocabulary
        """
        print('Shrinking Vocabulary...')
        tmp_dict = Counter()
        for v in self.freqs[k].keys():
            if self.freqs[k][v] >= self.min_freq:
                tmp_dict[v] = self.freqs[k][v]
        if max_size is not None:
            tmp_list_dict = tmp_dict.most_common(max_size)
            self.freqs[k] = Counter()
            for (t, v) in tmp_list_dict:
                self.freqs[k][t] = v
        print('Shrinking Vocabulary of tokens: ' + str(len(self.freqs[k])))
