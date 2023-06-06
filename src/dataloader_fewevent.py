# -*- coding: utf-8 -*-

import os
import json
import random
from collections import OrderedDict, defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer


class BaseDataset(Dataset):
    def __init__(self, dataset_path, mode, N, K, max_length, tokenizer,
                 vocab=None, prompt_template=None, zero_shot=False):        
        self.dataset_path = dataset_path
        self.mode = mode
        self.N = N
        self.K = K
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.prompt_template = prompt_template
        self.zero_shot = zero_shot
    
    def get_vocab(self, events, vocab=None):
        if not vocab: vocab = {"None": 0}
        for event in events:
            vocab[event] = len(vocab)
        
        return vocab
    
    def permute(self, input_set):
        permute_ids = np.random.permutation(len(input_set['input_ids']))
        for k in input_set.keys():
            input_set[k] = [input_set[k][i] for i in permute_ids]
        
        return input_set

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class FewEventDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_data = json.load(open(self.dataset_path, "r"))
        self.classes = list(self.raw_data.keys())
        self.vocab = self.get_vocab(self.classes, self.vocab)
    
    def preprocess(self, instance, event_type):
        result = {'tokens': [], 'trigger': [], 'trigger_mask':[], 'trigger_label': event_type} 
        sentence = instance['tokens']
        result['tokens'] = sentence
        result['trigger'] = instance['trigger']
        
        trigger_mask = [0] * len(sentence)
        trigger_length = len(instance['trigger'])
        trigger_start_pos = instance['position'][0]
        trigger_end_pos = trigger_start_pos + trigger_length
        for i in range(trigger_start_pos, trigger_end_pos):
            trigger_mask[i] = 1
        
        result['trigger_mask'] = trigger_mask
        
        return result

    def tokenize(self, instance):
        def is_subarray(A, B):
            s = self.max_length
            i, j = 0, 0
            n, m = len(A), len(B)
            while (i < n and j < m):
                if (A[i] == B[j]):
                    s = min(i, s)
                    i += 1
                    j += 1
                    if (j == m):
                        return True, s
                else:
                    i = i - j + 1
                    j = 0
            return False, s

        if self.prompt_template:
            raw_tokens = self.prompt_template.format(' '.join(instance['tokens']))
        else:
            raw_tokens = ' '.join(instance['tokens'])
        raw_trigger = ' '.join(instance['trigger'])
        # raw_trigger_mask = instance['trigger_mask']  # not used in roberta
        # raw_trigger_label = instance['trigger_label']  # not used in roberta

        tokenized_result = self.tokenizer(raw_tokens, max_length=self.max_length, padding='max_length', 
                                          truncation=True, return_attention_mask=True)
        tokenized_trigger_ = self.tokenizer(raw_trigger, add_special_tokens=False, return_attention_mask=False)['input_ids']
        # roberta encodes the space on token's left
        tokenized_trigger_l = self.tokenizer(' '+raw_trigger, add_special_tokens=False, return_attention_mask=False)['input_ids']
        
        in_span, start_idx = is_subarray(tokenized_result['input_ids'], tokenized_trigger_)
        tokenized_trigger = tokenized_trigger_
        if not in_span:
            in_span, start_idx = is_subarray(tokenized_result['input_ids'], tokenized_trigger_l)
            tokenized_trigger = tokenized_trigger_l
        if not in_span or 50264 not in tokenized_result['input_ids']:
            # print('Tokenization failed on instance', raw_tokens)
            return None
        
        end_idx = start_idx + len(tokenized_trigger) - 1
        # tokenized_result['start_positions'] = start_idx
        # tokenized_result['end_positions'] = end_idx
        tokenized_result['trigger_mask'] = [0] * self.max_length
        tokenized_result['trigger_mask'][start_idx:end_idx+1] = [1] * (end_idx-start_idx+1)

        return tokenized_result
    
    def __len__(self):
        if self.mode in ['train', 'dev']:
            return 9999  # return 2 ** 31
        else:
            return 10
    
    def __getitem__(self, index):
        if self.zero_shot:
            return self.get_zeroshot_item(index)
        else:
            return self.get_fewshot_item(index)

    def get_fewshot_item(self, index):
        if len(self.classes) > self.N:
            target_classes = np.random.permutation(self.classes)[:self.N]
        else:
            target_classes = self.classes
        
        support_instances, query_instances = [], []
        support_labels, query_labels = [], []
        for i, class_name in enumerate(target_classes):
            count = 0
            indices = np.random.permutation(len(self.raw_data[class_name]))
            for j in indices:
                if count < self.K:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    tokenized_instance = self.tokenize(instance)
                    if tokenized_instance is None: continue
                    support_instances.append(tokenized_instance)
                    support_labels.append(i)
                else:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    tokenized_instance = self.tokenize(instance)
                    if tokenized_instance is None: continue
                    query_instances.append(tokenized_instance)
                    query_labels.append(i)
                count += 1
                if self.mode == 'train' and count == 2*self.K: break

        support_set, query_set = defaultdict(list), defaultdict(list)
        for instance, label in zip(support_instances, support_labels):
            for key in instance: support_set[key].append(instance[key])
            support_set['label'].append(label)
        for instance, label in zip(query_instances, query_labels):
            for key in instance: query_set[key].append(instance[key])
            query_set['label'].append(label)
        
        support_set = self.permute(support_set)
        query_set = self.permute(query_set)
        
        return support_set, query_set
    
    def get_zeroshot_item(self, index):
        if self.mode == 'train':
            if len(self.classes) > self.N:
                support_classes = np.random.permutation(self.classes)[:self.N]
                query_classes = np.random.permutation(self.classes)[:self.N]
            else:
                support_classes = np.random.permutation(self.classes)
                query_classes = np.random.permutation(self.classes)
            
            support_instances, query_instances = [], []
            support_labels, query_labels = [], []
            for i, class_name in enumerate(support_classes):
                count = 0
                indices = np.random.permutation(len(self.raw_data[class_name]))
                for j in indices:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    tokenized_instance = self.tokenize(instance)
                    if tokenized_instance is None: continue
                    support_instances.append(tokenized_instance)
                    support_labels.append(i)
                    count += 1
                    if count == self.K: break
            
            for i, class_name in enumerate(query_classes):
                count = 0
                indices = np.random.permutation(len(self.raw_data[class_name]))
                for j in indices:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    tokenized_instance = self.tokenize(instance)
                    if tokenized_instance is None: continue
                    query_instances.append(tokenized_instance)
                    query_labels.append(i)
                    count += 1
                    if count == self.K: break

            support_set, query_set = defaultdict(list), defaultdict(list)
            for instance, label in zip(support_instances, support_labels):
                for key in instance: support_set[key].append(instance[key])
                support_set['label'].append(label)
            for instance, label in zip(query_instances, query_labels):
                for key in instance: query_set[key].append(instance[key])
                query_set['label'].append(label)
            
            support_set = self.permute(support_set)
            query_set = self.permute(query_set)
            
        else:
            support_set = defaultdict(list)
            query_classes = np.random.permutation(self.classes)[:self.N]

            query_instances, query_labels = [], []
            for i, class_name in enumerate(query_classes):
                indices = np.random.permutation(len(self.raw_data[class_name]))
                for j in indices:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    tokenized_instance = self.tokenize(instance)
                    if tokenized_instance is None: continue
                    query_instances.append(tokenized_instance)
                    query_labels.append(i)
            
            query_set = defaultdict(list)
            for instance, label in zip(query_instances, query_labels):
                for key in instance: query_set[key].append(instance[key])
                query_set['label'].append(label)
            
            query_set = self.permute(query_set)
        
        return support_set, query_set


def collate_fn(data):
    batch_support, batch_query = [], []
    support_sets, query_sets = zip(*data)
    
    for i in range(len(support_sets)):
        batch_support.append(support_sets[i])
        batch_query.append(query_sets[i])
    
    return batch_support, batch_query


def get_fewevent_loaders(args, N, K, max_length, tokenizer=None, prompt_template=None, train_size=3,
                         eval_size=1, num_workers=0, collate_fn=collate_fn, zero_shot=False):
    fewevent_data_dir = args.fewevent_data_dir
    train_file = args.fewevent_train_file
    dev_file = args.fewevent_dev_file
    test_file = args.fewevent_test_file
    
    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_path = os.path.join(fewevent_data_dir, train_file)
    train_dataset = FewEventDataset(train_path, 'train', N, K, max_length, tokenizer, vocab=None,
                                    prompt_template=prompt_template, zero_shot=zero_shot)
    
    dev_path = os.path.join(fewevent_data_dir, dev_file)
    dev_dataset = FewEventDataset(dev_path, 'dev', N, K, max_length, tokenizer, vocab=train_dataset.vocab,
                                  prompt_template=prompt_template, zero_shot=zero_shot)
    
    test_path = os.path.join(fewevent_data_dir, test_file)
    test_dataset = FewEventDataset(test_path, 'test', N, K, max_length, tokenizer, vocab=dev_dataset.vocab, 
                                   prompt_template=prompt_template, zero_shot=zero_shot)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=eval_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=eval_size,
                             shuffle=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, test_dataset.vocab