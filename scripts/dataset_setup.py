import spacy
# python -m spacy download en_core_web_sm
import csv
import random
import numpy as np
import regex as re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import text_preprocessing as txt

nlp = spacy.load('en_core_web_sm')


class TDataset:
    def __init__(self, query, cq, answer, label, cq_id):
        self.query = query
        self.cq = cq
        self.answer = answer
        self.label = int(label)
        self.cq_id = int(cq_id)


def get_dataset(word2index, template_path, dataset_path, test_dataset_path, test_size):
    exclude_set = []
    template_q = {}
    template_a = {}
    classes = []
    with open(template_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            tq = txt.text_cleaner(row[1]).strip()
            ta = ""
            for item in range(2, len(row)):
                ta = ta + " " + txt.text_cleaner(row[item])
            ta = ta.strip()
            template_q[row[0]] = tq
            template_a[row[0]] = ta
            print(tq, ta, len(tq.split()), len(ta.split()))
            classes.append(row[0])

    # print(template_q.keys())
    total_cq_label = len(classes) + 2

    def query_modifier(s):
        s = s.strip().lower()
        s = txt.text_cleaner(s)
        return s

    def build_dataset(table):
        ret_dataset = []
        data_counter_p = {}
        p_count = 0
        data_counter_n = {}
        n_count = 0
        for row in table:
            query = query_modifier(row[1])
            cq_t = np.zeros(total_cq_label + 1)
            for x in range(2, len(row)):
                if len(row[x]) > 0:
                    cq_t[int(row[x][1:])] = 1
            c = 0
            for i in range(len(classes)):
                template = 'T' + str(int(i + 1))
                if template in exclude_set:
                    continue
                if (template_q.get(template) is not None) and (int(cq_t[i + 1]) == 1) and (template in classes):
                    ret_dataset.append(
                        TDataset(query, template_q.get(template), template_a.get(template), 1, int(i + 1)))

                    if template not in data_counter_p:
                        data_counter_p[template] = 1
                        p_count = p_count + 1
                    else:
                        data_counter_p[template] = data_counter_p[template] + 1
                        p_count = p_count + 1

                else:
                    ret_dataset.append(
                        TDataset(query, template_q.get(template), template_a.get(template), 0, int(i + 1)))

                    if template not in data_counter_n:
                        data_counter_n[template] = 1
                        n_count = n_count + 1
                    else:
                        data_counter_n[template] = data_counter_n[template] + 1
                        n_count = n_count + 1

        print("positives", p_count, data_counter_p)
        print("negatives", n_count, data_counter_n)
        return ret_dataset

    test_queries = []
    with open(test_dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            test_queries.append(row[0])

    # test = 20%, so total 40
    print("total seed test queries", len(test_queries))
    assert test_size * 2 == len(test_queries)

    train = []
    test = []
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            if row[0] in test_queries:
                test.append(row)
            else:
                train.append(row)

    print("test length: ", len(test))

    train_dataset = build_dataset(train)
    # print("train len", len(train_dataset))

    test_dataset = build_dataset(test)
    print("test dataset length", len(test_dataset))

    train_dataset = np.array(train_dataset)
    np.random.shuffle(train_dataset)

    test_dataset = np.array(test_dataset)
    np.random.shuffle(test_dataset)

    train_dataset = ISQDataset(train_dataset, word2index)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_dataset = ISQDataset(test_dataset, word2index)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    return train_loader, test_loader, classes, train_dataset, test_dataset


class ISQDataset(Dataset):
    def __init__(self, dataset1, word2index):
        self.word2index = word2index
        self.max_cq_len = 15
        self.max_answer_len = 10
        self.max_query_len = 20
        self.dataset = self._build_dataset(dataset1)

    def _build_dataset(self, dataset1):
        data = {'query': list(),
                'cq': list(),
                'answer': list(),
                'label': list(),
                'original_query': list(),
                'cq_id': list(),
                }

        for item in dataset1:
            data = self._add_values(data, item.query, item.cq, item.answer, item.label, item.cq_id)

        dataset = pd.DataFrame(data)
        dataset = self._preprocess(dataset)

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset.iloc[idx]
        sample = self._to_dict(sample)
        return sample

    def _to_dict(self, sample):
        new_sample = {'query': sample['query'],
                      'query_len': sample['query_len'],
                      'cq': sample['cq'],
                      'answer': sample['answer'],
                      'cq_len': sample['cq_len'],
                      'answer_len': sample['answer_len'],
                      'label': sample['label'],
                      'original_query': sample['original_query'],
                      'cq_id': sample['cq_id']}
        return new_sample

    def _add_values(self, data, query, cq, answer, label, cq_id):
        data['query'].append(query)
        data['cq'].append(cq)
        data['answer'].append(answer)
        data['label'].append(label)
        data['original_query'].append(query)
        data['cq_id'].append(cq_id)
        return data

    def _preprocess(self, dataset):
        dataset['query'] = dataset['query'].apply(self._word2index)
        dataset['query_len'] = dataset['query'].apply(
            lambda x: self.max_query_len if len(x) > self.max_query_len else len(x))
        dataset['query'] = dataset['query'].apply(self._padding_query)
        dataset['query'] = dataset['query'].apply(self._to_tensor)
        dataset['query_len'] = dataset['query_len'].apply(self._to_tensor)

        dataset['cq'] = dataset['cq'].apply(self._word2index)
        dataset['cq_len'] = dataset['cq'].apply(lambda x: self.max_cq_len if len(x) > self.max_cq_len else len(x))
        dataset['cq'] = dataset['cq'].apply(self._padding_cq)
        dataset['cq'] = dataset['cq'].apply(self._to_tensor)
        dataset['cq_len'] = dataset['cq_len'].apply(self._to_tensor)

        dataset['answer'] = dataset['answer'].apply(self._word2index)
        dataset['answer_len'] = dataset['answer'].apply(
            lambda x: self.max_answer_len if len(x) > self.max_answer_len else len(x))
        dataset['answer'] = dataset['answer'].apply(self._padding_answer)
        dataset['answer'] = dataset['answer'].apply(self._to_tensor)
        dataset['answer_len'] = dataset['answer_len'].apply(self._to_tensor)

        return dataset

    # all data transformations
    def _clear_text(self, text):
        return text

    def _word2index(self, text):
        return [self.word2index[w].index for w in text.split() if w in self.word2index]

    def _to_tensor(self, value):
        return torch.tensor(value, dtype=torch.long)

    def _padding_query(self, values):
        return self._padding(values, self.max_query_len)

    def _padding_cq(self, values):
        return self._padding(values, self.max_cq_len)

    def _padding_answer(self, values):
        return self._padding(values, self.max_answer_len)

    def _padding(self, values, limit):
        padded = np.zeros((limit,), dtype=np.int64)
        if len(values) > limit:
            padded[:] = values[:limit]
        else:
            padded[:len(values)] = values
        return padded
