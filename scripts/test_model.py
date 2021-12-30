import argparse
import csv

import torch
from sklearn.metrics import classification_report
from torch import optim, nn

import dataset_setup as ds
import logging
import numpy as np

from gensim.scripts.glove2word2vec import glove2word2vec
import gensim

import sys

print(torch.cuda.is_available())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', help='File path to embeddings', required=True)
    parser.add_argument('--models', help='File path to models', required=True)
    parser.add_argument('--dataset', help='File path to dataset', required=True)
    parser.add_argument('--templates', help='File path to template questions', required=True)
    parser.add_argument('--test-dataset', help='File path to test dataset', required=True)
    parser.add_argument('--all-dataset', help='File path to all dataset', required=True)
    parser.add_argument('--results', help='File path to results', required=True)
    parser.add_argument('--n-epochs', help='Number of epochs', default=10, type=int)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
    parser.add_argument('--filler-size', help='Filler size', default='2 3')
    parser.add_argument('--test', help='Test size', type=int, default=20)
    return parser.parse_args()


args = parse_args()
print(args)

dataset_path = args.dataset
embedding_path = args.embeddings
models_path = args.models
outputs_path = args.results
test_size = args.test

N_EPOCH = args.n_epochs
FILTER_SIZES = [int(x) for x in args.filler_size.split()]
device = torch.device(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

template_path = args.templates
all_dataset_path = args.all_dataset
test_dataset_path = args.test_dataset
vector_path = embedding_path

sys.path.insert(0, dataset_path)
sys.path.insert(0, embedding_path)
sys.path.insert(0, models_path)
sys.path.insert(0, outputs_path)
print(sys.path)

from model_CNN_blstm import InfoSeekingModel_CNN_BiLSTM


def read_w2v_model(path_in):
    path_out = '/'.join(path_in.split('/')[:-1]) + '/w2v_vectors.txt'
    glove2word2vec(path_in, path_out)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path_out)
    if '<PAD>' not in w2v_model or w2v_model.vocab['<PAD>'].index != 0:
        raise ValueError('No <PAD> token in embeddings! Provide embeddings with <PAD> token.')
    return w2v_model


w2v_model = read_w2v_model(vector_path)
train_loader, test_loader, classes, train_dataset, test_dataset = ds.get_dataset(w2v_model.vocab, template_path,
                                                                                 all_dataset_path, test_dataset_path,
                                                                                 test_size)

print(len(train_loader))
print(len(test_loader))
print("device", device)
print(classes)

logging.info('Running on {0}'.format(device))

# ===================================================================================================== #


net = InfoSeekingModel_CNN_BiLSTM(w2v_model.vectors, FILTER_SIZES)
net.to(device)

criterion = nn.BCELoss(reduction='mean')
criterion = criterion.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)


def binary_accuracy(y_pred, y_test):
    rounded_preds = torch.round(y_pred)
    correct = (rounded_preds == y_test).float()
    acc = correct.sum() / len(correct)
    acc = torch.round(acc * 100)
    return acc


def evaluate(model, iterator, save_results=False):
    model.eval()
    with torch.no_grad():
        total_preds = []
        total_labels = []

        original_queries = []
        labels_ind = []
        org_labels = []
        pred_labels = []

        for data in iterator:
            query = data['query']
            cq = data['cq']
            answer = data['answer']
            labels = data['label']

            if device.type != 'cpu':
                query = query.to(device)
                cq = cq.to(device)
                answer = answer.to(device)
                labels = labels.to(device)

            outputs = model(query, cq, answer).squeeze(1)

            if device.type != 'cpu':
                if len(total_preds) == 0:
                    total_preds = torch.round(outputs).cpu().detach().numpy()
                    total_labels = labels.cpu().detach().numpy()
                else:
                    total_preds = np.append(total_preds, torch.round(outputs).cpu().detach().numpy())
                    total_labels = np.append(total_labels, labels.cpu().detach().numpy())
            else:
                if len(total_preds) == 0:
                    total_preds = torch.round(outputs).detach().numpy()
                    total_labels = labels.detach().numpy()
                else:
                    total_preds = np.append(total_preds, torch.round(outputs).detach().numpy())
                    total_labels = np.append(total_labels, labels.detach().numpy())

            if save_results:
                original_query = data['original_query']
                cq_id = data['cq_id']
                original_queries.extend(original_query)
                if device.type != 'cpu':
                    labels_ind.extend(cq_id.cpu().detach().numpy())
                    org_labels.extend((list(labels.cpu().detach().numpy())))
                    pred_labels.extend((list(outputs.cpu().detach().numpy())))
                else:
                    labels_ind.extend(cq_id.detach().numpy())
                    org_labels.extend((list(labels.detach().numpy())))
                    pred_labels.extend((list(outputs.detach().numpy())))
        if save_results:
            print(len(original_queries))

            dict_for_save = {}
            dict_for_save_original = {}
            for i in range(len(original_queries)):
                if original_queries[i] not in dict_for_save:
                    dict_for_save[original_queries[i]] = {}
                    dict_for_save_original[original_queries[i]] = {}
                    for j in range(1, len(classes)):
                        dict_for_save[original_queries[i]]['T' + str(j)] = {}
                        dict_for_save_original[original_queries[i]]['T' + str(j)] = {}

                dict_for_save[original_queries[i]]['T' + str(labels_ind[i])] = pred_labels[i]
                dict_for_save_original[original_queries[i]]['T' + str(labels_ind[i])] = org_labels[i]

            fields = []
            fields.append('query')
            fields = fields + classes

            print(len(dict_for_save))
            with open(outputs_path + "/test_original_output.csv", "w") as f:
                def get_value(field, k):
                    if field == 'query':
                        return k
                    return 0

                w = csv.DictWriter(f, fields)
                w.writeheader()
                for k in dict_for_save:
                    w.writerow({field: dict_for_save_original[k].get(field) or get_value(field, k) for field in fields})

            with open(outputs_path + "/test_predicted_output.csv", "w") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                for k in dict_for_save:
                    w.writerow({field: dict_for_save[k].get(field) or k for field in fields})

            with open(outputs_path + "/test_rank_1_output.csv", "w") as f:
                counter_ranking = {}
                for i in range(17):
                    counter_ranking['T' + str(i)] = 0
                w = csv.writer(f, delimiter=',')
                w.writerow(['query', 'r@1 template', 'value'])
                for item in dict_for_save:
                    a = 0.0
                    b = 'T0'
                    for temp in dict_for_save[item]:
                        if float(dict_for_save[item][temp]) > a:
                            a = float(dict_for_save[item][temp])
                            b = temp
                    w.writerow([item, b, a])
                    counter_ranking[b] = counter_ranking[b] + 1
                print(counter_ranking)
    print(classification_report(total_labels, total_preds))


for epoch in range(N_EPOCH):
    print('Epoch {0}/{1}'.format((epoch + 1), N_EPOCH))
    epoch_loss = 0.0
    epoch_acc = 0.0

    net.train()

    for data in train_loader:
        query = data['query']
        cq = data['cq']
        answer = data['answer']
        labels = data['label']

        if device.type != 'cpu':
            query = query.to(device)
            cq = cq.to(device)
            answer = answer.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(query, cq, answer).squeeze(1)
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
        # loss = weighted_binary_cross_entropy(outputs.to(torch.float32), labels.to(torch.float32), [1.0, 6.0])
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        acc = binary_accuracy(outputs, labels)
        epoch_acc += acc.item()

    train_loss, train_acc = epoch_loss / len(train_loader), epoch_acc / len(train_loader)
    print(f'\t Train Loss: {train_loss:.3f} |  Train Acc: {train_acc:.2f}%')
    evaluate(net, test_loader, save_results=False)
evaluate(net, test_loader, save_results=True)
