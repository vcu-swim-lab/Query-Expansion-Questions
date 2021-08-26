import torch.nn as nn
import torch
import torch.nn.functional as F


class InfoSeekingModel_CNN_BiLSTM(nn.Module):
    def __init__(self, weights, filter_sizes, n_filters=256, hidden_dim=256):
        super(InfoSeekingModel_CNN_BiLSTM, self).__init__()

        # layer 1 - embeddings
        weights = torch.FloatTensor(weights)
        self.emb_layer = nn.Embedding.from_pretrained(weights)
        self.emb_layer.requires_grad = False

        # layer 2 - LSTM and CNN
        self.blstm_layer = nn.LSTM(input_size=self.emb_layer.embedding_dim, hidden_size=hidden_dim, batch_first=True,
                                   bidirectional=True)
        self.cnn_layer = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, self.emb_layer.embedding_dim)) for fs in
             filter_sizes])
        self.cnn_fc1 = nn.Linear(len(filter_sizes) * n_filters, self.emb_layer.embedding_dim)
        self.cnn_fc2 = nn.Linear(self.emb_layer.embedding_dim, n_filters)

        # layer 3 - dense layer
        self.layer1 = nn.Linear(4 * hidden_dim, 2 * self.emb_layer.embedding_dim)
        self.batchnorm1 = nn.BatchNorm1d(2 * self.emb_layer.embedding_dim)

        self.layer2 = nn.Linear(2 * self.emb_layer.embedding_dim, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.layer3 = nn.Linear(256, 128)
        self.batchnorm3 = nn.BatchNorm1d(128)

        self.layerout = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.act1 = nn.Sigmoid()

    def forward(self, query, question, answers):
        question_emb_out = self.emb_layer(question)
        question_lstm_out, test = self.blstm_layer(question_emb_out)

        answers_emb_out = self.emb_layer(answers).unsqueeze(1)
        answers_cnn_out = [F.relu(conv(answers_emb_out)).squeeze(3) for conv in self.cnn_layer]
        answers_cnn_out = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in answers_cnn_out]
        answers_cnn_out = self.cnn_fc1(torch.cat(answers_cnn_out, dim=1))
        answers_cnn_out = self.cnn_fc2(answers_cnn_out)

        query_emb_out = self.emb_layer(query).unsqueeze(1)
        query_cnn_out = [F.relu(conv(query_emb_out)).squeeze(3) for conv in self.cnn_layer]
        query_cnn_out = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in query_cnn_out]
        query_cnn_out = self.cnn_fc1(torch.cat(query_cnn_out, dim=1))
        query_cnn_out = self.cnn_fc2(query_cnn_out)

        x = torch.cat((query_cnn_out, question_lstm_out.mean(dim=1), answers_cnn_out), 1)

        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layerout(x)
        x = self.act1(x)
        return x
