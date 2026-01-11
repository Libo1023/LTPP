import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math


class ConvTransE(torch.nn.Module):
    """RE-GCN ConvTransE decoder for entity prediction"""
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, 
                 feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        # 2 channels: entity and relation
        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, triplets, mode="train"):
        e1_embedded_all = F.tanh(embedding)
        device = e1_embedded_all.device
        triplets = triplets.to(device)
        
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        
        # Stack entity and relation
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        
        # Score against all entities
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        return x