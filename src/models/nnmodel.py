
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class DocScorer(nn.Module):
    '''
    Document scoring model for a given query document pair
    Currently restricting to a 2-layer MLP with ReLU activation
    '''
    def __init__(self, doc_feat_dim, hidden_dim1, hidden_dim2, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        self.doc_feat_dim = doc_feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # Series of linear layers and ReLU activation functions
        self.mlp_layers = nn.Sequential(
            nn.Linear(doc_feat_dim, hidden_dim1),  
            nn.ReLU(),
            #nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2), 
            nn.ReLU(),
            #nn.Tanh(),
            nn.Linear(hidden_dim2, 1)
            # log-scores are sufficient, so no need to do softplus
            #nn.Softplus()
        )
        #torch.nn.init.xavier_uniform_(self.mlp_layers[0].weight)
        #torch.nn.init.xavier_uniform_(self.mlp_layers[0].weight)
        # for m in self.mlp_layers.children():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight)
        #         m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.mlp_layers(x)
        return x
    

class Logger(nn.Module):
    '''
    Document scoring model for a given query document pair
    Currently restricting to a 2-layer MLP with ReLU activation
    '''
    def __init__(self, doc_feat_dim, hidden_dim1, hidden_dim2, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        self.doc_feat_dim = doc_feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # Series of linear layers and ReLU activation functions
        self.mlp_layers = nn.Sequential(
            nn.Linear(doc_feat_dim, hidden_dim1),  
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), 
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
            # log-scores are sufficient, so no need to do softplus
            #nn.Softplus()
        )

    def forward(self, x):
        x = self.mlp_layers(x)
        return x


# class DocScorer(nn.Module):
#     '''
#     Document scoring model for a given query document pair
#     Currently restricting to a 2-layer MLP with ReLU activation
#     '''
#     def __init__(self, doc_feat_dim, hidden_dim1, hidden_dim2, **kwargs):
#         super().__init__()
#         # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
#         self.doc_feat_dim = doc_feat_dim
#         self.hidden_dim1 = hidden_dim1
#         self.hidden_dim2 = hidden_dim2
#         m = nn.Dropout(p=0.5)

#         # Series of linear layers and ReLU activation functions
#         self.mlp_layers = nn.Sequential(
#             nn.Linear(doc_feat_dim, hidden_dim1),  
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Linear(hidden_dim1, 1)
#             # log-scores are sufficient, so no need to do softplus
#             #nn.Softplus()
#         )

#     def forward(self, x):
#         x = self.mlp_layers(x)
#         return x