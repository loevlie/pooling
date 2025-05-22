# PyTorch
import torch
# Importing our custom module(s)
import layers

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling="max", use_pos_embedding=False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
                
        self.use_pos_embedding = use_pos_embedding
        if self.use_pos_embedding:
            self.pos_embedding = layers.PositionalEmbeddingLayer()
            
        self.hidden_dim = self.in_features + 1 if self.use_pos_embedding else self.in_features
        self.clf = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.out_features, bias=True)
            
        self.pooling = pooling
        if self.pooling == "max":
            self.pool = layers.MaxPooling()
        elif self.pooling == "mean":
            self.pool = layers.MeanPooling()
        elif self.pooling == "attention":
            self.pool = layers.AttentionBasedPooling(in_features=self.out_features)
        else:
            raise NotImplementedError(f"The specified pooling operation \"{self.pooling}\" is not implemented.")


    def forward(self, x, lengths):
        
        if self.use_pos_embedding:
            x = self.pos_embedding(x, lengths)
            
        x = self.clf(x)
            
        x = self.pool(x, lengths)
        
        return x
    