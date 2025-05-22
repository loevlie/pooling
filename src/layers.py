# PyTorch
import torch

class AttentionBasedPooling(torch.nn.Module):
    def __init__(self, in_features, temp=1.0):
        super().__init__()
        self.temp = temp
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x, lengths):
        
        _, hidden_dim = x.shape
        batch_size = len(lengths)
        
        attention_logits = self.mlp(x)
        attention_weights = torch.cat([torch.nn.functional.softmax(weights_i/self.temp, dim=0) for weights_i in torch.split(attention_logits, lengths)])
        attention_weighted_x = attention_weights*x
        context_vectors = torch.cat([torch.sum(attention_weighted_x_i, dim=0, keepdim=True) for attention_weighted_x_i in torch.split(attention_weighted_x, lengths)])
        
        return context_vectors, attention_weights
        
class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        _, hidden_dim = x.shape
        batch_size = len(lengths)
        
        out = torch.cat([torch.max(x_i, dim=0, keepdim=True).values for x_i in torch.split(x, lengths)])
        attention_weights = torch.cat([torch.argmax(x_i, dim=0, keepdim=True) for x_i in torch.split(x, lengths)])
        
        return out, attention_weights  
    
class MeanPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        device = x.device
        _, hidden_dim = x.shape
        batch_size = len(lengths)
        
        out = torch.cat([torch.mean(x_i, dim=0, keepdim=True) for x_i in torch.split(x, lengths)])
        attention_weights = torch.cat([torch.ones(length, device=device) / length for length in lengths])
        
        return out, attention_weights

class PositionalEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        
        device = x.device
        
        positional_encoding = torch.cat([torch.arange(1, length + 1, device=device) / length for length in lengths])
        Phi = torch.cat([positional_encoding.unsqueeze(1), x], dim=1)
        
        return Phi
