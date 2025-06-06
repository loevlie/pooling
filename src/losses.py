# PyTorch
import torch
# Importing our custom module(s)
import utils

class ERMLoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):
        
        nll = self.criterion(logits, labels)
        
        return {'loss': nll, 'nll': nll}
    
class L1Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * torch.abs(params).sum()
        
        return {'loss': nll + penalty, 'nll': nll}
    
class L2Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * (params**2).sum()
        
        return {'loss': nll + penalty, 'nll': nll}

class GuidedAttentionL1Loss(torch.nn.Module):
    def __init__(self, alpha, beta, criterion=torch.nn.CrossEntropyLoss(), max_std=1000.0, min_std=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.max_std = max_std
        self.min_std = min_std

    def get_x(self, y):
        return torch.arange(1, len(y) + 1, device=y.device) / len(y)

    def calc_mean(self, y):
        x = self.get_x(y)
        return torch.sum(x * y) / torch.sum(y)

    def calc_std(self, y):
        x = self.get_x(y)
        mean = torch.sum(x * y) / torch.sum(y)
        variance = torch.sum((x - mean)**2) / torch.sum(y)
        return torch.sqrt(variance)

    def forward(self, logits, labels, **kwargs):

        attention_weights = kwargs["attention_weights"]
        device = attention_weights.device
        lengths = kwargs["lengths"]
        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        
        with torch.no_grad():
            js = [self.get_x(attention_weights_i) for attention_weights_i in torch.split(attention_weights, lengths)]
            means = [self.calc_mean(attention_weights_i) for attention_weights_i in torch.split(attention_weights, lengths)]
            #stds = [self.calc_std(attention_weights_i) for attention_weights_i in torch.split(attention_weights, lengths)]
            ideal_stds = [self.min_std/length if label == 1.0 else self.max_std/length for label, length in zip(labels, lengths)]
            r_hats = torch.cat([utils.normal_pdf(j, mean, ideal_std) for j, mean, ideal_std in zip(js, means, ideal_stds)])
            rs = torch.cat([r_hat/(r_hat.sum() + 1e-6) for r_hat in torch.split(r_hats, lengths)])
            
        penalty = (self.alpha/2) * torch.abs(params).sum()
        attention_weights_penalty = (self.beta/2) * torch.stack([(diff**2).mean() for diff in torch.split(attention_weights-rs, lengths)]).mean()
        
        return {'loss': nll + penalty + attention_weights_penalty, 'nll': nll}
    