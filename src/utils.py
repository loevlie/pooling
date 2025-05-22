# PyTorch
import torch

def generate_data(N, delta=1.0, deltaS=3, h=0, p_y1=0.5, S=23, seed=42):
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    X = torch.randn(S*N, 768, generator=g)
    lengths = (S,) * N
    u = torch.cat([torch.randint(0, S-(deltaS-1), size=(1,), generator=g) for length in lengths])
    y = torch.bernoulli(p_y1 * torch.ones(size=(N,)), generator=g).to(torch.int)
    
    for i, X_i in enumerate(torch.split(X, lengths)):
        if y[i] == 1:
            X_i[u[i]:u[i]+deltaS,h] += delta
            
    return X, lengths, u, y

def normal_pdf(x, mu=0.0, sigma=1.0):
    
    return 1 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def proba_y1_given_h(h, delta=1.0, deltaS=3, p_y1=0.5):
    
    S_i, = h.shape
    
    u_i = S_i - (deltaS-1)
    p_y0 = 1.0 - p_y1
    
    p_h_y0 = torch.prod(torch.tensor([normal_pdf(h[s]) for s in range(S_i)])) * p_y0
    
    p_u = (1/u_i) * torch.ones(size=(u_i,))
    p_h_u_y1 = torch.prod(torch.tensor([[normal_pdf(h[s], delta) if s >= u and s < (u + deltaS) else normal_pdf(h[s]) for u in range(u_i)] for s in range(S_i)]), dim=0)
    p_h_y1 = torch.sum(p_h_u_y1 * p_u) * p_y1
    
    p_y1_h = p_h_y1 / (p_h_y0 + p_h_y1)
    
    return p_y1_h
