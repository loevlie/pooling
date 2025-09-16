# PyTorch
import torch
# Importing our custom module(s)
import utils

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
        
        attn_logits = self.mlp(x)
        attn_weights = torch.cat([torch.nn.functional.softmax(weights_i/self.temp, dim=0) for weights_i in torch.split(attn_logits, lengths)])
        attn_weighted_x = attn_weights*x
        context_vectors = torch.cat([torch.sum(attn_weighted_x_i, dim=0, keepdim=True) for attn_weighted_x_i in torch.split(attn_weighted_x, lengths)])
        
        return context_vectors, attn_weights
    
class Inflate(torch.nn.Module):
    def __init__(self, input_instances=3):
        super().__init__()
        self.input_instances = input_instances
        self.half_input_instances = int(input_instances/2)

    def forward(self, x, lengths):
        num_instances, hidden_dim = x.shape
        x = torch.cat([torch.nn.functional.pad(x_i, (0, 0, self.half_input_instances, self.half_input_instances), mode='constant', value=0.0) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + (2 * self.half_input_instances) for length in lengths)
        x = torch.cat([x_i.unfold(0, self.input_instances, 1) for x_i in torch.split(x, lengths)])
        x = x.reshape(num_instances, hidden_dim * self.input_instances)
        return x
    
class InstanceConv1d(torch.nn.Module):
    def __init__(self, in_features, kernel_size=3):
        super().__init__()
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=self.kernel_size, groups=self.in_features, padding="same")
        
    def forward(self, x, lengths):
        x = torch.cat([self.conv(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        return x
    
class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        out = torch.cat([torch.max(x_i, dim=0, keepdim=True).values for x_i in torch.split(x, lengths)])
        attn_weights = None
        
        return out, attn_weights  
    
class MeanPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        device = x.device        
        out = torch.cat([torch.mean(x_i, dim=0, keepdim=True) for x_i in torch.split(x, lengths)])
        attn_weights = torch.cat([torch.ones(size=(length, 1), device=device) / length for length in lengths])
        
        return out, attn_weights
    
class NormalPooling(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=2),
        )
        
    def get_x(self, y):
        assert y.dim() == 1, "get_x() expects 1D tensor, got shape {y.shape}"
        return torch.arange(1, len(y) + 1, device=y.device) / len(y)

    def calc_mean(self, y):
        assert y.dim() == 1, "calc_mean() expects 1D tensor, got shape {y.shape}"
        max_y = torch.max(y)
        w = torch.exp(y - max_y)
        x = self.get_x(y)
        return torch.sum(x * w) / torch.sum(w)

    def forward(self, x, lengths):
        
        means_and_stds = self.mlp(x)
        means = torch.stack([self.calc_mean(mean_i) for mean_i in torch.split(means_and_stds[:,0], lengths)])
        stds = torch.stack([torch.nn.functional.softplus(std_i.mean()) for std_i in torch.split(means_and_stds[:,1], lengths)])
        weights = torch.cat([utils.normal_pdf(self.get_x(mean_i), means[i], stds[i]) for i, mean_i in enumerate(torch.split(means_and_stds[:,0], lengths))])
        attn_weights = torch.cat([weights_i/(torch.sum(weights_i) + 1e-3) for weights_i in torch.split(weights, lengths)]).view(-1, 1)
        x = torch.stack([torch.sum(context_vector_i, dim=0) for context_vector_i in torch.split(attn_weights*x, lengths)])
        
        return x, attn_weights

class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, in_features, num_heads=1, temp=1.0):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.temp = temp
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)
    
    def forward(self, x, lengths):
        
        x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + 1 for length in lengths)
        # First transformer layer
        out, attn_weights = zip(*[self.self_attn(x_i / self.temp**0.5, x_i / self.temp**0.5, x_i) for x_i in torch.split(x, lengths)])
        x = torch.cat(out)
        # Get attention weight values from class token
        # Remove attention weight value for class token
        attn_weights = torch.cat([attn_weights_i[0,1:] for attn_weights_i in attn_weights])        
        # Get class token
        x = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        return x, attn_weights
    
class PPEG(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.proj1 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=3, groups=self.in_features, padding="same")
        self.proj2 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=5, groups=self.in_features, padding="same")
        self.proj3 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=7, groups=self.in_features, padding="same")

    def forward(self, x, lengths):
        
        # Split class tokens and features
        cls_token = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        x = torch.cat([x_i[1:,:] for x_i in torch.split(x, lengths)])
        lengths = tuple(length - 1 for length in lengths)
        # Reshape patch tokens into 1D image space
        # Use different sized convolutions
        proj1 = torch.cat([self.proj1(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        proj2 = torch.cat([self.proj2(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        proj3 = torch.cat([self.proj3(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        # Fuse together different spatial information
        x = x + proj1 + proj2 + proj3
        x = torch.cat([torch.cat((cls_token[i][None,:], x_i)) for i, x_i in enumerate(torch.split(x, lengths))])

        return x

class PositionalEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        
        device = x.device
        positional_encoding = torch.cat([torch.arange(1, length + 1, device=device) / length for length in lengths])
        Phi = torch.cat([positional_encoding.unsqueeze(1), x], dim=1)
        
        return Phi
    
class TransformerLayer(torch.nn.Module):
    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.norm = torch.nn.LayerNorm(normalized_shape=self.in_features)
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)
        
    def forward(self, x, lengths):
        out, attn_weights = zip(*[self.attn(x_i, x_i, x_i) for x_i in torch.split(self.norm(x), lengths)])
        x = x + torch.cat(out)
        return x, attn_weights
    
class TransformerBasedPooling(torch.nn.Module):
    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))
        self.layer1 = TransformerLayer(in_features=in_features, num_heads=self.num_heads)
        self.position_layer = PPEG(in_features=self.in_features)        
        self.layer2 = TransformerLayer(in_features=in_features, num_heads=self.num_heads)

    def forward(self, x, lengths):
        
        device = x.device
        # Concatenate class token
        x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + 1 for length in lengths)
        # Add positional encoding
        positional_encoding = torch.cat([torch.arange(0, length, device=device) / length for length in lengths])        
        x = x + positional_encoding[:,None]
        # First transformer layer
        x, _ = self.layer1(x, lengths)
        # Pyramid position encoding generator layer
        x = self.position_layer(x, lengths)
        # Second transformer layer
        x, attn_weights = self.layer2(x, lengths)
        # Get attention weight values from class token
        # Remove attention weight value for class token
        attn_weights = torch.cat([attn_weights_i[0,1:] for attn_weights_i in attn_weights])
        # Get class token
        x = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        
        return x, attn_weights


class ApproxSm(torch.nn.Module):
    """Approximate Smooth operator using iterative approach.
    Based on: https://github.com/Franblueee/SmMIL/blob/main/code/models/modules/Sm.py
    """
    def __init__(self, alpha=0.5, num_steps=1):
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps
        
        if isinstance(self.alpha, float):
            self.coef = (1.0/(1.0-self.alpha)-1)
        elif self.alpha == 'trainable':
            self.coef = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            raise ValueError("alpha must be float or 'trainable'")
    
    def forward(self, f, A_mat):
        """
        Apply approximate smoothing via iterative updates.
        
        Args:
            f: features tensor (batch_size, bag_size, d_dim)
            A_mat: adjacency matrix tensor (batch_size, bag_size, bag_size)
        
        Returns:
            g: smoothed features (batch_size, bag_size, d_dim)
        """
        # torch.sparse bug workaround
        recover_f = False
        if f.shape[2] == 1:
            recover_f = True
            f = torch.stack([f, f], dim=2).squeeze(-1)  # (batch_size, bag_size, 2)
        
        g = f
        alpha = 1.0 / (1.0 + self.coef)
        
        for _ in range(self.num_steps):
            g = (1.0 - alpha) * f + alpha * torch.bmm(A_mat, g)  # (batch_size, bag_size, d_dim)
        
        if recover_f:
            g = g[:, :, 0].unsqueeze(-1)  # (batch_size, bag_size, 1)
        
        return g


class ExactSm(torch.nn.Module):
    """Exact Smooth operator using matrix inversion.
    Based on: https://github.com/Franblueee/SmMIL/blob/main/code/models/modules/Sm.py
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
        if isinstance(self.alpha, float):
            self.coef = (1.0/(1.0-self.alpha)-1)
        elif self.alpha == 'trainable':
            self.coef = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            raise ValueError("alpha must be float or 'trainable'")
    
    def forward(self, f, A_mat):
        """
        Apply exact smoothing via linear system solution.
        
        Args:
            f: features tensor (batch_size, bag_size, d_dim)
            A_mat: adjacency matrix tensor (batch_size, bag_size, bag_size)
        
        Returns:
            g: smoothed features (batch_size, bag_size, d_dim)
        """
        batch_size = f.shape[0]
        bag_size = f.shape[1]
        
        # Create identity matrix
        id_mat = torch.eye(bag_size, device=A_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Construct system matrix: M = (1 + coef) * I - coef * A
        M = (1 + self.coef) * id_mat - self.coef * A_mat  # (batch_size, bag_size, bag_size)
        
        # Solve M * g = f for g
        g = torch.linalg.solve(M, f)  # (batch_size, bag_size, d_dim)
        
        return g


class SmMILPooling(torch.nn.Module):
    """SmMIL: Smooth Multiple Instance Learning pooling.
    Implements attention-based pooling with optional smoothing via graph convolution.
    Based on: https://github.com/Franblueee/SmMIL
    """
    def __init__(self, in_features, temp=1.0, sm_alpha=0.5, sm_mode='approx',
                 sm_steps=1, sm_when='late'):
        super().__init__()
        self.in_features = in_features
        self.temp = temp
        self.sm_alpha = sm_alpha
        self.sm_mode = sm_mode
        self.sm_steps = sm_steps
        self.sm_when = sm_when  # 'early', 'mid', 'late', or None

        # Attention mechanism similar to ABMIL
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=1),
        )

        # Smooth operator
        if self.sm_when is not None:
            if self.sm_mode == 'approx':
                self.sm = ApproxSm(alpha=sm_alpha, num_steps=sm_steps)
            elif self.sm_mode == 'exact':
                self.sm = ExactSm(alpha=sm_alpha)
            else:
                raise ValueError(f"Unknown sm_mode: {sm_mode}. Use 'approx' or 'exact'")
        else:
            self.sm = None

    def compute_adjacency_matrix(self, length, device):
        """Create adjacency matrix for local connectivity (neighboring instances).
        This creates a normalized adjacency matrix where each instance is connected
        to its immediate neighbors.
        """
        if length == 1:
            return torch.ones((1, 1), device=device)

        # Create adjacency matrix with connections to immediate neighbors
        A = torch.zeros((length, length), device=device)

        # Add connections to neighbors
        for i in range(length):
            # Self-connection
            A[i, i] = 1.0
            # Connection to previous neighbor
            if i > 0:
                A[i, i-1] = 1.0
            # Connection to next neighbor
            if i < length - 1:
                A[i, i+1] = 1.0

        # Row-normalize the adjacency matrix
        row_sums = A.sum(dim=1, keepdim=True)
        A = A / row_sums

        return A

    def forward(self, x, lengths):
        device = x.device

        # Early smoothing (on features before attention)
        if self.sm and self.sm_when == 'early':
            smoothed_x = []
            for x_i in torch.split(x, lengths):
                length = x_i.shape[0]
                A = self.compute_adjacency_matrix(length, device)
                # Add batch dimension and apply smoothing
                x_i_smooth = self.sm(x_i.unsqueeze(0), A.unsqueeze(0)).squeeze(0)
                smoothed_x.append(x_i_smooth)
            x = torch.cat(smoothed_x)

        # Compute attention logits
        attn_logits = self.attention(x)

        # Mid smoothing (on attention logits before softmax)
        if self.sm and self.sm_when == 'mid':
            smoothed_logits = []
            for logits_i in torch.split(attn_logits, lengths):
                length = logits_i.shape[0]
                A = self.compute_adjacency_matrix(length, device)
                # Add batch dimension and apply smoothing
                logits_i_smooth = self.sm(logits_i.unsqueeze(0), A.unsqueeze(0)).squeeze(0)
                smoothed_logits.append(logits_i_smooth)
            attn_logits = torch.cat(smoothed_logits)

        # Apply softmax to get attention weights
        attn_weights = torch.cat([
            torch.nn.functional.softmax(weights_i/self.temp, dim=0)
            for weights_i in torch.split(attn_logits, lengths)
        ])

        # Late smoothing (on attention weights after softmax)
        if self.sm and self.sm_when == 'late':
            smoothed_weights = []
            for weights_i in torch.split(attn_weights, lengths):
                length = weights_i.shape[0]
                A = self.compute_adjacency_matrix(length, device)
                # Add batch dimension and apply smoothing
                weights_i_smooth = self.sm(weights_i.unsqueeze(0), A.unsqueeze(0)).squeeze(0)
                # Renormalize after smoothing to ensure valid probability distribution
                weights_i_smooth = weights_i_smooth / (weights_i_smooth.sum() + 1e-8)
                smoothed_weights.append(weights_i_smooth)
            attn_weights = torch.cat(smoothed_weights)

        # Apply attention weights to features
        attn_weighted_x = attn_weights * x

        # Aggregate to get bag representation
        context_vectors = torch.cat([
            torch.sum(attn_weighted_x_i, dim=0, keepdim=True)
            for attn_weighted_x_i in torch.split(attn_weighted_x, lengths)
        ])

        return context_vectors, attn_weights


class MaskedTransformerLayer(torch.nn.Module):
    def __init__(self, in_features, num_heads=8, local_window=3):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.local_window = local_window
        self.norm = torch.nn.LayerNorm(normalized_shape=self.in_features)
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)

    def create_local_mask(self, seq_len, device):
        """Create mask that forces local connections within a window."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, start:end] = 0.0

        return mask

    def forward(self, x, lengths):
        out_list = []
        attn_weights_list = []

        for x_i in torch.split(self.norm(x), lengths):
            seq_len = x_i.shape[0]
            mask = self.create_local_mask(seq_len, x_i.device)

            out_i, attn_weights_i = self.attn(x_i, x_i, x_i, attn_mask=mask)
            out_list.append(out_i)
            attn_weights_list.append(attn_weights_i)

        x = x + torch.cat(out_list)
        return x, attn_weights_list


def create_sinusoidal_positional_encoding(seq_len, d_model, device):
    """Create sinusoidal positional encoding"""
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                         -(torch.log(torch.tensor(10000.0)) / d_model))

    pos_encoding = torch.zeros(seq_len, d_model, device=device)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    return pos_encoding


class MultiLayerTransformer(torch.nn.Module):
    def __init__(self, in_features, num_layers=2, num_heads=8, local_window=5, use_cls_token=True):
        super().__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        # Ensure num_heads is compatible with in_features
        self.num_heads = min(num_heads, in_features) if in_features % num_heads != 0 else num_heads
        if in_features % self.num_heads != 0:
            self.num_heads = 1  # Fall back to single head if not divisible
        self.local_window = local_window
        self.use_cls_token = use_cls_token

        if self.use_cls_token:
            self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))

        # Create transformer layers - first one is masked, rest are standard
        self.layers = torch.nn.ModuleList()
        self.layers.append(MaskedTransformerLayer(
            in_features=in_features,
            num_heads=self.num_heads,
            local_window=local_window
        ))
        for _ in range(num_layers - 1):
            self.layers.append(TransformerLayer(
                in_features=in_features,
                num_heads=self.num_heads
            ))

        # Layer norms for stability
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(normalized_shape=self.in_features)
            for _ in range(num_layers)
        ])

        # Final attention and norm for CLS token
        if self.use_cls_token:
            self.final_attn = torch.nn.MultiheadAttention(
                embed_dim=self.in_features,
                num_heads=self.num_heads
            )
            self.final_norm = torch.nn.LayerNorm(normalized_shape=self.in_features)

        # Store attention maps for analysis
        self.intermediate_attentions = []
        self.cls_attention = None

    def forward(self, x, lengths):
        device = x.device
        self.intermediate_attentions = []

        # Add sinusoidal positional encoding to input
        x_with_pos = []
        for x_i in torch.split(x, lengths):
            seq_len = x_i.shape[0]
            pos_encoding = create_sinusoidal_positional_encoding(seq_len, self.in_features, device)
            x_with_pos.append(x_i + pos_encoding)
        x = torch.cat(x_with_pos)

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Apply layer norm before attention
            x_norm = torch.cat([self.layer_norms[i](x_i) for x_i in torch.split(x, lengths)])
            x_out, layer_attn_weights = layer(x_norm, lengths)
            x = x_out

            # Store attention weights for analysis
            avg_attn_weights = []
            for attn_i in layer_attn_weights:
                if attn_i.dim() == 3:  # Multi-head attention
                    avg_attn_i = attn_i.mean(dim=0)
                else:
                    avg_attn_i = attn_i
                avg_attn_weights.append(avg_attn_i)
            self.intermediate_attentions.append(avg_attn_weights)

        if self.use_cls_token:
            # Add CLS token
            x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
            lengths_with_cls = tuple(length + 1 for length in lengths)

            # Add positional encoding with CLS
            x_with_cls_pos = []
            for x_i in torch.split(x, lengths_with_cls):
                seq_len = x_i.shape[0]
                pos_encoding = create_sinusoidal_positional_encoding(seq_len, self.in_features, device)
                x_with_cls_pos.append(x_i + pos_encoding)
            x = torch.cat(x_with_cls_pos)

            # Final attention layer
            out, attn_weights_list = zip(*[
                self.final_attn(self.final_norm(x_i), self.final_norm(x_i), self.final_norm(x_i))
                for x_i in torch.split(x, lengths_with_cls)
            ])
            x = x + torch.cat(out)

            # Store CLS attention
            self.cls_attention = []
            for attn_i in attn_weights_list:
                if attn_i.dim() == 3:  # Multi-head
                    cls_attn = attn_i.mean(dim=0)[0, 1:]  # Average heads, CLS row, exclude self
                else:
                    cls_attn = attn_i[0, 1:]
                self.cls_attention.append(cls_attn)

            # Extract final attention weights for output
            final_attn_weights = torch.cat([
                attn_weights_i.mean(dim=0)[0, 1:] if attn_weights_i.dim() == 3 else attn_weights_i[0, 1:]
                for attn_weights_i in attn_weights_list
            ])

            # Get CLS token representation
            x = torch.stack([x_i[0, :] for x_i in torch.split(x, lengths_with_cls)])
        else:
            # Mean pooling if no CLS token
            x = torch.cat([torch.mean(x_i, dim=0, keepdim=True) for x_i in torch.split(x, lengths)])
            final_attn_weights = torch.cat([torch.ones(length, device=device) / length for length in lengths])

        return x, final_attn_weights.unsqueeze(1) if final_attn_weights.dim() == 1 else final_attn_weights

    def compute_attention_regularization(self, lengths, labels=None):
        """Light attention regularization to prevent extreme behaviors"""
        if self.cls_attention is None or labels is None:
            return torch.tensor(0.0, device=self.cls_token.device if hasattr(self, 'cls_token') else 'cpu')

        reg_loss = 0.0
        for i, cls_attn in enumerate(self.cls_attention):
            is_positive = labels[i].item() > 0.5
            seq_len = len(cls_attn)
            entropy = -(cls_attn * torch.log(cls_attn + 1e-8)).sum()
            max_entropy = torch.log(torch.tensor(float(seq_len)))
            normalized_entropy = entropy / max_entropy

            if is_positive:
                # For positive: light penalty if too uniform (entropy > 0.9)
                if normalized_entropy > 0.9:
                    reg_loss += (normalized_entropy - 0.9) ** 2 * 0.01
            else:
                # For negative: light penalty if too focused (entropy < 0.4)
                if normalized_entropy < 0.4:
                    reg_loss += (0.4 - normalized_entropy) ** 2 * 0.01

        return reg_loss


def entropy_regularization_loss(attn_weights, labels, lengths, entropy_weight=0.1):
    """
    Entropy regularization loss that encourages:
    - Peaky attention weights for positive labels (low entropy)
    - Flat attention weights for negative labels (high entropy)

    Args:
        attn_weights: attention weights tensor
        labels: binary labels (0 or 1)
        lengths: sequence lengths for each sample
        entropy_weight: weight for entropy regularization term

    Returns:
        entropy_loss: regularization loss
    """
    entropy_losses = []

    for i, (weights_i, label) in enumerate(zip(torch.split(attn_weights, lengths), labels)):
        # Compute entropy: -sum(p * log(p))
        eps = 1e-8
        entropy = -torch.sum(weights_i * torch.log(weights_i + eps))

        if label > 0.5:  # Positive label - encourage low entropy (peaky)
            entropy_loss = entropy
        else:  # Negative label - encourage high entropy (flat)
            # Use negative entropy to encourage high entropy
            max_entropy = torch.log(torch.tensor(len(weights_i), dtype=torch.float32, device=weights_i.device))
            entropy_loss = max_entropy - entropy

        entropy_losses.append(entropy_loss)

    total_entropy_loss = torch.stack(entropy_losses).mean()
    return entropy_weight * total_entropy_loss
