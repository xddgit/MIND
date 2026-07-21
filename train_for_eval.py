# Model definition.
import os
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    class PyTorchModelHubMixin:
        pass
import sys
from einops import rearrange
import numpy as np

PUBLIC_ROOT = os.path.dirname(os.path.abspath(__file__))
# ==============================================================================
# Helper for Debugging
# ==============================================================================
GLOBAL_DEBUG_ENABLED = False

def debug_log(msg):
    if not GLOBAL_DEBUG_ENABLED:
        return

    # Print with timestamp and rank info to stderr to ensure it flushes immediately
    rank = int(os.environ.get("RANK", 0))
    try:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    except NameError:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
    
    sys.stderr.write(f"[{timestamp}] [Rank {rank}] {msg}\n")
    sys.stderr.flush()
# ==============================================================================
# 0. Evaluation entry point
# ==============================================================================
# The sampler module handles deterministic token generation and GigaTok decoding.
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
import os
os.environ.setdefault('HF_HOME', os.path.join(PUBLIC_ROOT, 'hf_cache'))
# Diffusion SDE Coefficients
K1 = 6e-1
K2 = 1
Embed_Dim = 16
Embed_sub = 4
vocab_size = 16384

# ==============================================================================
# 2. Modeling Components
# ==============================================================================
debug_log("Start: Loading Modeling Components")
import sys 
sys.path.insert(0, os.path.join(PUBLIC_ROOT, 'model'))
try:
    import rotary 
    from fused_add_dropout_scale import (
        bias_dropout_add_scale_fused_train, 
        bias_dropout_add_scale_fused_inference, 
        modulate_fused,
    )
    debug_log("Success: Imported custom CUDA kernels")
except ImportError:
    debug_log("Warning: Custom CUDA kernels not found. Using Python fallback implementation.")

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]

class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        cos, sin = rotary_cos_sin
        qkv = rotary.apply_rotary_pos_emb(qkv, cos, sin)

        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        # input shape of the attention should be [b h s d]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # x = rearrange(x, 'b s (h d)')
        x = rearrange(x, 'b h s d -> b s (h d)')

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x
        
class GeometryAwareEmbedding(nn.Module):
    def __init__(self, vocab_dim, num_top=0, p_threshold=0.04):
        super().__init__()
        # Internal dimension is compressed 
        self.internal_dim = Embed_Dim
        self.embedding = nn.Parameter(torch.randn(vocab_dim, self.internal_dim))
        
        # Xavier Normal Initialization
        nn.init.xavier_normal_(self.embedding, gain=0.02)

    def Mapping(self,X,idx_y,k_):
        X_, idx_x = X.topk(dim=2,k=k_)
        X_ = F.softmax(X_, dim=-1)

        idx_y = idx_y
        embedding_param = self.embedding_norm()
        
        X_ = X_ / (X_.sum(dim=2,keepdim=True) + 1e-5)
        emb_x = embedding_param[idx_x]
        emb_x = emb_x*X_[:,:,:,None]
        emb_x = emb_x.sum(dim = 2)
        emb_x = emb_x / (emb_x.norm(dim=2,keepdim=True)+1e-6)
        emb_y = embedding_param[idx_y]
        return emb_x, emb_y
    
    def embedding_norm(self):
        # 1. Center
        emb = self.embedding
        # 2. Reshape to pairs for L2 norm
        N, C = emb.shape
        emb = emb.view(N, C // Embed_sub, Embed_sub) #(N,1,C)  #
        # 3. Normalize pairs and scale
        emb_norm = torch.norm(emb, dim=2, keepdim=True)#* math.sqrt(C // 2)
        emb = emb / (emb_norm + 1e-6) 
        # 4. Flatten
        return emb.view(N, C)

    def forward(self, idx_x):
        embedding_param = self.embedding_norm()
        result = F.embedding(idx_x.squeeze(-1), embedding_param)
        return result

class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        x = self.linear(x)
        return x
class HighFreqFeatureMapper(nn.Module):
    """
    Frequency-feature mapping used instead of a single linear input projection.
    The local dimensions avoid coupling to module-level model settings.
    """
    def __init__(self, 
                 in_features,
                 out_features,
                 map_dim=256,
                 freq_scale=10.0,
                 is_trainable=True):
        super().__init__()

        self.freq_projector = nn.Linear(in_features, map_dim, bias=False)
        
        nn.init.normal_(self.freq_projector.weight, mean=0.0, std=freq_scale)
        
        if not is_trainable:
            self.freq_projector.weight.requires_grad = False

        expanded_dim = map_dim * 2

        mid_dim = max(expanded_dim, out_features * 2)
        
        self.feature_mlp = nn.Sequential(
            # Layer 1
            nn.Linear(expanded_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            
            # Layer 2
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            
            nn.Linear(mid_dim, out_features)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [Batch, ..., in_features] 
        projected = self.freq_projector(x) * 2 * np.pi        
        encoded = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)        
        output = self.feature_mlp(encoded)
        
        return output

class DiT(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_size: int,
            n_heads: int,
            cond_dim: int,
            dropout: float,
            n_blocks: int,
            num_classes: int,
            class_dropout_prob: float,
            **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_embed = GeometryAwareEmbedding(input_dim, num_top=3000, p_threshold=0.1)
        
        self.Embedding_map = HighFreqFeatureMapper(
            in_features=Embed_Dim,   # 16
            out_features=hidden_size, # 768
            map_dim=256,             
            freq_scale=10.0,         
            is_trainable=True        
        )        
        
        self.sigma_map = TimestepEmbedder(cond_dim)
        
        self.y_embedder = LabelEmbedder(num_classes, cond_dim, class_dropout_prob)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) 
            for _ in range(n_blocks)
        ])

        self.output_layer = DiTFinalLayer(hidden_size, output_dim, cond_dim)
        self.kwargs = kwargs
        initial_tensor = torch.tensor(1.0).reshape(1,1)
        self.scalar = nn.Parameter(initial_tensor)

        # ============================================================
        # ============================================================
        self.embed_proj = nn.Sequential(
            nn.Linear(1024, hidden_size//2, bias=False),
            nn.LayerNorm(hidden_size//2),
            nn.SiLU(),
            nn.Linear(hidden_size//2, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size*2, bias=False),
            nn.LayerNorm(hidden_size*2),
            nn.SiLU(),
            nn.Linear(hidden_size*2, hidden_size*2, bias=False),
            nn.LayerNorm(hidden_size*2),
            nn.SiLU(),
            nn.Linear(hidden_size*2, hidden_size, bias=False),
        )
        nn.init.zeros_(self.embed_proj[-1].weight)

    def tensor_embedding(self, input_vector):
        x = self.vocab_embed(input_vector)
        return x
        
    def Mapping(self, X,Y, k_):
        x_, y_ = self.vocab_embed.Mapping(X,Y, k_)
        return x_, y_

    # ============================================================
    # ============================================================
    def compute_embeddings(self, x_t):
        """
        Compute and project a high-frequency sinusoidal embedding for the latent input.
        x_t has shape [batch, sequence_length, embedding_dimension].
        """
        B, L, C = x_t.shape
        
        if not x_t.requires_grad:
            x_t.requires_grad_(True)
        
        freq_dim = 1024 // C 
        
        flat_x = x_t.reshape(-1) # [B*L*C]
        
        x_t_emb = self.sigma_map.timestep_embedding(
            flat_x, 
            dim=freq_dim, 
            max_period=1000
        ).reshape(B, L, 1024)
        
        return self.embed_proj(x_t_emb.to(self.embed_proj[-1].weight.dtype))

    def forward_backbone(self, x, t, class_labels):
        hidden_states = self.Embedding_map(x)
        
        # ============================================================
        # ============================================================
        # if self.training:
        #     projected_freq = torch.utils.checkpoint.checkpoint(
        #         self.compute_embeddings, x, use_reentrant=False
        #     )
        # else:
        projected_freq = self.compute_embeddings(x)
        
        hidden_states = hidden_states + projected_freq
        # ============================================================

        c = F.silu(self.sigma_map(t))
        y = self.y_embedder(class_labels, self.training)
        c = c + y 
        rotary_cos_sin = self.rotary_emb(hidden_states) 

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                hidden_states = self.blocks[i](hidden_states, rotary_cos_sin, c, seqlens=None)
            
        return hidden_states, c 

    def forward(self, x, t, class_labels):
        return self.forward_backbone(x, t, class_labels)            
 
    def forward_with_cfg(self, x, t, label, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, c = self.forward_backbone(combined, t, label)
        
        cond_out, uncond_out = torch.split(model_out,len(model_out)//2,dim=0)
        guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        output = torch.cat([guided_out, guided_out], dim=0)
        return output, c
# ==============================================================================
# ==============================================================================
from functools import partial

class DiTImageProcessor(DiT):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        n_heads: int,
        cond_dim: int,
        dropout: float,
        n_blocks: int,
        debug: bool = True,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            n_heads=n_heads,
            cond_dim=cond_dim,
            dropout=dropout,
            n_blocks=n_blocks,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,      
            **kwargs
        )
        self.debug = debug        
        debug_log("Note: Tokenizer is NOT initialized in DiTImageProcessor (Using Pre-computed Tokens)")
        self.tokenizer = None

    def forward(self, batch_token_ids, t_original, class_labels, cfg=None, debug=False):
        batch_indices = batch_token_ids 
        raw_embed = self.vocab_embed(batch_indices.unsqueeze(-1)) 
        raw_embed = raw_embed.to(dtype=torch.bfloat16)
        
        latent = raw_embed * (t_original[:, None, None]**0.5) * K2
        noise = torch.sqrt(1 - t_original[:, None, None]) * torch.randn_like(latent, dtype=torch.bfloat16) * K1
        fused_latent = latent + noise
        
        hidden_state, condition = super().forward(fused_latent, t_original, class_labels)
        
        return hidden_state, condition, batch_indices, t_original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Public eval-only ImageNet64 sampler")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PUBLIC_ROOT, "checkpoints", "imagenet64_checkpoints_v90_21"))
    parser.add_argument("--target_tag", type=str, default="global_step1400000")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    import test_v4_cfg_interval_gigatok

    diffusion_params = {
        "K1": K1,
        "K2": K2,
        "Embed_Dim": Embed_Dim,
    }
    test_v4_cfg_interval_gigatok.run_evaluation(
        args,
        args.output_dir,
        target_tag=args.target_tag,
        ModelClass=DiTImageProcessor,
        vocab_size=vocab_size,
        diffusion_params=diffusion_params,
    )
