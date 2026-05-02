import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion

import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import functional as F
import pytorch_lightning as L
from itertools import chain

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

class CatLoss(nn.Module):
    """Catogorical Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_heads, mlp_ratio, num_sampling_steps, grad_checkpointing=False):
        super(CatLoss, self).__init__()
        self.in_channels = target_channels
        # SimpleMLPAdaLN, HopfieldMLPAdaLN, HopfieldEBTAdaLN
        self.net = HopfieldEBTAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_blocks=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            grad_checkpointing=grad_checkpointing
        )
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)


    def forward(self, z, x, target, anchor, mask=None, bsz=None, seq_len=None):
        t = torch.zeros(x.shape[:-1], device=target.device)

        # print(f"Catogorical Loss - z: {z.shape}, x: {x.shape}, target: {target.shape}, anchor: {anchor.shape}, t: {t.shape}")

        model_kwargs = dict(c=z, a=anchor, bsz=bsz, seq_len=seq_len)
        model_output = self.net(x, t, **model_kwargs)
        
        # print(f"Catogorical Loss - model_output: {model_output.shape}")

        loss_dict = self.criterion(model_output, target)
        loss = loss_dict["loss"]
        if mask is not None:
            if mask.dim() > 1:
                mask = mask.flatten(start_dim=0, end_dim=1)
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class HopfieldAttention(nn.Module):
    r"""Z = softmax(β Q K^T) · V, with multi-head Hopfield energy computation"""
    def __init__(self,
                 dim_emb: int,
                 dim_query: int | None = None,
                 dim_mem: int | None = None,
                 num_heads: int = 16,
                 qkv_bias: bool = False,
                 qk_scale: float | None = None,
                 proj_bias: bool = True,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 hetero: bool = False,
                 **kwargs):
        super().__init__()

        dim_query = dim_query or dim_emb
        dim_mem = dim_mem or dim_emb
        self.num_heads = num_heads
        self.head_dim  = dim_emb // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.hetero = hetero

        # print(f"Hopfield Attention - dim_query: {dim_query}, dim_mem: {dim_mem}, dim_emb: {dim_emb}")

        # projections for Q, K and (K→V)
        self.W_Q = nn.Linear(dim_query, dim_emb, bias=qkv_bias) 
        # if dim_query != dim_emb else nn.Identity()
        self.W_K = nn.Linear(dim_mem, dim_emb, bias=qkv_bias) 
        # if dim_mem != dim_emb else nn.Identity()
        self.W_V = nn.Linear(dim_emb, dim_emb, bias=qkv_bias) if hetero else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim_emb, dim_emb, bias=proj_bias)
        # self.proj = nn.Identity()
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, memory=None): # query ≡ R, memory ≡ Y

        # print(f"Hopfield Attention - query: {query.shape}, memory: {memory.shape}")

        # Full-dim projections
        Q = self.W_Q(query)  # (B,Lq,D)
        K = self.W_K(query if memory is None else memory)  # (B, Lk, D)

        # Split into multi-heads
        Qh = einops.rearrange(Q, 'K L (H D) -> K H L D', H=self.num_heads)   # (K, H, Lq, D)
        Kh = einops.rearrange(K, 'B L (H D) -> B H L D', H=self.num_heads)  # (B, H, Lk, D)

        # print(f"Hopfield Attention - Qh: {Qh.shape}, Kh: {Kh.shape}")

        # Compute multi-head attention
        logits = torch.einsum('K H q D, B H k D -> B K H q k', Qh, Kh) # (B, K, H, Lq, Lk)
        logits = logits * self.scale

        # attn = F.softmax(logits, dim=-1)
        # attn = self.attn_drop(attn)

        print(f"Hopfield Attention - logits: {logits.shape}")

        # # Compute Hopfield gradient: 
        # # ∇_K E_h = β · softmax(β Q K^T)^T · V_Q
        # V_Q = self.W_V(Q)  # (B,Lk,D)
        # V_Qh = einops.rearrange(V_Q, 'B L (H D) -> B H L D', H=self.num_heads)  # (B,H,Lq,Dh)
        # grad_Kh = + 1.0 * attn.transpose(-2, -1) @ V_Qh # (B,H,Lq,Dh)
        # grad_K = einops.rearrange(grad_Kh, 'B H Lk Dh -> B Lk (H Dh)')
        # # ∇_Q E_h = β · softmax(β Q K^T) · V_K
        # V_K = self.W_V(K)  # (B,Lk,D)
        # V_Kh = einops.rearrange(V_K, 'B L (H D) -> B H L D', H=self.num_heads)  # (B,H,Lk,Dh)
        # grad_Qh = + 1.0 * attn @ V_Kh  # (B,H,Lq,Dh)
        # grad_Q = einops.rearrange(grad_Qh, 'B H L D -> B L (H D)')  # (B,Lq,D)

        # Project and drop
        # grad_Q = self.proj(grad_Q)
        # grad_Q = self.proj_drop(grad_Q)

        energies = torch.logsumexp(logits, dim=-2)  # (B, K)
        energies = energies / self.scale
        energies = energies.mean(dim=1)  # (B, K)

        print(f"Hopfield Attention - energies: {energies.shape}")

        return energies


class EBTBlock(L.LightningModule):
    """
    A EBT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attn=False, **block_kwargs):
        super().__init__()
        self.cross_attn = cross_attn

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if cross_attn:
            self.attn = HopfieldAttention(hidden_size, num_heads=num_heads, qkv_bias=True, hetero=False, **block_kwargs)
            self.normc = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.num_adaLN_params = 1
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            self.num_adaLN_params = 1
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, self.num_adaLN_params * hidden_size, bias=True)
        )
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c=None):
        # print("x:", x.shape, "t:", t.shape, "c:", None if c is None else c.shape)
        if self.cross_attn and c is not None:
            state = self.norm1(x)
            memory = self.normc(c)
            # print(f"EBT Block - state: {state.shape}, memory: {memory.shape}")
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True): #NOTE may want to turn this off for inference eventually
                energies = self.attn(state, memory) # needed to set this as regular sdpa from pt didnt support higher order gradients
        return energies

class HopfieldEBTAdaLN(nn.Module):
    """
    The MLP for Catogorical Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_blocks,
        num_heads,
        mlp_ratio,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.grad_checkpointing = grad_checkpointing

        print(f"Hopfield Attention - in_channels: {in_channels}, model_channels: {model_channels}, out_channels: {out_channels}, z_channels: {z_channels}, num_blocks: {num_blocks}, num_heads: {num_heads}, mlp_ratio: {mlp_ratio}")

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        # self.in_blocks = nn.ModuleList([
        #     # EBTBlock(model_channels, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks // 2)
        #     ResBlock(model_channels) for _ in range(num_blocks // 2)
        #     ])

        # self.mid_blocks = nn.ModuleList([
        #     # EBTBlock(model_channels, num_heads, mlp_ratio=mlp_ratio) for _ in range(1)
        #     ResBlock(model_channels) for _ in range(1)
        #     ])

        # self.out_blocks = nn.ModuleList([
        #     EBTBlock(model_channels, num_heads, mlp_ratio=mlp_ratio, cross_attn=True) for _ in range(num_blocks // 2)
        #     ])
        
        # self.hopfield_layer = HopfieldAttention(dim_emb=model_channels, num_heads=1)

        self.ebt_blocks = nn.ModuleList([
            EBTBlock(model_channels, num_heads, mlp_ratio=mlp_ratio, cross_attn=True) for _ in range(1)
            ])

        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in EBT blocks:
        self.blocks = chain(self.res_blocks, self.ebt_blocks)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0) # turned off bias for final output for ebm

    def forward(self, x, t, c, a, bsz, seq_len, mask_to_pred=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """

        # print(f"Catogorical Model - x: {x.shape}, t: {t.shape}, c: {c.shape}, bsz: {bsz}, seq_len: {seq_len}")
        # print(f"Catogorical Model - anchor: {None if anchor is None else anchor.shape}, mask_to_pred: {None if mask_to_pred is None else mask_to_pred.shape}")

        # if t.dim() > 1:
        #     bsz, seq_len = t.shape
        #     t = t.flatten(start_dim=0, end_dim=1)

        # x = self.input_proj(x)
        # t = self.time_embed(t)
        # c = self.cond_embed(c)
        # a = self.input_proj(anchor)

        # t = t.reshape(bsz, -1, self.model_channels)
        # x = x.reshape(bsz, -1, self.model_channels)
        # c = c.reshape(bsz, -1, self.model_channels)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     for block in self.res_blocks:
        #         h = checkpoint(block, h, y) # y, t
        # else:
        #     for block in self.res_blocks:
        #         h = block(h, y) # y, t

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.ebt_blocks:
                logits = checkpoint(block, a, c)
        else:
            for block in self.ebt_blocks:
                logits = block(a, c)

        print(f"Catogorical Model (After Hopfield Layer) - logits: {logits.shape}")

        return logits