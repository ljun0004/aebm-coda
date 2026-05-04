import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion

import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import functional as F
# import pytorch_lightning as L
from itertools import chain

from torch._dynamo import disable

from transport import create_transport, Sampler

class DDPMLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, num_sampling_steps, grad_checkpointing=False):
        super().__init__()
        self.in_channels = target_channels
        self.score_model = ScoreModel(target_channels)

        # self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="linear", learn_sigma=False)
        # self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="linear", learn_sigma=False)
        
        self.transport = create_transport(
            path_type="Linear",
            prediction="score",
            loss_weight=None,
            train_eps=None,
            sample_eps=None,
            use_cosine_loss=False,
            use_lognorm=True,
        )

        self.sampler = Sampler(self.transport)
        self.num_sampling_steps = num_sampling_steps

    def forward(self, mar, x, mask, labels, cookbook, gt_indices, warmup):

        # print(f"DDPMLoss.forward - x_start: {x_start.shape}, z: {z.shape}, mask: {mask.shape}, cookbook: {cookbook.shape}, gt_indices: {gt_indices.shape}")

        # timestep sampling and embed
        # bsz = x_start.shape[0]

        # x_start = x_start.reshape(bsz * seq_len, -1)

        # t = torch.randint(0, self.train_diffusion.num_timesteps, (bsz,), device=x_start.device)
        # t = t.unsqueeze(dim=1).expand(-1, seq_len).flatten(start_dim=0, end_dim=1)
        # t = t.repeat_interleave(seq_len)

        model_kwargs = dict(mar=mar, x_start=x, mask=mask, mask_to_pred=None, labels=labels, cookbook=cookbook, gt_indices=gt_indices, warmup=warmup, cfg_scale=None)
        # loss_dict = self.train_diffusion.training_losses(self.score_model, x_start, t, model_kwargs)
        loss_dict = self.transport.training_losses(self.score_model, x.detach().clone(), model_kwargs)

        return loss_dict["mse"], loss_dict["ce"], loss_dict["re"], loss_dict["logits"], loss_dict["q"], loss_dict["pi"], loss_dict["score"], loss_dict["temb"], loss_dict["scale"]

    def sample(self, mar, x, mask, mask_to_pred, labels, cookbook, temperature=1.0, cfg=1.0, mode="diffusion", imgs=None, gt_indices=None):

        # print(f"DDPMLoss.sample - x: {x.shape}, mask: {mask.shape}, labels: {labels.shape}")

        bsz, _, h, w = x.shape

        sample_fn = self.sampler.sample_ode(
            sampling_method="euler",
            num_steps=self.num_sampling_steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            timestep_shift=0.3,
        )

        # diffusion loss sampling
        if not cfg == 1.0:
            # noise = torch.randn(((bsz // 2) * seq_len), self.in_channels).cuda()
            noise = torch.randn_like(x[:(bsz // 2)])
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(mar=mar, x_start=x, mask=mask, mask_to_pred=mask_to_pred, labels=labels, cookbook=cookbook, gt_indices=gt_indices, warmup=False, cfg_scale=cfg)
            model_fn = self.score_model.forward_with_cfg
        else:
            # noise = torch.randn((bsz * seq_len), self.in_channels).cuda()
            noise = torch.randn_like(x)
            model_kwargs = dict(mar=mar, x_start=x, mask=mask, mask_to_pred=mask_to_pred, labels=labels, cookbook=cookbook, gt_indices=gt_indices, warmup=False, cfg_scale=None)
            model_fn = self.score_model.forward

        if mode == "reconstruction":
            # assert cfg == 1.0
            if imgs is None:
                raise ValueError(f"x must be provided for mode={mode}, but got None.")
            x_start = imgs
            t = torch.tensor([0] * bsz).cuda()
            _, _, q, _, _, _, _ = model_fn(x_start, t, **model_kwargs)
            sampled_token_latent = q.permute(0, 2, 1).reshape(bsz, -1, h, w)
        elif mode == "diffusion":
            # sampled_token_latent = sample_fn(
            #     model_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            #     temperature=temperature
            # )
            sampled_token_latent = sample_fn(noise, model_fn, **model_kwargs)[-1]
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'reconstruction' or 'diffusion'.")

        # sampled_token_latent = sampled_token_latent.reshape(bsz, seq_len, -1)
        # sampled_token_latent = sampled_token_latent[mask_to_pred.nonzero(as_tuple=True)]

        return sampled_token_latent

class ScoreModel(nn.Module):
    """
    Unified MAR-based score model used for both training and sampling.
    """
    def __init__(self, target_channels):
        super().__init__()
        self.in_channels = target_channels

    # @disable
    def forward(self, x_t, t, mar, x_start, mask, mask_to_pred, labels, cookbook, gt_indices, warmup, cfg_scale):
        """
        x_t : [B, L, D]
        t   : [B] or scalar
        z, mask: MAR encoder outputs and masks
        cookbook: codebook matrix [K, D]
        """

        with torch.enable_grad():

            # print(f"ScoreModel forward - x: {x.shape}, x_t: {x_t.shape}, t: {t.shape}, labels: {labels.shape}")

            # bsz, c, h, w = x.shape
            mask = mask.to(x_t.dtype).detach()
            # mask_spatial = mask.view(bsz, mar.token_h, mar.token_w)
            # mask_spatial = mask.view(bsz, mar.seq_h, mar.seq_w).repeat_interleave(mar.patch_size, dim=1).repeat_interleave(mar.patch_size, dim=2)
            x_t = x_t.detach().requires_grad_(True)

            # z = mar.z_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # z_t = mar.z_proj(x_t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # z = mar.z_proj(x)
            # z_t = mar.z_proj(x_t)
            # z_tokens = self.patchify(z, mar)
            # zt_tokens = self.patchify(z_t, mar)
            z_tokens = mar.x_embedder(x_start)
            zt_tokens = mar.x_embedder(x_t)

            # print(f"ScoreModel forward - z_tokens: {z_tokens.shape}, zt_tokens: {zt_tokens.shape}, mask: {mask.shape}")

            z_masked = ((1.0 - mask.unsqueeze(dim=-1)) * z_tokens) + (mask.unsqueeze(dim=-1) * zt_tokens)

            # x_tokens = self.patchify(x, mar)
            # xt_tokens = self.patchify(x_t, mar)
            # x_masked = ((1.0 - mask.unsqueeze(dim=-1)) * x_tokens) + (mask.unsqueeze(dim=-1) * xt_tokens)
            # z_masked = mar.z_proj(x_masked)

            z_start = z_tokens.detach()
            # z_c = mar.z_proj(cookbook).detach()
            k = cookbook.shape[0]
            # z_c = mar.c_proj(cookbook.view(k, -1, 1, 1)).view(k, -1)

            # time embedding
            # t = t.reshape(bsz, seq_len)
            # t_padded = torch.cat([torch.zeros(bsz, mar.buffer_size, dtype=t.dtype, device=t.device), t], dim=1)
            # t_padded = t_padded.flatten(start_dim=0, end_dim=1)

            # t = t.view(bsz, seq_len)[:, 0]
            t_embedding = mar.t_embedder(t)

            # class embedding
            class_embedding = mar.y_embedder(labels, mar.training)

            # encoder
            h = mar.forward_mae_encoder(z_masked, mask, t_embedding, class_embedding)

            # decoder
            # h = mar.forward_mae_decoder(z, mask, t_embedding, class_embedding)

            # final layer
            # word_embedding = mar.word_embedding
            # word_embedding = torch.zeros(mar.cookbook_size, mar.final_layer.model_channels, dtype=x.dtype, device=x.device)
            logits, q, pi, v = mar.final_layer(mar, h, t_embedding, class_embedding, cookbook_embedding=None, gt_indices=gt_indices)

            grad_q = mar.alpha * q - mar.beta * v.detach()
            
            score = torch.autograd.grad(
                outputs = q, 
                inputs = x_t,
                grad_outputs = grad_q,
                retain_graph = True,
                create_graph = mar.training and (mar.ddpmloss_scale > 0)
                )[0]            

            model_output = - score
            
        return model_output, logits, q, pi, z_start, grad_q

    def patchify(self, x, mar):
        bsz, d, h, w = x.shape
        p = mar.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, d, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, d * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x, mar):
        bsz, l, d = x.shape
        h_ = w_ = int(l ** 0.5)
        p = mar.patch_size
        c = mar.vae_embed_dim

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def forward_with_cfg(self, x_t, t, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_output, *extras = self.forward(combined, t, **kwargs)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_output[:, :3], model_output[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + kwargs['cfg_scale'] * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1), *extras


class DetachedJacobianVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, x, u):
        """
        Forward: s = J_Q(x)^T * u
        Block gradients flowing back through J_Q.
        """
        with torch.enable_grad():
            score = torch.autograd.grad(
                outputs=q,
                inputs=x,
                grad_outputs=u,
                create_graph=False, 
                retain_graph=True
            )[0]

        ctx.save_for_backward(q, x)
        return score

    @staticmethod
    def backward(ctx, grad_s):
        """
        Backward: dL/du = J_Q(x) * grad_s
        This computes the Jacobian-Vector Product (JVP).
        """
        q, x = ctx.saved_tensors
        
        if grad_s is None:
            return None, None, None
        
        with torch.enable_grad():
            dummy = torch.zeros_like(q, requires_grad=True)
            
            vjp_dummy = torch.autograd.grad(
                outputs=q,
                inputs=x,
                grad_outputs=dummy,
                create_graph=True, 
                retain_graph=True
            )[0]
            
            grad_u = torch.autograd.grad(
                outputs=vjp_dummy,
                inputs=dummy,
                grad_outputs=grad_s,
                create_graph=False
            )[0]

        return None, None, grad_u

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)
