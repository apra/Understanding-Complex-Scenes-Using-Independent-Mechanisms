import torch
import torch.nn as nn
# import torch.distributions as dists
import torch.nn.functional as F
import torchvision

from attrdict import AttrDict

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import numpy as np


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)


class INConvBlock(nn.Module):
    def __init__(self, nin, nout, stride=1, instance_norm=True, act=nn.ReLU()):
        super(INConvBlock, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not instance_norm)
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(nout, affine=True)
        else:
            self.instance_norm = None
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return self.act(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UNet(nn.Module):
    def __init__(self, filter_start=32):
        super(UNet, self).__init__()
        c = filter_start
        self.down = nn.ModuleList([
            INConvBlock(4, c),
            INConvBlock(c, c),
            INConvBlock(c, 2 * c),
            INConvBlock(2 * c, 2 * c),
            INConvBlock(2 * c, 2 * c),  # no downsampling
        ])
        self.up = nn.ModuleList([
            INConvBlock(4 * c, 2 * c),
            INConvBlock(4 * c, 2 * c),
            INConvBlock(4 * c, c),
            INConvBlock(2 * c, c),
            INConvBlock(2 * c, c)
        ])
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(4 * 4 * 2 * c, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4 * 4 * 2 * c), nn.ReLU()
        )
        self.final_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        x_up = self.mlp(x_down[-1]).view(batch_size, -1, 4, 4)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        return self.final_conv(x_up)


class SimpleSBP(nn.Module):

    def __init__(self, core):
        super(SimpleSBP, self).__init__()
        self.core = core

    def forward(self, x, steps_to_run):
        # Initialise lists to store tensors over K steps
        log_m_k = []
        # Set initial scope to all ones, so log scope is all zeros
        log_s_k = [torch.zeros_like(x)[:, :1, :, :]]
        # Loop over steps
        for step in range(steps_to_run):
            # Compute mask and update scope. Last step is different
            # Compute a_logits given input and current scope
            core_out = self.core(torch.cat((x, log_s_k[step]), dim=1))

            # Take first channel as logits for masks
            a_logits = core_out[:, :1, :, :]

            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)

            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[step] + log_a)

            # Update scope given attention
            log_s_k.append(log_s_k[step] + log_neg_a)

        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])

        return log_m_k, log_s_k

    def masks_from_zm_k(self, zm_k, img_size):
        # zm_k: K*(batch_size, ldim)
        b_sz = zm_k[0].size(0)
        log_m_k = []
        log_s_k = [torch.zeros(b_sz, 1, img_size, img_size)]
        other_k = []

        for zm in zm_k:
            core_out = self.core.decode(zm)
            # Take first channel as logits for masks
            a_logits = core_out[:, :1, :, :]
            log_a = F.logsigmoid(a_logits)
            log_neg_a = F.logsigmoid(-a_logits)
            # Take rest of channels for other
            other_k.append(core_out[:, 1:, :, :])
            # Compute mask. Note that old scope needs to be used!!
            log_m_k.append(log_s_k[-1] + log_a)
            # Update scope given attention
            log_s_k.append(log_s_k[-1] + log_neg_a)
        # Set mask equal to scope for last step
        log_m_k.append(log_s_k[-1])
        return log_m_k, log_s_k, other_k


class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim))

        self.register_buffer('g_1', g_1.view((1, 1) + g_1.shape))
        self.register_buffer('g_2', g_2.view((1, 1) + g_2.shape))

    def forward(self, x):
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1)
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1)
        return torch.cat((x, g_1, g_2), dim=1)


class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)

    def forward(self, x):
        b_sz = x.size(0)
        # Broadcast
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)
        return self.pixel_coords(x)


class BroadcastDecoder(nn.Module):

    def __init__(self, in_chnls, out_chnls, h_chnls, num_layers, img_dim, act):
        super(BroadcastDecoder, self).__init__()
        broad_dim = img_dim + 2 * num_layers
        mods = [BroadcastLayer(broad_dim),
                nn.Conv2d(in_chnls + 2, h_chnls, 3),
                act]
        for _ in range(num_layers - 1):
            mods.extend([nn.Conv2d(h_chnls, h_chnls, 3), act])
        mods.append(nn.Conv2d(h_chnls, out_chnls, 1))
        self.seq = nn.Sequential(*mods)

    def forward(self, x):
        return self.seq(x)


def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-8


class MONetCompEncoder(nn.Module):
    def __init__(self, act):
        super(MONetCompEncoder, self).__init__()
        nin = 3
        c = 32
        self.ldim = 16
        nin_mlp = 2 * c * (64 // 16) ** 2
        nhid_mlp = max(256, 2 * self.ldim)
        self.module = nn.Sequential(nn.Conv2d(nin + 1, c, 3, 2, 1), act,
                                    nn.Conv2d(c, c, 3, 2, 1), act,
                                    nn.Conv2d(c, 2 * c, 3, 2, 1), act,
                                    nn.Conv2d(2 * c, 2 * c, 3, 2, 1), act,
                                    Flatten(),
                                    nn.Linear(nin_mlp, nhid_mlp), act,
                                    nn.Linear(nhid_mlp, 2 * self.ldim))

    def forward(self, x):
        return self.module(x)

class ComponentVAE(nn.Module):

    def __init__(self, nout, act, cfg):
        super(ComponentVAE, self).__init__()
        self.ldim = 16  # paper uses 16
        self.montecarlo = False
        self.pixel_bound = cfg["pixel_bound"]
        # Sub-Modules
        self.encoder_module = MONetCompEncoder(act=act)
        self.decoder_module = BroadcastDecoder(
            in_chnls=self.ldim,
            out_chnls=nout,
            h_chnls=32,
            num_layers=4,
            img_dim=64,
            act=act
        )

    def forward(self, x, log_mask):
        """
        Args:
            x (torch.Tensor): Input to reconstruct [batch size, 3, dim, dim]
            log_mask (torch.Tensor or list of torch.Tensors):
                Mask to reconstruct [batch size, 1, dim, dim]
        """
        # -- Check if inputs are lists
        K = 1
        b_sz = x.size(0)
        if isinstance(log_mask, list) or isinstance(log_mask, tuple):
            K = len(log_mask)
            # Repeat x along batch dimension
            x = x.repeat(K, 1, 1, 1)
            # Concat log_m_k along batch dimension
            log_mask = torch.cat(log_mask, dim=0)

        # -- Encode
        x = torch.cat((log_mask, x), dim=1)  # Concat along feature dimension
        mu, sigma = self.encode(x)

        # -- Sample latents
        q_z = Normal(mu, sigma)
        # z - [batch_size * K, l_dim] with first axis: b0,k0 -> b0,k1 -> ...
        z = q_z.rsample()

        # -- Decode
        # x_r, m_r_logits = self.decode(z)
        x_r = self.decode(z)

        # -- Track quantities of interest and return
        x_r_k = torch.chunk(x_r, K, dim=0)
        z_k = torch.chunk(z, K, dim=0)
        mu_k = torch.chunk(mu, K, dim=0)
        sigma_k = torch.chunk(sigma, K, dim=0)
        stats = AttrDict(mu_k=mu_k, sigma_k=sigma_k, z_k=z_k)
        return x_r_k, stats

    def encode(self, x):
        x = self.encoder_module(x)
        mu, sigma_ps = torch.chunk(x, 2, dim=1)
        sigma = to_sigma(sigma_ps)
        return mu, sigma

    def decode(self, z):
        x_hat = self.decoder_module(z)
        if self.pixel_bound:
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def sample(self, batch_size=1, steps=1):
        raise NotImplementedError


def x_loss(x, log_m_k, x_r_k, std):
    # 1.) Sum over steps for per pixel & channel (ppc) losses
    p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
    log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
    log_m_stack = torch.stack(log_m_k, dim=4)
    log_mx = log_m_stack + log_xr_stack

    err_ppc = -torch.log(log_mx.exp().sum(dim=4))
    # 2.) Sum accross channels and spatial dimensions
    return err_ppc.sum(dim=(1, 2, 3))


def get_kl(z, q_z, p_z, montecarlo):
    if isinstance(q_z, list) or isinstance(q_z, tuple):
        assert len(q_z) == len(p_z)
        kl = []
        for i in range(len(q_z)):
            if montecarlo:
                assert len(q_z) == len(z)
                kl.append(get_mc_kl(z[i], q_z[i], p_z[i]))
            else:
                kl.append(kl_divergence(q_z[i], p_z[i]))
        return kl
    elif montecarlo:
        return get_mc_kl(z, q_z, p_z)
    return kl_divergence(q_z, p_z)


def get_mc_kl(z, q_z, p_z):
    return q_z.log_prob(z) - p_z.log_prob(z)


def check_log_masks(log_m_k):
    summed_masks = torch.stack(log_m_k, dim=4).exp().sum(dim=4)
    summed_masks = summed_masks.clone().data.cpu().numpy()
    flat = summed_masks.flatten()
    diff = flat - np.ones_like(flat)
    idx = np.argmax(diff)
    max_diff = diff[idx]
    if max_diff > 1e-3 or np.any(np.isnan(flat)):
        print("Max difference: {}".format(max_diff))
        for i, log_m in enumerate(log_m_k):
            mask_k = log_m.exp().data.cpu().numpy()
            print("Mask value at k={}: {}".format(i, mask_k.flatten()[idx]))
        raise ValueError("Masks do not sum to 1.0. Not close enough.")


class Monet(nn.Module):

    def __init__(self, cfg):
        super(Monet, self).__init__()
        # Configuration
        self.K_steps = cfg["k_steps"]
        self.prior_mode = "softmax"
        self.mckl = False
        self.debug = False
        self.pixel_bound = cfg["pixel_bound"]
        # Sub-Modules
        # - Attention Network
        core = UNet(32)
        self.att_process = SimpleSBP(core)
        # - Component VAE
        self.comp_vae = ComponentVAE(nout=4, act=nn.ReLU(), cfg=cfg)
        self.comp_vae.pixel_bound = cfg["pixel_bound"]
        # Initialise pixel output standard deviations
        std = 0.7 * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = 0.7  # first step
        self.register_buffer('std', std)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images [batch size, 3, dim, dim]
        """
        # --- Predict segmentation masks ---
        log_m_k, log_s_k = self.att_process(x, self.K_steps - 1)

        # --- Reconstruct components ---
        x_m_r_k, comp_stats = self.comp_vae(x, log_m_k)
        # Split into appearances and mask prior
        x_r_k = [item[:, :3, :, :] for item in x_m_r_k]
        m_r_logits_k = [item[:, 3:, :, :] for item in x_m_r_k]
        # Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]

        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        recon = (m_stack * x_r_stack).sum(dim=4)

        # --- Reconstruct masks ---
        log_m_r_stack = self.get_mask_recon_stack(m_r_logits_k, self.prior_mode, log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]

        # --- Loss terms ---
        losses = AttrDict()

        # -- Reconstruction loss
        losses['err'] = x_loss(x, log_m_k, x_r_k, self.std)

        # -- Attention mask KL
        losses['kl_m'] = self.kl_m_loss(log_m_k=log_m_k, log_m_r_k=log_m_r_k)

        # -- Component KL
        q_z_k = [Normal(m, s) for m, s in zip(comp_stats.mu_k, comp_stats.sigma_k)]
        kl_l_k = get_kl(comp_stats.z_k, q_z_k, len(q_z_k) * [Normal(0, 1)], self.mckl)
        losses['kl_l_k'] = [kld.sum(1) for kld in kl_l_k]

        # Track quantities of interest
        stats = AttrDict(
            recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            log_m_r_k=log_m_r_k,
            mx_r_k=[x * logm.exp() for x, logm in zip(x_r_k, log_m_k)])

        # Sanity check that masks sum to one if in debug mode
        if self.debug:
            assert len(log_m_k) == self.K_steps
            assert len(log_m_r_k) == self.K_steps
            check_log_masks(log_m_k)
            check_log_masks(log_m_r_k)

        return recon, losses, stats, att_stats, comp_stats

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)

    def get_mask_recon_stack(self, m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_scope = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == self.K_steps - 1:
                    log_m_r_k.append(log_scope)
                else:
                    log_m = F.logsigmoid(logits)
                    log_neg_m = F.logsigmoid(-logits)
                    log_m_r_k.append(log_scope + log_m)
                    log_scope = log_scope + log_neg_m
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError("No valid prior mode.")

    def kl_m_loss(self, log_m_k, log_m_r_k):
        batch_size = log_m_k[0].size(0)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        # Lower bound to 1e-5 to avoid infinities
        m_stack = torch.max(m_stack, torch.tensor(1e-5).cuda())
        m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5).cuda())
        q_m = Categorical(m_stack.view(-1, self.K_steps))
        p_m = Categorical(m_r_stack.view(-1, self.K_steps))
        kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
        return kl_m_ppc.sum(dim=1)

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        # Sample latents
        z_batched = Normal(0, 1).sample((batch_size * K_steps, self.comp_vae.ldim))
        # Pass latent through decoder
        x_hat_batched = self.comp_vae.decode(z_batched.cuda())
        # Split into appearances and masks
        x_r_batched = x_hat_batched[:, :3, :, :]
        m_r_logids_batched = x_hat_batched[:, 3:, :, :]
        # Apply pixel bound to appearances
        if self.pixel_bound:
            x_r_batched = torch.sigmoid(x_r_batched)
        # Chunk into K steps
        x_r_k = torch.chunk(x_r_batched, K_steps, dim=0)
        m_r_logits_k = torch.chunk(m_r_logids_batched, K_steps, dim=0)
        # Normalise masks
        m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, self.prior_mode, log=False)
        # Apply masking and sum to get generated image
        x_r_stack = torch.stack(x_r_k, dim=4)
        gen_image = (m_r_stack * x_r_stack).sum(dim=4)
        # Tracking
        log_m_r_k = [item.squeeze(dim=4) for item in
                     torch.split(m_r_stack.log(), 1, dim=4)]
        stats = AttrDict(gen_image=gen_image, x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x * m.exp() for x, m in zip(x_r_k, log_m_r_k)])
        return gen_image, stats
