import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
import torchvision

from attrdict import AttrDict

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import numpy as np


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super(UNet, self).__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2 ** i))
            cur_in_channels = channel_base * 2 ** i

        self.up_sample = nn.ModuleList()
        for i in range(num_blocks - 1, 0, -1):
            self.up_sample.append(nn.ConvTranspose2d(channel_base * 2 ** i,
                                                     channel_base * 2 ** (i - 1),
                                                     2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks - 2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2 ** (i + 1), channel_base * 2 ** i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, kernel_size=1)

    def down(self, x, layer):
        x = nn.MaxPool2d(2)(x)
        return self.down_convs[layer](x)

    def up(self, x, skip, layer):
        x = self.up_sample[layer](x)
        # concat along channels
        x = torch.cat((x, skip), dim=1)
        return self.up_convs[layer](x)

    def forward(self, x):
        skip_activations = []
        cur = self.down_convs[0](x)
        for down_layer in range(1, self.num_blocks):
            skip_activations.append(cur)
            cur = self.down(cur, down_layer)

        for i in range(self.num_blocks - 1):
            cur = self.up(cur, skip_activations[-i - 1], i)

        return {
            "out": self.final_conv(cur)
        }


class SimpleSBP(nn.Module):

    def __init__(self, params):
        super(SimpleSBP, self).__init__()
        input_channels = params["input_channels"]
        self.core = UNet(num_blocks=params["num_blocks"],
                         in_channels=input_channels + 1,
                         out_channels=1,
                         channel_base=params["channel_base"])

    def forward(self, x, log_s_t):
        # Compute mask and update scope. Last step is different
        # Compute a_logits given input and current scope
        core_out = self.core(torch.cat((x, log_s_t), dim=1))["out"]  # [B, C+1, H, W]

        # Take output channel as logits for masks
        a_logits = core_out[:, 0:1, :, :]  # [B, 1, H, W]
        log_a = F.logsigmoid(a_logits)  # [B, 1, H, W]
        log_neg_a = F.logsigmoid(-a_logits)  # [B, 1, H, W]

        # Compute region
        log_r = log_s_t + log_a

        # Update scope
        next_log_s = log_s_t + log_neg_a

        # if torch.isnan(log_r).any():
        #     test_tensor(log_s_t, "ATTENTION log_s_t")
        #     test_tensor(core_out, "ATTENTION core_out")
        #     test_tensor(log_a, "ATTENTION log_a")

        return {
            "next_log_s": next_log_s,
            "log_r": log_r
        }


class ExpertDecoder(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class ExpertEncoder(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x


def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-8


class ComponentVAE(nn.Module):

    def __init__(self, sigma=0.7, params=None):
        super(ComponentVAE, self).__init__()
        self.ldim = 16  # paper uses 16
        # Sub-Modules
        self.encoder_module = ExpertEncoder(height=64, width=64)
        self.decoder_module = ExpertDecoder(height=64, width=64)
        self.sigma = sigma

    def forward(self, x, log_r):
        """
        Args:
            x: Input to reconstruct [batch size, 3, dim, dim]
            log_r: region
            log_s: scope
        """

        encoding_results = self.encode(x, log_r)

        # -- Decode
        decoding_results = self.decode(encoding_results["z"])

        return {
            **encoding_results,
            **decoding_results
        }

    def encode(self, x, log_r):
        x = torch.cat((log_r, x), dim=1)
        x = self.encoder_module(x)
        mu, sigma_ps = torch.chunk(x, 2, dim=1)
        sigma = to_sigma(sigma_ps)
        q_phi_z = Normal(mu, sigma)
        z = q_phi_z.rsample()
        kl = kl_divergence(q_phi_z, Normal(0., 1.))

        return {
            "mu": mu,
            "sigma": sigma,
            "z": z,
            "kl": kl
        }

    def decode(self, z):
        decoder_output = self.decoder_module(z)
        mu_x_logits = decoder_output[:, :3]
        m_pred_logits = decoder_output[:, 3:]

        return {
            "m_pred_t_logits": m_pred_logits,
            "mu_x_logits": mu_x_logits,
            "log_mu_x": F.logsigmoid(mu_x_logits),
            "log_m_pred_t": F.logsigmoid(m_pred_logits)
        }

    def sample(self, batch_size=1, steps=1):
        raise NotImplementedError


def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    # print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    # print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())


def test_tensor(ten, name="name"):
    print("{} ... max: {}, min: {}, mean: {}".format(name, torch.max(ten), torch.min(ten),
                                                     torch.mean(ten)))


class Expert(nn.Module):

    def __init__(self, params):
        super(Expert, self).__init__()
        # - Attention Network
        self.attention = SimpleSBP(params)
        # - Component VAE
        self.comp_vae = ComponentVAE()
        # Initialise pixel output standard deviations
        self.sigma = 0.7

        self.lambda_competitive = params["lambda_competitive"]

        self.beta_loss = params["beta_loss"]
        self.gamma_loss = params["gamma_loss"]

    def forward(self, x, log_s_t, last_object=False):
        """
        Args:
            x: Input images [B, C, H, W]
            log_s_t: logarithm of scope for element t
            last_object: if this is the last object to be reconstructed
        """
        next_log_s_t = torch.zeros_like(log_s_t)
        log_r_t = log_s_t
        if not last_object:
            # --- Predict segmentation masks ---
            attention_results = self.attention(x, log_s_t)
            next_log_s_t = attention_results["next_log_s"]
            log_r_t = attention_results["log_r"]

        # --- Reconstruct components ---
        vae_results = self.comp_vae(x, log_r_t)
        log_mu_x = vae_results["log_mu_x"]
        log_m_pred_t = vae_results["log_m_pred_t"]

        r_t_pred = (log_s_t + log_m_pred_t).exp()
        prob_r_t_pred = torch.clamp(r_t_pred, 1e-5, 1 - 1e-5)
        prob_log_r_t = torch.clamp(log_r_t.exp(), 1e-5, 1 - 1e-5)
        # test_tensor(r_t_pred, "r_t_pred")
        # test_tensor(r_t_pred, "R r_t_pred")
        # test_tensor(log_r_t.exp(), "R log_r_t.exp()")
        p_r_t = dists.Bernoulli(probs=prob_r_t_pred)
        q_psi_t = dists.Bernoulli(probs=prob_log_r_t)

        raw_loss_x_t = (log_r_t.exp() / (2 * self.sigma)) * torch.square(x - log_mu_x.exp())
        raw_loss_z_t = vae_results["kl"]
        raw_loss_r_t = kl_divergence(q_psi_t, p_r_t)

        loss_x_t = torch.sum(raw_loss_x_t, [1, 2, 3])  # sum over all pixels and channels
        loss_z_t = torch.sum(raw_loss_z_t, [1])  # sum over all latent dimensions
        loss_r_t = torch.sum(raw_loss_r_t, [1, 2, 3])  # sum over all pixels and channels

        # the reconstruction, scope*predicted shape*mean of output
        x_recon_t = (log_s_t + log_m_pred_t + log_mu_x).exp()

        # region the attention network is looking at
        region_attention = log_r_t.exp()

        # the reconstructed mask from the VAE
        mask_recon = log_m_pred_t.exp()

        # debug
        # if torch.isinf(loss_r_t).any():
        #     test_tensor(loss_r_t, "R loss_r_t")
        # if torch.isnan(loss_x_t).any():
        #     print(log_r_t)
        #     print("Is last object: {}".format(last_object))
        #     test_tensor(log_s_t, "R log_s_t")
        #     test_tensor(r_t_pred, "R r_t_pred")
        #     test_tensor(log_r_t.exp(), "R log_r_t.exp()")

        return {
            "loss_x_t": loss_x_t,
            "loss_z_t": loss_z_t,
            "loss_r_t": loss_r_t,
            "x_recon_t": x_recon_t,
            "x_recon_t_not_masked": log_mu_x.exp(),
            "next_log_s_t": next_log_s_t,
            "region_attention": region_attention,
            "mask_recon": mask_recon,
            "loss": loss_x_t + self.gamma_loss * loss_r_t + self.beta_loss * loss_z_t,
            "competition_objective": -1 * (self.lambda_competitive * loss_x_t + loss_r_t)
        }

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)

    def sample(self, batch_size, num_objects):
        return NotImplementedError


class ECON(nn.Module):

    def __init__(self, params):
        super(ECON, self).__init__()

        self.num_experts = params["num_experts"]
        self.num_objects = params["num_objects"]
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            self.experts.append(Expert(params=params))
        self.register_buffer('best_objective', torch.tensor(float("inf")))

        self.punish_factor = params["punish_factor"]

    def forward(self, x):
        loss = torch.zeros_like(x[:, 0, 0, 0])
        loss_x = torch.zeros_like(x[:, 0, 0, 0])
        loss_z = torch.zeros_like(x[:, 0, 0, 0])
        loss_r = torch.zeros_like(x[:, 0, 0, 0])
        batch_size = x.shape[0]
        log_scope = torch.zeros_like(x[:, 0:1])
        final_recon = torch.zeros_like(x)
        masks = []
        recons_steps = []
        recons_steps_not_masked = []
        attention_regions = []
        selected_expert_per_object = []

        indexes = list(range(batch_size))

        for i in range(self.num_objects):
            competition_results = []
            results = []
            for j, expert in enumerate(self.experts):
                results_expert = expert(x, log_scope, last_object=(i == (self.num_objects - 1)))
                competition_results.append(results_expert["competition_objective"])
                results.append(results_expert)

            # stack of reconstruction losses for every expert at object j
            losses = torch.stack([x["loss"] for x in results], dim=1)

            # if torch.isinf(losses).any():
            #     print("INF")

            decision = dists.Categorical(
                probs=F.softmax(torch.stack(competition_results, dim=1), dim=1))
            selected_expert = decision.sample()
            selected_expert_per_object.append(selected_expert)
            # print("asdfads", selected_expert)

            loss = loss + losses[indexes, selected_expert]

            # partial reconstructions
            recons_t = torch.stack([x["x_recon_t"] for x in results], dim=1)[
                indexes, selected_expert].data
            recons_steps.append(recons_t)
            # partial attention regions
            all_attention_regions = torch.stack([x["region_attention"] for x in results], dim=1)
            region_attention_t = all_attention_regions[indexes, selected_expert].data
            attention_regions.append(region_attention_t)

            recons_steps_not_masked_t = \
                torch.stack([x["x_recon_t_not_masked"] for x in results], dim=1)[
                    indexes, selected_expert].data
            recons_steps_not_masked.append(recons_steps_not_masked_t)

            # the final reconstruction
            final_recon += recons_t

            loss_x += torch.stack([x["loss_x_t"] for x in results], dim=1)[
                indexes, selected_expert].data
            loss_r += torch.stack([x["loss_r_t"] for x in results], dim=1)[
                indexes, selected_expert].data
            loss_z += torch.stack([x["loss_z_t"] for x in results], dim=1)[
                indexes, selected_expert].data

            #dontpayattention = torch.zeros_like(region_attention_t)

            #optim_objective = optim_objective + losses[selected_expert]
                              # + self.punish_factor * \
                              # torch.nn.BCELoss()(all_attention_regions[indexes, ~selected_expert],
                              #                     dontpayattention) / (len(self.experts) - 1)

            # update the next scope for the network
            candidate_scopes = torch.stack([x["next_log_s_t"] for x in results], dim=1)
            log_scope = candidate_scopes[indexes, selected_expert]
            # print(next_scopes.shape, scope.shape)

        return {
            #"optim_obj": optim_objective,
            "loss": loss,
            "recons_steps": recons_steps,
            "recons_steps_not_masked": recons_steps_not_masked,
            "recon": final_recon,
            "attention_regions": attention_regions,
            "loss_x": loss_x,
            "loss_r": loss_r,
            "loss_z": loss_z,
            "selected_expert_per_object": selected_expert_per_object
        }
