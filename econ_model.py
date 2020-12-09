import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F

from torch.distributions.normal import Normal
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

        return {
            "next_log_s": next_log_s,
            "log_r": log_r
        }


class ExpertDecoder(nn.Module):
    def __init__(self, height, width, ldim=16):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(ldim+2, 32, kernel_size=3),
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
    def __init__(self, width, height, ldim=16):
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
            nn.Linear(256, 2*ldim)
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
        self.ldim = params["latent_dim"]  # paper uses 16
        # Sub-Modules
        self.encoder_module = ExpertEncoder(height=64, width=64, ldim=self.ldim)
        self.decoder_module = ExpertDecoder(height=64, width=64, ldim=self.ldim)
        self.sigma = sigma

    def forward(self, x, log_r):
        """
        Args:
            x: Input to reconstruct [batch size, 3, dim, dim]
            log_r: region
            log_s: scope
        """

        encoding_results = self.encode(x, log_r)

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

    def sample(self, batch_size=1):
        q_phi_z = Normal(torch.zeros((batch_size, self.ldim)), 1.)
        z = q_phi_z.sample()
        return self.decode(z)


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
        self.comp_vae = ComponentVAE(params=params)

        self.lambda_competitive = params["lambda_competitive"]
        print(self.lambda_competitive)

        self.beta_loss = params["beta_loss"]
        self.gamma_loss = params["gamma_loss"]

    def forward(self, x, log_s_t, sigma_x, last_object=False):
        """
        Args:
            x: Input images [B, C, H, W]
            log_s_t: logarithm of scope for element t
            last_object: if this is the last object to be reconstructed
            sigma_x: the sigma for the reconstruction distribution
            last_object: True if this the last object being reconstructed
        """
        next_log_s_t = torch.zeros_like(log_s_t)
        log_r_t = log_s_t
        if not last_object:
            # --- Predict segmentation masks ---
            attention_results = self.attention(x, log_s_t)
            next_log_s_t = attention_results["next_log_s"]
            log_r_t = attention_results["log_r"]

        vae_results = self.comp_vae(x, log_r_t)
        log_mu_x = vae_results["log_mu_x"]
        log_m_pred_t = vae_results["log_m_pred_t"]

        r_t_pred = (log_s_t + log_m_pred_t).exp()
        prob_r_t_pred = torch.clamp(r_t_pred, 1e-5, 1 - 1e-5)
        prob_log_r_t = torch.clamp(log_r_t.exp(), 1e-5, 1 - 1e-5)

        p_r_t = dists.Bernoulli(probs=prob_r_t_pred)
        q_psi_t = dists.Bernoulli(probs=prob_log_r_t)

        raw_loss_x_t = (log_r_t.exp() / (2 * sigma_x * sigma_x)) * torch.square(
            x - log_mu_x.exp())
        raw_loss_z_t = vae_results["kl"]
        raw_loss_r_t = kl_divergence(q_psi_t, p_r_t)

        loss_x_t = torch.sum(raw_loss_x_t, [1, 2, 3])  # sum over all pixels and channels
        loss_z_t = torch.sum(raw_loss_z_t, [1])  # sum over all latent dimensions
        loss_r_t = torch.sum(raw_loss_r_t, [1, 2, 3])  # sum over all pixels and channels

        # the reconstruction, scope*predicted shape*mean of output
        x_recon_t = (log_s_t + log_r_t + log_mu_x).exp()

        # region the attention network is looking at
        region_attention = log_r_t.exp()

        # the reconstructed mask from the VAE
        mask_recon = log_m_pred_t.exp()

        return {
            "loss_x_t": loss_x_t,
            "loss_z_t": loss_z_t,
            "loss_r_t": loss_r_t,
            "x_recon_t": x_recon_t,
            "x_recon_t_not_masked": log_mu_x.exp(),
            "next_log_s_t": next_log_s_t,
            "region_attention": region_attention,
            "mask_recon": mask_recon,
            "log_s_t": log_s_t,
            "competition_objective": -1 * (self.lambda_competitive * loss_x_t + loss_r_t)# + 1.*torch.var(log_mu_x,dim=(1,2,3)))
        }

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)


def all_to_cpu(data):
    result = {}
    for key, value in data.items():
        result[key] = value.detach().cpu().numpy()
    return result


class ECON(nn.Module):

    def __init__(self, params):
        super(ECON, self).__init__()

        self.num_experts = params["num_experts"]
        self.num_objects = params["num_objects"]
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            self.experts.append(Expert(params=params))

        self.punish_factor = params["punish_factor"]

        self.competition_temperature = params["competition_temperature"]
        self.beta_loss = params["beta_loss"]
        self.gamma_loss = params["gamma_loss"]

        # Initialise pixel output standard deviations
        self.sigmas_x = params["sigmas_x"]

    def run_single_expert(self, x, expert_id, beta=None, gamma=None):
        beta_loss = self.beta_loss
        gamma_loss = self.gamma_loss
        if beta is not None:
            beta_loss = beta
        if gamma is not None:
            gamma_loss = gamma

        collected_results = []
        selected_expert_per_object = []

        log_scopes = [torch.zeros_like(x[:, 0:1])]

        for obj in range(self.num_objects):
            results_expert = self.experts[expert_id](x, log_scopes[-1], self.sigmas_x[obj],
                                                     last_object=(obj == (self.num_objects - 1)))

            # compute the overall loss
            loss_object = results_expert["loss_x_t"] + \
                          gamma_loss * results_expert["loss_r_t"] + \
                          beta_loss * results_expert["loss_z_t"]

            collected_results.append(all_to_cpu(results_expert))

            if obj == 0:
                loss = loss_object
            else:
                loss += loss_object

            log_scopes.append(results_expert["next_log_s_t"])

        return {
            "loss": loss,
            "results": collected_results,
            "selected_expert_per_object": selected_expert_per_object
        }

    def forward(self, x, beta=None, gamma=None):
        beta_loss = self.beta_loss
        gamma_loss = self.gamma_loss
        if beta is not None:
            beta_loss = beta
        if gamma is not None:
            gamma_loss = gamma

        batch_size = x.shape[0]

        collected_results = []
        selected_expert_per_object = []

        indexes = list(range(batch_size))

        log_scopes = [torch.zeros_like(x[:, 0:1])]

        for obj in range(self.num_objects):
            competition_results = []
            losses = []
            results = []
            candidate_scopes = []
            candidate_attentions = []
            for j, expert in enumerate(self.experts):
                # run the expert
                results_expert = expert(x, log_scopes[-1], self.sigmas_x[obj],
                                        last_object=(obj == (self.num_objects - 1)))

                # collect the gradient-related results
                competition_results.append(results_expert["competition_objective"])
                candidate_scopes.append(results_expert["next_log_s_t"])
                candidate_attentions.append(results_expert["region_attention"])

                # compute the overall loss
                losses.append(results_expert["loss_x_t"] +
                              gamma_loss * results_expert["loss_r_t"] +
                              beta_loss * results_expert["loss_z_t"])

                # collect the non-gradient related results to analyze performance
                results.append(all_to_cpu(results_expert))
                del results_expert

            collected_results.append(results)

            # stack of reconstruction losses for every expert at object j
            losses = torch.stack(losses, dim=0)

            if self.num_experts == 1:
                selected_expert = torch.zeros((batch_size,), dtype=torch.long)
                probs = torch.ones((batch_size,), dtype=torch.float)
            else:
                probs = F.softmax(
                    torch.stack(competition_results, dim=1) / self.competition_temperature, dim=1)
                decision = dists.Categorical(probs=probs)
                selected_expert = decision.sample()

            selected_expert_per_object.append(selected_expert.cpu().numpy())

            if obj == 0:
                loss = losses[selected_expert, indexes]
            else:
                loss += losses[selected_expert, indexes]
            # update the next scope for the network
            candidate_scopes = torch.stack(candidate_scopes, dim=0)

            log_scopes.append(candidate_scopes[selected_expert, indexes])

            # candidate_attentions = torch.stack(candidate_attentions, dim=0)
            # winning_attention = candidate_attentions[selected_expert, indexes]

            #loss += self.punish_factor*torch.nn.BCELoss(reduction="sum")(winning_attention, torch.ones_like(winning_attention))

            # if self.num_experts > 1:
            #     experts_sorted = torch.argsort(probs, descending=True)
            #
            #     candidate_attentions = torch.stack(candidate_attentions, dim=0)
            #     for i in range(self.num_experts):
            #         attention = candidate_attentions[i]
            #
            #         winning_attention = candidate_attentions[selected_expert, indexes].clone().detach()
            #
            #         loss -= self.punish_factor*torch.nn.BCELoss(reduction="sum")(attention, winning_attention)/(self.num_experts-1)


            # print(next_scopes.shape, scope.shape)

        return {
            "loss": loss,
            "results": collected_results,
            "selected_expert_per_object": selected_expert_per_object
        }


# follows code from Andrea Dittadi: https://github.com/addtt/boiler-pytorch
def is_conv(module):
    """Returns whether the module is a convolutional layer."""
    return isinstance(module, torch.nn.modules.conv._ConvNd)


def is_linear(module):
    """Returns whether the module is a linear layer."""
    return isinstance(module, torch.nn.Linear)


def to_np(x):
    """
    Converts to numpy and puts in cpu, handles lists and numpy arrays as well, converts them to numpy array
    of numpy arrays.
    Args:
        x: input list of tensors

    Returns:
        the numpy'ed tensor list
    """
    if isinstance(x, (list, np.ndarray)):
        for i, item in enumerate(x):
            try:
                x[i] = x[i].detach().cpu().numpy()
            except AttributeError:
                print("error to_np")
                return x[i]
        return x
    else:
        try:
            return x.detach().cpu().numpy()
        except AttributeError:
            print("error to_np")
            return x


debug = True


def _get_data_dep_hook(init_scale):
    """Creates forward hook for data-dependent initialization.
    The hook computes output statistics of the layer, corrects weights and
    bias, and corrects the output accordingly in-place, so the forward pass
    can continue.
    Args:
        init_scale (float): Desired scale (standard deviation) of each
            layer's output at initialization.
    Returns:
        Forward hook for data-dependent initialization
    """

    def hook(module, inp, out):
        inp = inp[0]

        out_size = out.size()

        if is_conv(module):
            separation_dim = 1
        elif is_linear(module):
            separation_dim = -1
        dims = tuple([i for i in range(out.dim()) if i != separation_dim])
        mean = out.mean(dims, keepdim=True)
        var = out.var(dims, keepdim=True)

        if debug:
            # print("Shapes:\n   input:  {}\n   output: {}\n   weight: {}".format(
            #    inp.size(), out_size, module.weight.size()))
            print("Dims to compute stats over:", dims)
            print("Input statistics:\n   mean: {}\n   var: {}".format(
                to_np(inp.mean(dims)), to_np(inp.var(dims))))
            print("Output statistics:\n   mean: {}\n   var: {}".format(
                to_np(out.mean(dims)), to_np(out.var(dims))))
            print("Weight statistics:   mean: {}   var: {}".format(
                to_np(module.weight.mean()), to_np(module.weight.var())))

        # Given channel y[i] we want to get
        #   y'[i] = (y[i]-mu[i]) * is/s[i]
        #         = (b[i]-mu[i]) * is/s[i] + sum_k (w[i, k] * is / s[i] * x[k])
        # where * is 2D convolution, k denotes input channels, mu[i] is the
        # sample mean of channel i, s[i] the sample variance, b[i] the current
        # bias, 'is' the initial scale, and w[i, k] the weight kernel for input
        # k and output i.
        # Therefore the correct bias and weights are:
        #   b'[i] = is * (b[i] - mu[i]) / s[i]
        #   w'[i, k] = w[i, k] * is / s[i]
        # And finally we can modify in place the output to get y'.

        scale = torch.sqrt(var + 1e-5)

        # Fix bias
        module.bias.data = ((module.bias.data - mean.flatten()) * init_scale /
                            scale.flatten())

        # Get correct dimension if transposed conv
        transp_conv = getattr(module, 'transposed', False)
        ch_out_dim = 1 if transp_conv else 0

        # Fix weight
        size = tuple(-1 if i == ch_out_dim else 1 for i in range(out.dim()))
        weight_size = module.weight.size()
        module.weight.data *= init_scale / scale.view(size)
        assert module.weight.size() == weight_size

        # Fix output in-place so we can continue forward pass
        out.data -= mean
        out.data *= init_scale / scale

        assert out.size() == out_size

    return hook


def data_dependent_init(model, inp, init_scale=.1):
    """Performs data-dependent initialization on a model.
    Updates each layer's weights such that its outputs, computed on a batch
    of actual data, have mean 0 and the same standard deviation. See the code
    for more details.
    Args:
        model (torch.nn.Module):
        model_input_dict (dict): Dictionary of inputs to the model.
        init_scale (float, optional): Desired scale (standard deviation) of
            each layer's output at initialization. Default: 0.1.
    """

    hook_handles = []
    modules = filter(lambda m: is_conv(m) or is_linear(m), model.modules())
    for module in modules:
        # Init module parameters before forward pass
        nn.init.kaiming_normal_(module.weight.data)
        module.bias.data.zero_()

        # Forward hook: data-dependent initialization
        hook_handle = module.register_forward_hook(
            _get_data_dep_hook(init_scale))
        hook_handles.append(hook_handle)

    # Forward pass one minibatch
    model(inp)  # dry-run

    # Remove forward hooks
    for hook_handle in hook_handles:
        hook_handle.remove()
