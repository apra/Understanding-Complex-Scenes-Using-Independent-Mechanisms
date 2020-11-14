# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
# import visdom

import os

import monet_model
import monet_genesis
import econ_model
import datasets
import config

from logging_utils import Logger
import visualize

import matplotlib.pyplot as plt

# vis = visdom.Visdom()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Logger.cluster_log(device)


def numpify(tensor):
    return tensor.cpu().detach().numpy()


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
                Logger.error("error to_np")
                return x[i]
        return x
    else:
        try:
            return x.detach().cpu().numpy()
        except AttributeError:
            Logger.error("error to_np")
            return x


def run_training(monet, params, trainloader, logger):
    checkpoint_file = os.path.join(logger.checkpoints_dir, "checkpoint.ckpt")
    if params["load_parameters"] and os.path.isfile(checkpoint_file):
        monet.load_state_dict(torch.load(checkpoint_file))
        Logger.cluster_log('Restored parameters from', checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        Logger.cluster_log('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(params["num_epochs"]):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, counts = data
            images = images.cuda()
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % params["vis_every"] == 0:
                Logger.cluster_log('[%d, %5d] loss: %.3f' %
                                   (epoch + 1, i + 1, running_loss / params["vis_every"]))
                running_loss = 0.0
                visualize.visualize_masks(numpify(images[:8]),
                                          numpify(output['masks'][:8]),
                                          numpify(output['reconstructions'][:8]), logger=logger)

        torch.save(monet.state_dict(), checkpoint_file)

    Logger.cluster_log('training done')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm


def convert_to_np_im(torch_tensor, batch_idx=0):
    return np.moveaxis(torch_tensor.data.numpy()[batch_idx], 0, -1)


def plot(axes, ax1, ax2, tensor=None, title=None, grey=False, axis=False,
         fontsize=4):
    if tensor is not None:
        im = convert_to_np_im(tensor)
        if grey:
            im = im[:, :, 0]
            axes[ax1, ax2].imshow(im, norm=NoNorm(), cmap='gray')
        else:
            axes[ax1, ax2].imshow(im)
    if not axis:
        axes[ax1, ax2].axis('off')
    else:
        axes[ax1, ax2].set_xticks([])
        axes[ax1, ax2].set_yticks([])
    if title is not None:
        axes[ax1, ax2].set_title(title, fontsize=fontsize)
    # axes[ax1, ax2].set_aspect('equal')


def run_training_genesis(monet, params, trainloader, logger, cfg):
    checkpoint_file = os.path.join(logger.checkpoints_dir, "checkpoint.ckpt")
    # if params["load_parameters"] and os.path.isfile(checkpoint_file):
    #     monet.load_state_dict(torch.load(checkpoint_file))
    #     Logger.cluster_log('Restored parameters from', checkpoint_file)
    # else:
    #     for w in monet.parameters():
    #         std_init = 0.01
    #         nn.init.normal_(w, mean=0., std=std_init)
    #     Logger.cluster_log('Initialized parameters')

    optimizer = optim.Adam(monet.parameters(), lr=0.0001)

    for epoch in range(params["num_epochs"]):
        running_loss = 0.0
        running_kl = 0.
        for i, data in enumerate(trainloader, 0):
            monet.train()
            images, counts = data
            images = images.cuda()
            optimizer.zero_grad()
            output, losses, stats, att_stats, comp_stats = monet(images)

            # Reconstruction error
            err = losses.err.mean(0)
            # KL divergences
            kl_m, kl_l = torch.tensor(0), torch.tensor(0)
            # -- KL stage 1
            kl_m = losses.kl_m.mean(0)
            # -- KL stage 2
            kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()

            loss = err + 0.5 * (kl_l + kl_m)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            running_kl += (kl_l + kl_m).detach().cpu().item()

            if (i + 1) % params["vis_every"] == 0:
                # Visualise
                monet.eval()
                K_steps = cfg["k_steps"]
                num_images = 1
                # Forward pass
                output, _, stats, _, _ = monet(images)
                # Set up figure
                fig, axes = plt.subplots(nrows=4, ncols=1 + K_steps)

                # Input and reconstruction
                plot(axes, 0, 0, images.cpu(), title='Input image', fontsize=12)
                plot(axes, 1, 0, output.cpu(), title='Reconstruction', fontsize=12)
                # Empty plots
                plot(axes, 2, 0, fontsize=12)
                plot(axes, 3, 0, fontsize=12)

                # Put K reconstruction steps into separate subfigures
                x_k = stats['x_r_k']
                log_m_k = stats['log_m_k']
                mx_k = [x * m.exp() for x, m in zip(x_k, log_m_k)]
                log_s_k = stats['log_s_k'] if 'log_s_k' in stats else None
                for step in range(K_steps):
                    mx_step = mx_k[step]
                    x_step = x_k[step]
                    m_step = log_m_k[step].exp()
                    if log_s_k:
                        s_step = log_s_k[step].exp()

                    pre = 'Mask x RGB ' if step == 0 else ''
                    plot(axes, 0, 1 + step, mx_step.cpu(), pre + f'k={step + 1}', fontsize=12)
                    pre = 'RGB ' if step == 0 else ''
                    plot(axes, 1, 1 + step, x_step.cpu(), pre + f'k={step + 1}', fontsize=12)
                    pre = 'Mask ' if step == 0 else ''
                    plot(axes, 2, 1 + step, m_step.cpu(), pre + f'k={step + 1}', True, fontsize=12)
                    if log_s_k:
                        pre = 'Scope ' if step == 0 else ''
                        plot(axes, 3, 1 + step, s_step.cpu(), pre + f'k={step + 1}', True,
                             axis=step == 0, fontsize=12)

                # Beautify and show figure
                plt.subplots_adjust(wspace=0.05, hspace=0.15)
                plt.savefig(logger.get_sequential_figure_name("plot"), bbox_inches="tight")
                plt.show()

                Logger.cluster_log('[{}, {:>5}] loss: {:.3f} kl: {:3f}'.format(epoch + 1, i + 1,
                                                                               running_loss /
                                                                               params[
                                                                                   "vis_every"],
                                                                               running_kl / params[
                                                                                   "vis_every"]))
                running_loss = 0.0
                running_kl = 0.
            del images
            del output, losses, att_stats, comp_stats
        torch.save(monet.state_dict(), checkpoint_file)

    Logger.cluster_log('training done')


def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())


def test_tensor(ten, name="name"):
    print("{} ... max: {}, min: {}, mean: {}".format(name, torch.max(ten), torch.min(ten),
                                                     torch.mean(ten)))


import torch.autograd.profiler as profiler


def run_training_ECON(monet, params, trainloader, testloader, logger, device):
    checkpoint_file = os.path.join(logger.checkpoints_dir, "checkpoint.ckpt")
    if params["load_parameters"] and os.path.isfile(checkpoint_file):
        monet.load_state_dict(torch.load(checkpoint_file))
        Logger.cluster_log('Restored parameters from {}'.format(checkpoint_file))
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        Logger.cluster_log('Initialized parameters')

    optimizer = optim.Adam(monet.parameters(), lr=params["learning_rate"])

    loss_history = {
        "loss": {"values": [], "time": []},
        "loss_x_t": {"values": [], "time": []},
        "loss_r_t": {"values": [], "time": []},
        "loss_z_t": {"values": [], "time": []}
    }
    val_loss_history = {
        "loss": {"values": [], "time": []},
        "loss_x_t": {"values": [], "time": []},
        "loss_r_t": {"values": [], "time": []},
        "loss_z_t": {"values": [], "time": []}
    }
    epochs = params["num_steps"] // len(trainloader) + 1
    #cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params["num_steps"]//params["vis_every"], eta_min=1e-6)
    # cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,
    #                                                           factor=1 / np.sqrt(10.))

    current_step = 0
    Logger.log("Epochs: {}".format(epochs))
    for epoch in range(epochs):
        running_loss = 0.
        cnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)//params["vis_every"], eta_min=1e-6)
        for data in trainloader:
            print(optimizer.param_groups[0]['lr'])
            torch.cuda.empty_cache()
            Logger.cluster_log("Step: {}".format(current_step))
            images, counts = data
            images = images.to(device)
            optimizer.zero_grad()
            output = monet(images)

            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()



            selected_experts = np.stack(output["selected_expert_per_object"])

            selected_results = {}

            for key in output["results"][0][0].keys():
                selected_results[key] = np.zeros((len(selected_experts), len(selected_experts[0]),
                                                  *output["results"][0][0][key][0].shape))
                for obj in range(len(selected_experts)):
                    for sample in range(len(selected_experts[0])):
                        selected_results[key][obj, sample] = \
                            output["results"][obj][selected_experts[obj][sample]][key][sample]

            for param in loss_history.keys():
                loss_history[param]["time"].append(current_step)
                loss_history[param]["values"].append(np.mean(selected_results[param]))

            running_loss += loss.cpu().item()
            del images
            del counts
            del output
            del loss

            if current_step % params["vis_every"] == 0:
                cnn_scheduler.step()
                with torch.no_grad():
                    monet.eval()
                    batch_val_loss_history = {}
                    for param in loss_history.keys():
                        batch_val_loss_history[param] = []

                    a = 0
                    for val_data in testloader:
                        images, counts = val_data

                        images = images.to(device)

                        output = monet(images)
                        selected_experts = np.stack(output["selected_expert_per_object"])
                        selected_results = {}

                        for key in output["results"][0][0].keys():
                            selected_results[key] = np.zeros((len(selected_experts),
                                                              len(selected_experts[0]),
                                                              *output["results"][0][0][
                                                                  key][0].shape))
                            for obj in range(len(selected_experts)):
                                for sample in range(len(selected_experts[0])):
                                    selected_results[key][obj, sample] = \
                                        output["results"][obj][selected_experts[obj][sample]][
                                            key][sample]
                        if a == 0:
                            a = 1
                            visualize.plot_figure(
                                recons=np.sum(selected_results["x_recon_t"], axis=0),
                                originals=images.detach().cpu().numpy(),
                                attention_regions=selected_results["region_attention"],
                                selected_experts=selected_experts,
                                recons_steps=selected_results["x_recon_t"],
                                recons_steps_not_masked=selected_results["x_recon_t_not_masked"],
                                logger=logger)

                        for param in loss_history.keys():
                            batch_val_loss_history[param].append(np.mean(selected_results[param]))
                        del images
                        del output

                    for param in val_loss_history.keys():
                        val_loss_history[param]["time"].append(current_step)
                        val_loss_history[param]["values"].append(
                            np.mean(batch_val_loss_history[param]))

                    Logger.cluster_log('[%3d, %5d] loss: %.3f' % (
                        epoch + 1, current_step + 1, running_loss / params["vis_every"]))

                    running_loss = 0.0

                    fig, ax = plt.subplots(1, len(loss_history.keys()), figsize=(40, 10))
                    for i, param in enumerate(loss_history.keys()):
                        ax[i].set_title("{}".format(param))
                        ax[i].plot(loss_history[param]["time"], loss_history[param]["values"],
                                   label="Training")
                        ax[i].plot(val_loss_history[param]["time"],
                                   val_loss_history[param]["values"],
                                   label="Validation")
                        ax[i].legend(frameon=False)
                    plt.savefig(logger.get_sequential_figure_name("ECON_LOSS"), bbox_inches="tight")
                    plt.close()

                    # plt.show()
                    if not params["dontstore"]:
                        torch.save(monet.state_dict(), checkpoint_file)
                    monet.train()

            current_step += 1


    Logger.cluster_log('training done')


def to_float(x):
    return x.float()


def sprite_experiment(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = cfg
    if "device" in params.keys():
        if params["device"] == "TPU":
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
    Logger.cluster_log(device)
    logger = Logger(params=params, version="0.0.1")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(to_float),
                                    ])
    trainset = datasets.Sprites(params["data_dir"],
                                rnd_background=params["rnd_bkg"],
                                train=True,
                                transform=transform,
                                n=params["num_samples"],
                                max_num_objs=params["max_num_objs"],
                                min_num_objs=params["min_num_objs"])
    testset = datasets.Sprites(params["data_dir"],
                               rnd_background=params["rnd_bkg"],
                               train=False,
                               transform=transform,
                               n=params["num_samples"],
                               max_num_objs=params["max_num_objs"],
                               min_num_objs=params["min_num_objs"])
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=params["batch_size"],
                                              shuffle=True,
                                              num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=params["batch_size"],
                                             shuffle=True,
                                             num_workers=0)
    if params["model_name"] == "genesis":
        monet = monet_genesis.Monet(cfg=params).to(device)
        run_training_genesis(monet, params, trainloader, logger=logger, cfg=params)
    elif params["model_name"] == "ECON":
        monet = econ_model.ECON(params=params).to(device)
        run_training_ECON(monet=monet, params=params, trainloader=trainloader,
                          testloader=testloader, logger=logger,
                          device=device)
    else:
        monet = monet_model.Monet(params=params, height=64, width=64).to(device)
        run_training(monet, params, trainloader, logger=logger)


seeds = [42, 24365517, 6948868, 96772882, 58236860, 7111973, 5016789, 19469290, 2384676, 10878630,
         26484779, 78421105, 46346829, 65958905, 69757054, 49361965, 84089155, 85116270, 8707926,
         26474437, 46028029]
if __name__ == '__main__':
    cfg = config.load_config()

    torch.manual_seed(seeds[cfg["seed"]])
    np.random.seed(seeds[cfg["seed"]])

    sprite_experiment(cfg)
