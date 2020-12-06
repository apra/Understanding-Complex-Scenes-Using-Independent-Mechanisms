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
import metric

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
    if params["load"] and os.path.isfile(checkpoint_file):
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


def sigmoid_annealing_(start_value, end_value, start_steps, end_steps, current_step):
    if current_step < start_steps:
        return start_value
    elif current_step > end_steps:
        return end_value
    else:
        return (1 / (1 + np.exp(
            -5 * (((current_step - start_steps) / (end_steps - start_steps)) - 0.5)
        )
                     )
                ) * (end_value - start_value) + start_value


def sigmoid_annealing(params, current_step):
    beta = sigmoid_annealing_(start_value=0., end_value=params["beta_loss"],
                              start_steps=params["annealing_start"],
                              end_steps=params["annealing_duration"] + params[
                                  "annealing_start"], current_step=current_step)
    gamma = sigmoid_annealing_(start_value=0., end_value=params["gamma_loss"],
                               start_steps=params["annealing_start"],
                               end_steps=params["annealing_duration"] + params[
                                   "annealing_start"], current_step=current_step)

    return beta, gamma


def store_average_progress(input, output, time, axis, avoid=[]):
    for param in output.keys():
        output[param]["time"].append(time)
        if param not in avoid:
            output[param]["values"].append(np.mean(input[param], axis=axis))


def get_selected_params(output, selected_experts):
    selected_results = {}
    num_objects = len(selected_experts)
    batch_size = len(selected_experts[0])
    for key in output["results"][0][0].keys():
        out_shape = output["results"][0][0][key][0].shape
        selected_results[key] = np.zeros((num_objects, batch_size, *out_shape))
        for obj in range(num_objects):
            for sample in range(batch_size):
                expert = selected_experts[obj][sample]
                selected_results[key][obj, sample] = output["results"][obj][expert][key][
                    sample]
    return selected_results


import seaborn as sns
def copyParams(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)

    for name, param in params_src:
        if name in dict_dest:
            dict_dest[name].data.copy_(param.data)


def run_training_ECON(monet, params, trainloader, testloader, logger, device):
    assert params["num_objects"] == len(params["sigmas_x"])
    # initialize the optimizer
    optimizer = optim.Adam(monet.parameters(),
                           lr=params["learning_rate"],
                           weight_decay=params["weight_decay"])
    # initialize the scheduler
    if params["scheduler"] == "plateau":
        cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   'min',
                                                                   patience=5,
                                                                   factor=1 / np.sqrt(10.))
    if params["load"]:
        state = logger.load_checkpoint()
        monet.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state.keys() and params["scheduler"] == "plateau":
            cnn_scheduler.load_state_dict(state["scheduler"])
        Logger.cluster_log('Restored parameters from {}'.format(logger.save_dir))
    else:
        if params["data_dep_init"]:
            # initialize network:
            steps = 5
            for i, data in enumerate(trainloader):
                if i > steps:
                    break
                images, counts = data
                images = images.to(device)
                optimizer.zero_grad()

                # forward pass
                output = monet.run_single_expert(images, expert_id=0, gamma=1., beta=1.)

                loss = torch.mean(output['loss'])

                # backward pass
                loss.backward()
                optimizer.step()

            for expert_id in range(1, params["num_experts"]):
                copyParams(monet.experts[0], monet.experts[expert_id])

            x, _ = next(iter(trainloader))
            x = x.to(device)
            for expert_id in range(0, params["num_experts"]):
                output = monet.run_single_expert(x, expert_id=0, gamma=1., beta=1.)
                selected_experts = [[0]*x.shape[0]]*params["num_objects"]
                for obj in range(len(output["results"])):
                    output["results"][obj] = [output["results"][obj]]

                selected_results = get_selected_params(output, selected_experts)
                visualize.plot_figure(
                    recons=np.sum(selected_results["x_recon_t"], axis=0),
                    originals=x.detach().cpu().numpy(),
                    attention_regions=selected_results["region_attention"],
                    selected_experts=selected_experts,
                    recons_steps=selected_results["x_recon_t"],
                    recons_steps_not_masked=selected_results["x_recon_t_not_masked"],
                    next_log_s=selected_results["log_s_t"],
                    mask_recon=selected_results["mask_recon"],
                    logger=logger,
                    title="initialization_test")

            # econ_model.data_dependent_init(monet, x.to(device))
            Logger.cluster_log("Finished data-dependent initialization")
        else:
            for w in monet.parameters():
                std_init = 0.01
                nn.init.normal_(w, mean=0., std=std_init)
            Logger.cluster_log('Initialized parameters')

    loss_history = {
        "loss": {"values": [], "time": []},
        "loss_x_t": {"values": [], "time": []},
        "loss_r_t": {"values": [], "time": []},
        "loss_z_t": {"values": [], "time": []},
        # "per_object_loss": {"values": [], "time": []}
    }
    loss_history_per_object = {
        "loss_x_t": {"values": [], "time": []},
        "loss_r_t": {"values": [], "time": []},
        "loss_z_t": {"values": [], "time": []},
        "competition_objective": {"values": [], "time": []},
    }
    val_loss_history = {
        "loss": {"values": [], "time": []},
        "loss_x_t": {"values": [], "time": []},
        "loss_r_t": {"values": [], "time": []},
        "loss_z_t": {"values": [], "time": []},
        # "per_object_loss": {"values": [], "time": []},
    }
    epochs = params["num_steps"] // len(trainloader) + 1

    current_step = 0
    Logger.log("Epochs: {}".format(epochs))
    beta = None
    gamma = None
    best_loss = np.infty

    for epoch in range(epochs):
        if params["scheduler"] == "cosann":
            cnn_scheduler = optim.lr_scheduler. \
                CosineAnnealingLR(optimizer,
                                  len(trainloader) // params["vis_every"] + 1,
                                  eta_min=1e-6)
        # start epoch
        for data in trainloader:
            Logger.cluster_log("Step: {}".format(current_step))

            # perform annealing if necessary
            if params["annealing_start"] > 0:
                beta, gamma = sigmoid_annealing(params, current_step)
            Logger.log(
                "Learning rate: {}. Beta: {}. Gamma: {}.".format(optimizer.param_groups[0]['lr'],
                                                                 beta, gamma))

            torch.cuda.empty_cache()
            # load images
            images, counts = data
            images = images.to(device)
            optimizer.zero_grad()
            # forward pass
            output = monet(images, beta, gamma)

            loss = torch.mean(output['loss'])

            # backward pass
            loss.backward()
            optimizer.step()

            selected_experts = np.stack(output["selected_expert_per_object"])

            selected_results = get_selected_params(output, selected_experts)

            store_average_progress(selected_results,
                                   loss_history,
                                   time=current_step,
                                   axis=(0, 1),
                                   avoid=["per_object_list", "loss"])

            loss_history["loss"]["values"].append(np.mean(loss.detach().cpu().numpy()))

            store_average_progress(selected_results,
                                   loss_history_per_object,
                                   time=current_step,
                                   axis=1)

            del images
            del counts
            del output
            del loss
            del selected_results

            # Evaluate performance
            if current_step % params["vis_every"] == 0:
                if params["scheduler"] == "cosann":
                    cnn_scheduler.step()

                with torch.no_grad():
                    scores = []
                    selected_object_per_expert = None
                    monet.eval()

                    selected_expert_per_object_frequency = []
                    for _ in range(params["num_objects"]):
                        selected_expert_per_object_frequency.append([])

                    batch_val_loss_history = {}
                    for param in loss_history.keys():
                        batch_val_loss_history[param] = []

                    a = -1
                    current_loss = 0
                    for val_data in testloader:
                        a += 1
                        if a > max(current_step, params["vis_every"]):
                            break
                        images, segmentation_mask = val_data

                        images = images.to(device)

                        output = monet(images)

                        current_loss += torch.mean(output["loss"]).detach().cpu().numpy()

                        selected_experts = np.stack(output["selected_expert_per_object"])
                        for i, exps in enumerate(output["selected_expert_per_object"]):
                            selected_expert_per_object_frequency[i].extend(exps)

                        # get results for selected experts
                        selected_results = get_selected_params(output, selected_experts)

                        if len(segmentation_mask)>0:
                            batch_scores, batch_selected_object_per_expert = \
                                metric.compute_segmentation_covering_expert_score(params=params,
                                                                                  attentions=selected_results["region_attention"],
                                                                                  selected_experts=selected_experts,
                                                                                  recons_t=selected_results["x_recon_t"],
                                                                                  segmentation_mask=segmentation_mask.detach().cpu().numpy())
                            scores.extend(list(batch_scores))

                            if selected_object_per_expert is None:
                                selected_object_per_expert = batch_selected_object_per_expert.astype(int)
                            else:
                                selected_object_per_expert = np.hstack((selected_object_per_expert,batch_selected_object_per_expert)).astype(int)


                        if a == 0:
                            visualize.plot_figure(
                                recons=np.sum(selected_results["x_recon_t"], axis=0),
                                originals=images.detach().cpu().numpy(),
                                attention_regions=selected_results["region_attention"],
                                selected_experts=selected_experts,
                                recons_steps=selected_results["x_recon_t"],
                                recons_steps_not_masked=selected_results["x_recon_t_not_masked"],
                                next_log_s=selected_results["log_s_t"],
                                mask_recon=selected_results["mask_recon"],
                                logger=logger)

                        for param in loss_history.keys():
                            if not (param == "per_object_loss" or param == "loss"):
                                batch_val_loss_history[param].append(
                                    np.mean(selected_results[param]))
                        batch_val_loss_history["loss"].append(
                            np.mean(output["loss"].detach().cpu().numpy()))

                        # batch_val_loss_history["per_object_loss"].append(
                        #     np.mean(selected_results["loss_x_t"], axis=1))
                        del images
                        del output
                        del selected_results

                    # find the best score, this is the current score of the model
                    logger.log_to_file("{}: {}\n".format(current_step, np.mean(scores)), "segmentation_score.log")
                    print("MAX SCORE: {}".format(np.max(scores)))

                    store_average_progress(batch_val_loss_history,
                                           val_loss_history,
                                           time=current_step,
                                           axis=None)

                    # val_loss_history["per_object_loss"]["values"].append(
                    #     np.mean(batch_val_loss_history["per_object_loss"],
                    #             axis=0))

                    visualize.plot_loss_history(loss_history=loss_history,
                                                val_loss_history=val_loss_history,
                                                loss_history_per_object=loss_history_per_object,
                                                logger=logger)
                    fig, ax = plt.subplots(params["num_objects"], 1,
                                           figsize=(7, 2 * params["num_objects"]))
                    bins = np.array(range(params["num_experts"] + 1)) - 0.5
                    plt.title("Training steps: {}".format(current_step))
                    for i in range(params["num_objects"]):
                        ax[i].set_title(
                            "Distribution of experts selected for object {}".format(i + 1))
                        sns.distplot(selected_expert_per_object_frequency[i], bins=bins, ax=ax[i],
                                     norm_hist=True, kde=False)
                        ax[i].set_xticks((bins[:-1] + 0.5).astype(int))
                    plt.subplots_adjust(hspace=0.5)
                    plt.savefig(logger.get_sequential_figure_name("selected_experts_histogram"),
                                bbox_inches="tight")
                    plt.close()

                    if params["name_config"] == "ECON_sprite":
                        segmentation_colors = ["Green", "Blue", "Red"]
                    elif params["name_config"] == "ECON_coinrun":
                        segmentation_colors = ["Main Character", "Boxes", "Ground", "Enemies"]
                    else:
                        segmentation_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
                    fig, ax = plt.subplots(params["num_experts"], 1,
                                           figsize=(4, 2 * params["num_experts"]))
                    bins = np.array(range(1, np.max(selected_object_per_expert) + 2)) - 0.5
                    plt.title("Training steps: {}".format(current_step))
                    for i in range(params["num_experts"]):
                        ax[i].set_title(
                            "Distribution of objects for expert {}".format(i))
                        sns.distplot(selected_object_per_expert[i], bins=bins, ax=ax[i],
                                     norm_hist=True, kde=False)
                        ax[i].set_xticks((bins[:-1] + 0.5).astype(int))
                        ax[i].set_xticklabels(segmentation_colors)
                    plt.subplots_adjust(hspace=0.5)
                    plt.savefig(logger.get_sequential_figure_name("selected_objects_per_expert"),
                                bbox_inches="tight")
                    plt.close()

                    # compute the validation loss
                    current_loss /= a

                    Logger.cluster_log('[%3d, %5d] val loss: %.3f' % (
                        epoch + 1, current_step + 1, current_loss))

                    if not params["dontstore"]:
                        logger.store_checkpoint(model=monet, optimizer=optimizer,
                                                steps=current_step)

                    if current_loss < best_loss:
                        best_loss = current_loss
                        Logger.cluster_log("LOSS IMPROVED: {}".format(best_loss))

                    if params["scheduler"] == "plateau":
                        cnn_scheduler.step(best_loss)

                    monet.train()

            current_step += 1

    Logger.cluster_log('training done')


def to_float(x):
    return x.float()


def run_model_training(params, trainloader, logger, testloader=None, model_name="ECON"):
    if model_name == "genesis":
        monet = monet_genesis.Monet(cfg=params).to(device)
        run_training_genesis(monet, params, trainloader, logger=logger, cfg=params)
    elif model_name == "ECON":
        monet = econ_model.ECON(params=params).to(device)
        run_training_ECON(monet=monet, params=params, trainloader=trainloader,
                          testloader=testloader, logger=logger,
                          device=device)
    else:
        monet = monet_model.Monet(params=params, height=64, width=64).to(device)
        run_training(monet, params, trainloader, logger=logger)


def load_sprite(params):
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
    return trainloader, testloader


def load_coinrun(params):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(to_float),
                                    ])
    trainset = datasets.Coinrun(params["data_dir"],
                                dataset_name=params["dataset_name"],
                                train=True,
                                transform=transform)
    testset = datasets.Coinrun(params["data_dir"],
                               dataset_name=params["dataset_name"],
                               train=False,
                               transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=params["batch_size"],
                                              shuffle=True,
                                              num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=params["batch_size"],
                                             shuffle=True,
                                             num_workers=0)
    return trainloader, testloader


seeds = [42, 24365517, 6948868, 96772882, 58236860, 7111973, 5016789, 19469290, 2384676, 10878630,
         26484779, 78421105, 46346829, 65958905, 69757054, 49361965, 84089155, 85116270, 8707926,
         26474437, 46028029]
if __name__ == '__main__':
    params = config.load_config()

    # select the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.cluster_log("DEVICE: {}".format(device))

    # initialize the logger
    logger = Logger(params=params)

    if params["load"]:
        # store the relevant input parameters
        steps = params["num_steps"]
        exp_folder = params["exp_folder"]
        # load all the parameters from the folder
        params = logger.load_parameters()
        params['num_steps'] = steps
        params["exp_folder"] = exp_folder

    torch.manual_seed(seeds[params["seed"]])
    np.random.seed(seeds[params["seed"]])

    if params["name_config"] == "ECON_sprite":
        trainloader, testloader = load_sprite(params)
    elif params["name_config"] == "ECON_coinrun":
        trainloader, testloader = load_coinrun(params)
    else:
        trainloader, testloader = None, None

    run_model_training(params=params,
                       trainloader=trainloader,
                       testloader=testloader,
                       logger=logger,
                       model_name=params["model_name"])
