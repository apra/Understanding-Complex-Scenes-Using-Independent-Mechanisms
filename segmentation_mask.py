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

import argparse
import string
import ast
import metric

# vis = visdom.Visdom()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Logger.cluster_log(device)


def to_float(x):
    return x.float()


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

import itertools


def test_generalization(params, num_objects, logger):
    monet = econ_model.ECON(params=params).to(device)
    state = logger.load_checkpoint()
    print(list(state.keys()))
    monet.load_state_dict(state["model"])

    monet.num_objects = num_objects
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
    current_step = -1
    scores = []
    selected_object_per_expert = None
    for data in trainloader:
        current_step += 1
        if current_step > params["num_samples"]:
            break
        images, segmentation_mask = data
        images = images.to(device)
        # forward pass
        output = monet(images, 0, 0)

        loss = torch.mean(output['loss'])

        # backward pass
        loss.backward()

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

        # if current_step == 0:
        batch_scores, batch_selected_object_per_expert = \
            metric.compute_segmentation_covering_expert_score(params=params,
                                                              attentions=selected_results[
                                                                  "region_attention"],
                                                              selected_experts=selected_experts,
                                                              recons_t=selected_results[
                                                                  "x_recon_t"],
                                                              segmentation_mask=segmentation_mask.detach().cpu().numpy())
        scores.extend(list(batch_scores))
        if selected_object_per_expert is None:
            selected_object_per_expert = batch_selected_object_per_expert.astype(int)
        else:
            selected_object_per_expert = np.hstack((selected_object_per_expert,batch_selected_object_per_expert)).astype(int)


        visualize.plot_figure(
            recons=np.sum(selected_results["x_recon_t"], axis=0),
            originals=images.detach().cpu().numpy(),
            attention_regions=selected_results["region_attention"],
            selected_experts=selected_experts,
            recons_steps=selected_results["x_recon_t"],
            recons_steps_not_masked=selected_results["x_recon_t_not_masked"],
            next_log_s=selected_results["log_s_t"],
            mask_recon=selected_results["mask_recon"],
            logger=logger,
            title="generalization")
    if params["name_config"] == "ECON_sprite":
        segmentation_colors = ["Red", "Blue", "Green"]
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
        sns.distplot(selected_object_per_expert[i][selected_object_per_expert[i]>0], bins=bins, ax=ax[i],
                     norm_hist=True, kde=False)
        ax[i].set_xticks((bins[:-1] + 0.5).astype(int))
        ax[i].set_xticklabels(segmentation_colors)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(logger.get_sequential_figure_name("selected_objects_per_expert"),
                bbox_inches="tight")
    plt.close()

    logger.log_to_file("{}\n".format(np.mean(scores)), "segmentation_score.log")


seeds = [42, 24365517, 6948868, 96772882, 58236860, 7111973, 5016789, 19469290, 2384676, 10878630,
         26484779, 78421105, 46346829, 65958905, 69757054, 49361965, 84089155, 85116270, 8707926,
         26474437, 46028029]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_folder',
                        type=str,
                        metavar='EXPFOLDER',
                        default="brrr",
                        help="Folder of the experiment.")
    parser.add_argument('--dataset_name',
                        type=str,
                        metavar='DATASETNAME',
                        default="coirun_dataset_generalization_2.hdf5",
                        help="Name of the dataset for coinrun.")
    parser.add_argument('--num_samples',
                        type=int,
                        metavar='NUMSAMPLES',
                        default=100,
                        help="Number of samples to test on.")
    parser.add_argument('--max_num_objs',
                        type=int,
                        metavar='MAXNUMOBJS',
                        default=4,
                        help="NMaximum number of objects.")
    parser.add_argument('--min_num_objs',
                        type=int,
                        metavar='MINNUMOBJS',
                        default=4,
                        help="Minimum number of objects.")
    parser.add_argument('--num_objects',
                        type=int,
                        metavar='NUMOBJECTS',
                        default=4,
                        help="Number of objects.")

    cfgs, namespace = parser.parse_known_args()
    params = {}
    for name, val in vars(cfgs).items():
        print(name)
        params[name] = val

    params["load"] = True
    params["log_dir"] = "experiments"

    num_samples = params["num_samples"]
    max_num_objs = params["max_num_objs"]
    min_num_objs = params["min_num_objs"]
    num_objects = params["num_objects"]
    dataset_name = params["dataset_name"]

    # select the available device
    device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.cluster_log("DEVICE: {}".format(device))

    # initialize the logger
    logger = Logger(params=params)

    # load all the parameters from the folder
    params = logger.load_parameters()

    params["num_samples"] = num_samples
    params["max_num_objs"] = max_num_objs
    params["min_num_objs"] = min_num_objs
    params["dataset_name"] = dataset_name
    params["batch_size"] = 4

    sigmas_x = [1] * num_objects
    params["sigmas_x"] = sigmas_x

    torch.manual_seed(seeds[params["seed"]])
    np.random.seed(seeds[params["seed"]])

    if params["name_config"] == "ECON_sprite":
        trainloader, testloader = load_sprite(params)
    elif params["name_config"] == "ECON_coinrun":
        trainloader, testloader = load_coinrun(params)
    else:
        trainloader, testloader = None, None

    test_generalization(params, num_objects, logger)
