import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

import datasets
import econ_model
import metric
import visualize
from logging_utils import Logger


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


def compute_segmentation_mask_score(params, num_objects, logger):
    monet = econ_model.ECON(params=params).to(device)
    state = logger.load_checkpoint()
    print(list(state.keys()))
    monet.load_state_dict(state["model"])

    monet.num_objects = num_objects

    current_step = -1
    scores = []
    selected_object_per_expert = None
    selected_expert_per_object_frequency = []
    for _ in range(num_objects):
        selected_expert_per_object_frequency.append([])
    for data in trainloader:
        current_step += 1
        print("Current step: ", current_step)
        if current_step > ((params["num_samples"] // params["batch_size"]) + 1):
            break
        images, segmentation_mask = data
        images = images.to(device)

        # forward pass
        output = monet(images, 0, 0)

        for i, exps in enumerate(output["selected_expert_per_object"]):
            selected_expert_per_object_frequency[i].extend(exps)

        selected_experts = np.stack(output["selected_expert_per_object"])

        selected_results = get_selected_params(output, selected_experts)

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
            selected_object_per_expert = np.hstack(
                (selected_object_per_expert, batch_selected_object_per_expert)).astype(int)

        if current_step < 10:
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
    bins = np.array(range(1, len(segmentation_colors) + 2)) - 0.5
    plt.title("Training steps: {}".format(current_step))
    for i in range(params["num_experts"]):
        ax[i].set_title(
            "Distribution of objects for expert {}".format(i))
        sns.distplot(selected_object_per_expert[i][selected_object_per_expert[i] > 0], bins=bins,
                     ax=ax[i],
                     norm_hist=True, kde=False)
        ax[i].set_xticks((bins[:-1] + 0.5).astype(int))
        ax[i].set_xticklabels(segmentation_colors)
    plt.subplots_adjust(hspace=0.5)
    print(logger.get_sequential_figure_name("selected_objects_per_expert"))
    plt.savefig(logger.get_sequential_figure_name("selected_objects_per_expert"),
                bbox_inches="tight")

    fig, ax = plt.subplots(num_objects, 1,
                           figsize=(7, 2 * num_objects))
    bins = np.array(range(params["num_experts"] + 1)) - 0.5
    plt.title("Training steps: {}".format(current_step))
    for i in range(num_objects):
        ax[i].set_title(
            "Distribution of experts selected for object {}".format(i + 1))
        sns.distplot(selected_expert_per_object_frequency[i], bins=bins, ax=ax[i],
                     norm_hist=True, kde=False)
        ax[i].set_xticks((bins[:-1] + 0.5).astype(int))
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(logger.get_sequential_figure_name("selected_experts_histogram"),
                bbox_inches="tight")
    plt.close()

    logger.log_to_file("{}\n".format(np.mean(scores)), "segmentation_score.log")


seeds = [42, 24365517, 6948868, 96772882, 58236860, 7111973, 5016789, 19469290, 2384676, 10878630,
         26484779, 78421105, 46346829, 65958905, 69757054, 49361965, 84089155, 85116270, 8707926,
         26474437, 46028029]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--is_batch',
                        action="store_true",
                        default=False,
                        help="Whether this is a batch job.")

    cfgs, namespace = parser.parse_known_args()
    params = {}
    for name, val in vars(cfgs).items():
        print(name)
        params[name] = val

    params["load"] = True

    num_samples = params["num_samples"]
    max_num_objs = params["max_num_objs"]
    min_num_objs = params["min_num_objs"]
    num_objects = params["num_objects"]
    dataset_name = params["dataset_name"]

    # select the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.cluster_log("DEVICE: {}".format(device))

    if params["is_batch"]:
        batch_folder = params["exp_folder"]
        params["log_dir"] = batch_folder
        exp_list = [x for x in os.listdir(batch_folder) if
                    os.path.isdir(os.path.join(batch_folder, x))]
    else:
        params["log_dir"] = "experiments"
        exp_list = [params["exp_folder"]]

    log_dir = params["log_dir"]

    for exp_folder in exp_list:
        params["load"] = True
        print(params)
        params["exp_folder"] = exp_folder
        params["log_dir"] = log_dir
        print("EXP:", exp_folder)
        print("LOGDIR:", params["log_dir"])
        # initialize the logger
        logger = Logger(params=params)

        logger.log_to_file("{}\n".format("test"), "segmentation_score.log")

        # load all the parameters from the folder
        params = logger.load_parameters()
        params["exp_folder"] = exp_folder
        params["log_dir"] = log_dir
        print(params)

        params["num_samples"] = num_samples
        params["max_num_objs"] = max_num_objs
        params["min_num_objs"] = min_num_objs
        params["dataset_name"] = dataset_name
        params["batch_size"] = 16

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

        compute_segmentation_mask_score(params, num_objects, logger)
