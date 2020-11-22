import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from logging_utils import Logger


def visualize_masks(imgs, masks, recons, logger, title="masks"):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0] // 2, c[1] // 2, c[2] // 2) for c in colors])
    colors.extend([(c[0] // 4, c[1] // 4, c[2] // 4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0

    full_imgs = np.concatenate((imgs, seg_maps, recons), 0).transpose((0, 2, 3, 1))
    columns = imgs.shape[0]
    rows = full_imgs.shape[0] // columns
    fig, ax = plt.subplots(rows, columns, figsize=(columns, rows))
    for column in range(columns):
        for row in range(rows):
            i = row * columns + column
            ax[row, column].grid(False)
            ax[row, column].axis('off')
            ax[row, column].imshow(full_imgs[i])

    plt.savefig(logger.get_sequential_figure_name(title), bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_figure(recons, originals, attention_regions, selected_experts, recons_steps,
                recons_steps_not_masked, logger, next_log_s=None, mask_recon=None, title=None):
    imgs = recons[:8]
    imgs = imgs.transpose((0, 2, 3, 1))
    originals = originals[:8]
    originals = originals.transpose((0, 2, 3, 1))

    columns = imgs.shape[0]
    len_log_s = 0
    if next_log_s is not None:
        len_log_s = len(next_log_s)
    len_mask_recon = 0
    if mask_recon is not None:
        len_mask_recon = len(mask_recon)
    rows = 2 + len(attention_regions) + len(recons_steps) + len(
        recons_steps_not_masked) + len_log_s + len_mask_recon
    fig, ax = plt.subplots(rows, columns, figsize=(columns, 1.6 * rows))
    for column in range(columns):
        axe = ax[1, column] if columns > 1 else ax[1]
        axe.grid(False)
        axe.axis('off')
        axe.imshow(imgs[column])
        axe.set_title("Recon")
        axe = ax[0, column] if columns > 1 else ax[0]
        axe.grid(False)
        axe.axis('off')
        axe.imshow(originals[column])
        axe.set_title("Orig")

    for att_id in range(len(attention_regions)):
        for column in range(columns):
            if columns > 1:
                axe = ax[2 + att_id, column]
                img = attention_regions[att_id][column]
                exp = selected_experts[att_id][column]
            else:
                axe = ax[2 + att_id]
                img = attention_regions[att_id]
                exp = selected_experts[att_id]
            axe.grid(False)
            axe.axis('off')
            axe.imshow(img.squeeze(), norm=NoNorm(), cmap='gray')
            axe.set_title("Atten\nEXP: {}".format(exp))

    for object_id in range(len(recons_steps)):
        for column in range(columns):
            if columns > 1:
                axe = ax[2 + len(attention_regions) + object_id, column]
                img = recons_steps[object_id][column].transpose((1, 2, 0))
                exp = selected_experts[object_id][column]
            else:
                axe = ax[2 + len(attention_regions) + object_id]
                img = recons_steps[object_id].transpose((1, 2, 0))
                exp = selected_experts[object_id]

            axe.grid(False)
            axe.axis('off')
            axe.imshow(img)
            axe.set_title("Recon\nEXP: {}".format(exp))

    for object_id in range(len(recons_steps_not_masked)):
        for column in range(columns):
            if columns > 1:
                axe = ax[2 + len(attention_regions) + len(recons_steps) + object_id, column]
                img = recons_steps_not_masked[object_id][column].transpose((1, 2, 0))
                exp = selected_experts[object_id][column]
            else:
                axe = ax[2 + len(attention_regions) + len(recons_steps) + object_id]
                img = recons_steps_not_masked[object_id].transpose((1, 2, 0))
                exp = selected_experts[object_id]
            axe.grid(False)
            axe.axis('off')
            axe.imshow(img)
            axe.set_title("No Mask\nEXP: {}".format(exp))

    if next_log_s is not None:
        for object_id in range(len(next_log_s)):
            for column in range(columns):
                if columns > 1:
                    axe = ax[2 + len(attention_regions) + len(recons_steps) + len(
                        recons_steps_not_masked) + object_id, column]
                    img = next_log_s[object_id][column].transpose((1, 2, 0))
                    exp = selected_experts[object_id][column]
                else:
                    axe = ax[2 + len(attention_regions) + len(recons_steps) + len(
                        recons_steps_not_masked) + object_id]
                    img = next_log_s[object_id].transpose((1, 2, 0))
                    exp = selected_experts[object_id]
                axe.grid(False)
                axe.axis('off')
                axe.imshow(np.exp(img).squeeze(), norm=NoNorm(), cmap='gray')
                axe.set_title("Scope\nEXP: {}".format(exp))
    if mask_recon is not None:
        for object_id in range(len(mask_recon)):
            for column in range(columns):
                if columns > 1:
                    axe = ax[2 + len(attention_regions) + len(recons_steps) + len(
                        recons_steps_not_masked) + len_log_s + object_id, column]
                    img = mask_recon[object_id][column].transpose((1, 2, 0))
                    exp = selected_experts[object_id][column]
                else:
                    axe = ax[2 + len(attention_regions) + len(recons_steps) + len(
                        recons_steps_not_masked) + len_log_s + + object_id]
                    img = mask_recon[object_id].transpose((1, 2, 0))
                    exp = selected_experts[object_id]
                axe.grid(False)
                axe.axis('off')
                axe.imshow(img.squeeze(), norm=NoNorm(), cmap='gray')
                axe.set_title("VAE m\nEXP: {}".format(exp))

    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    if title is None:
        title = "ECON"
    plt.savefig(logger.get_sequential_figure_name(title), bbox_inches="tight")
    plt.close()


def plot_loss_history(loss_history, val_loss_history, loss_history_per_object, logger):
    fig, ax = plt.subplots(1, len(loss_history.keys()), figsize=(40, 10))
    for i, param in enumerate(loss_history.keys()):

        ax[i].set_title("{}".format(param))
        if len(loss_history[param]["values"]) > 0:
            ax[i].plot(loss_history[param]["time"], loss_history[param]["values"],
                       label="Training")
        if len(val_loss_history[param]["values"]) > 0:
            ax[i].plot(val_loss_history[param]["time"],
                       val_loss_history[param]["values"],
                       label="Validation")
        ax[i].legend(frameon=False)
    plt.savefig(logger.get_sequential_figure_name("ECON_LOSS"), bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, len(loss_history_per_object.keys()), figsize=(40, 10))
    for i, param in enumerate(loss_history_per_object.keys()):
        print(i)
        ax[i].set_title("{}".format(param))
        if len(loss_history_per_object[param]["values"]) > 0:
            for obj in range(len(loss_history_per_object[param]["values"][0])):
                ax[i].plot(loss_history_per_object[param]["time"], np.array(loss_history_per_object[param]["values"])[:,obj], label="Obj: {}".format(obj))

        ax[i].legend(frameon=False)
    plt.savefig(logger.get_sequential_figure_name("ECON_LOSS_per_obj"), bbox_inches="tight")
    plt.close()
