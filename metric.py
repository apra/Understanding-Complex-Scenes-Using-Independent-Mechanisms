import numpy as np
import itertools
import matplotlib.pyplot as plt

def compute_segmentation_mask_score(params, attentions, selected_experts, recons_t, originals,
                                    segmentation_mask):
    batch_size = attentions.shape[1]
    num_objects = params["num_objects"]
    num_experts = params["num_experts"]
    if params["name_config"] == "ECON_sprite":
        available_colors = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
    elif params["name_config"] == "ECON_coinrun":
        available_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]]
    else:
        available_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]]

    width = attentions.shape[3]
    height = attentions.shape[4]
    color_permutations = itertools.permutations(available_colors, num_experts)
    color_permutations =  list(color_permutations)
    performance_combination = {}
    for color_combination_id, color_combination in enumerate(color_permutations):
        print(color_combination)
        object_colors = {

        }
        expert = 0
        for color in color_combination:
            object_colors[expert] = np.zeros((3, width, height))
            object_colors[expert][0, :] = color[0]
            object_colors[expert][1, :] = color[1]
            object_colors[expert][2, :] = color[2]
            expert += 1

        avg_score = 0
        for image_id in range(batch_size):
            final_image = np.zeros((3, width, height))
            for object in range(num_objects):
                partial = (attentions[object][image_id]>=0.5) * object_colors[
                    selected_experts[object][image_id]] * (recons_t[object][image_id] > 1e-1)
                final_image += partial

            fig, ax = plt.subplots(1, 2)
            plt.title(color_combination)
            ax[0].imshow(np.moveaxis(segmentation_mask[image_id], 0, 2))
            ax[1].imshow(final_image.transpose((1, 2, 0)))
            plt.show()
            plt.pause(0.4)
            plt.close()

            score = np.mean(np.square(
                np.linalg.norm(final_image - segmentation_mask[image_id], "fro", axis=(1, 2))))
            avg_score += score

        # fig, ax = plt.subplots(1, 2)
        # plt.title(color_combination)
        # ax[0].imshow(np.moveaxis(segmentation_mask[batch_size - 1], 0, 2))
        # ax[1].imshow(final_image.transpose((1, 2, 0)))
        # plt.show()
        # plt.pause(0.4)
        # plt.close()
        avg_score /= batch_size
        print(avg_score)
        performance_combination[color_combination_id] = avg_score
        # print(attentions)

    return performance_combination, color_permutations


def iou_binary(target_mask, prediction_mask):
    intersection = (np.sum(target_mask * prediction_mask, axis=(1,2,3))).astype(float)
    union = (np.sum(target_mask + prediction_mask, axis=(1,2,3))).astype(float)
    # Return -100 if union is zero, else return generalized IOU
    return np.where(union<1e-5, -100.0, intersection / union)


def convert_image_to_segmentation_mask(image, objects_color):
    final = np.zeros((image.shape[0],64, 64, 1))
    for i, color in enumerate(objects_color):
        final[np.expand_dims(np.all(image[:, :] == color, axis=3), 3)] = i+1
    return final


import seaborn as sns


def compute_segmentation_covering_expert_score(params, attentions, selected_experts, recons_t,
                                        segmentation_mask):
    batch_size = attentions.shape[1]
    num_objects = params["num_objects"]
    num_experts = params["num_experts"]
    if params["name_config"] == "ECON_sprite":
        segmentation_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    elif params["name_config"] == "ECON_coinrun":
        segmentation_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0]]
    else:
        segmentation_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    width = attentions.shape[3]
    height = attentions.shape[4]
    expert_id = np.zeros((num_experts, 1))
    for expert in range(num_experts):
        expert_id[expert][0] = expert+1

    expert_segmentation = np.zeros((batch_size, 1, width, height))
    for object in range(num_objects):
        selected_experts_id = np.expand_dims(np.repeat(np.expand_dims(np.repeat(expert_id[selected_experts[object]],64,axis=1),1),64,axis=1),1)
        foreground = np.expand_dims(np.any((recons_t[object] > 1e-1),axis=1),1)
        partial = (attentions[object] >= 0.5) * selected_experts_id * foreground

        expert_segmentation += partial

    fig,ax = plt.subplots()
    ax.imshow(expert_segmentation[0].transpose(1,2,0))
    plt.show()

    ground_truth_objects = convert_image_to_segmentation_mask(segmentation_mask.transpose(0,2,3,1),segmentation_colors)
    scores = np.zeros((batch_size, ))
    total_area = np.zeros((batch_size, ))

    expert_predicted_object = np.zeros((num_experts,batch_size))

    for expert_id in range(1, int(np.max(expert_segmentation))+1):
        pred_obj_region = expert_segmentation.transpose(0,2,3,1) == expert_id
        if np.sum(pred_obj_region) == 0:
            continue

        N = np.expand_dims(np.sum(pred_obj_region, axis=(1,2,3)),1)

        best_iou = np.zeros((batch_size, 1))
        best_objects = np.zeros((batch_size, 1))
        for ground_truth_object_id in range(1, int(np.max(ground_truth_objects))+1):
            gt_obj_region = ground_truth_objects == ground_truth_object_id
            if np.sum(gt_obj_region) == 0:
                continue
            iou = iou_binary(gt_obj_region, pred_obj_region)
            best_objects = np.where(np.greater(np.squeeze(iou), np.squeeze(best_iou)),
                                    ground_truth_object_id, np.squeeze(best_objects))
            best_iou = np.where(np.greater(np.squeeze(iou), np.squeeze(best_iou)), np.squeeze(iou), np.squeeze(best_iou))

        scores += np.squeeze(N)*best_iou
        print(np.max(best_iou))
        total_area += np.squeeze(N)
        expert_predicted_object[expert_id-1] = best_objects

        # fig, ax = plt.subplots()
        # plt.title("Expert: {}".format(expert_id))
        # sns.distplot(best_objects, ax=ax)
        # plt.show()

    score = scores/total_area

    # for img in range(score.shape[0]):
    #     fig, ax = plt.subplots(1,2)
    #     plt.title("Score: {}".format(score[img]))
    #     ax[0].imshow(segmentation_mask[img].transpose(1,2,0))
    #     ax[1].imshow(expert_segmentation[img].transpose(1, 2, 0))
    #     plt.show()

    return score, expert_predicted_object