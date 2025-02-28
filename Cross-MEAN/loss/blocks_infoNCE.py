import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable


def get_heartmap_pool(part_features, blocks=3, add_global=False, otherbranch=False):
    heatmap = torch.mean(part_features, dim=-1)
    size = part_features.size(1)
    arg = torch.argsort(heatmap, dim=1, descending=True)
    x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
    x_sort = torch.stack(x_sort, dim=0)

    # -- 按照地物自动聚类的类别数来将16*16的区域进行分类
    split_each = size / blocks
    split_list = [int(split_each) for i in range(blocks - 1)]
    split_list.append(size - sum(split_list))
    split_x = x_sort.split(split_list, dim=1)

    split_list = [torch.mean(split, dim=1) for split in split_x]
    part_featuers_ = torch.stack(split_list, dim=2)
    if add_global:
        global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, blocks)
        part_featuers_ = part_featuers_ + global_feat
    if otherbranch:
        otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
        return part_featuers_, otherbranch_
    return part_featuers_


class blocks_InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale, weights, blocks=3):
        image_features1_flatten = image_features1.view(image_features1.size(0), image_features1.size(1), -1).transpose(
            -2, -1)
        image_features2_flatten = image_features1.view(image_features2.size(0), image_features2.size(1), -1).transpose(
            -2, -1)

        heat_result_1 = get_heartmap_pool(image_features1_flatten, blocks)
        heat_result_2 = get_heartmap_pool(image_features2_flatten, blocks)

        # 1. concate
        if 1:
            image_features_blocks_1 = torch.cat((heat_result_1[:, :, 0], heat_result_1[:, :, 1], heat_result_1[:, :, 2]),
                                                dim=-1)
            image_features_blocks_2 = torch.cat((heat_result_2[:, :, 0], heat_result_2[:, :, 1], heat_result_2[:, :, 2]),
                                                dim=-1)

            image_features1 = F.normalize(image_features_blocks_1, dim=-1)
            image_features2 = F.normalize(image_features_blocks_2, dim=-1)

            logits_per_image1 = logit_scale * image_features1 @ image_features2.T

            logits_per_image2 = logits_per_image1.T

            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

            loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        # 2. weight and sum (效果更差)
        # image_features_blocks_1 = (3 * weights[0] * heat_result_1[:, :, 0] +
        #                            2 * weights[1] * heat_result_1[:, :, 1] + 1 * weights[2] * heat_result_1[:, :, 2])
        # image_features_blocks_2 = (3 * weights[0] * heat_result_2[:, :, 0] +
        #                            2 * weights[1] * heat_result_2[:, :, 1] + 1 * weights[2] * heat_result_2[:, :, 2])

        # 3. multi-head loss
        else:
            image_features_blocks_1_1, image_features_blocks_1_2, image_features_blocks_1_3 =\
                heat_result_1[:, :, 0], heat_result_1[:, :, 1], heat_result_1[:, :, 2]
            image_features_blocks_2_1, image_features_blocks_2_2, image_features_blocks_2_3 = \
                heat_result_2[:, :, 0], heat_result_2[:, :, 1], heat_result_2[:, :, 2]

            image_features1_1, image_features1_2, image_features1_3 =\
                F.normalize(image_features_blocks_1_1, dim=-1), F.normalize(image_features_blocks_1_2, dim=-1), F.normalize(image_features_blocks_1_3, dim=-1)
            image_features2_1, image_features2_2, image_features2_3 = \
                F.normalize(image_features_blocks_2_1, dim=-1), F.normalize(image_features_blocks_2_2, dim=-1), F.normalize(image_features_blocks_2_3, dim=-1)

            #--
            logits_per_image1_1 = logit_scale * image_features1_1 @ image_features2_1.T
            logits_per_image2_1 = logits_per_image1_1.T
            labels = torch.arange(len(logits_per_image1_1), dtype=torch.long, device=self.device)
            loss1 = (self.loss_function(logits_per_image1_1, labels) + self.loss_function(logits_per_image2_1,
                                                                                          labels)) / 2

            logits_per_image1_2 = logit_scale * image_features1_2 @ image_features2_2.T
            logits_per_image2_2 = logits_per_image1_2.T
            labels = torch.arange(len(logits_per_image1_2), dtype=torch.long, device=self.device)
            loss2 = (self.loss_function(logits_per_image1_2, labels) + self.loss_function(logits_per_image2_2,
                                                                                          labels)) / 2

            logits_per_image1_3 = logit_scale * image_features1_3 @ image_features2_3.T
            logits_per_image2_3 = logits_per_image1_3.T
            labels = torch.arange(len(logits_per_image1_3), dtype=torch.long, device=self.device)
            loss3 = (self.loss_function(logits_per_image1_3, labels) + self.loss_function(logits_per_image2_3,
                                                                                          labels)) / 2

            loss = (loss1 + loss2 + loss3) / 3

        return loss
