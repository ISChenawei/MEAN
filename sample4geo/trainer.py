import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sample4geo.loss.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss
import torch.nn as nn
from thop import profile

def train(train_config, model, dataloader, loss_functions, optimizer, epoch, train_steps_per, tensorboard=None,
          scheduler=None, scaler=None):
    # set model train mode
    model.train()
    loss_cls=torch.tensor(0)
    loss_DSA=torch.tensor(0)
    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    criterion = nn.CrossEntropyLoss()

    # for loop over one epoch
    for query, reference, ids, labels in bar:

        if scaler:
            with (autocast()):  # -- 使用混合精度
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                # Forward pass
                if train_config.handcraft_model is not True:
                    features1, features2 = model(query, reference)
                else:
                    output1, output2 = model(query, reference)
                    features_dsa_1, features_dsa_2 = output1[0], output2[0]  # -- for DSA loss
                    print('output4', features_dsa_1.shape, features_dsa_2.shape)
                    features1, features2 = output1[-2], output2[-2]  # -- for contrastive
                    print('output1', features1.shape, features2.shape)
                    features_tri_1, features_tri_2 = output1[2], output2[2]  # -- for triplet
                    print('output2', features_tri_1.shape, features_tri_2.shape)
                    features_cls_1, features_cls_2 = output1[1], output2[1]  # -- for classifier
                    print('output3', features_cls_1.shape, features_cls_2.shape)
                    features_fine_1, features_fine_2 = output1[-1], output2[-1]  # -- for fine-grained


                if torch.cuda.device_count() >= 1 and len(train_config.gpu_ids) >= 1:
                # if torch.cuda.device_count() >= 1:
                    loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
                    # loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
                    # 2. Classification
                    loss_cls = cal_loss(features_cls_1, labels, criterion) + cal_loss(features_cls_2, labels, criterion)


                    # 3. Domian Space Alignment Loss
                    # loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2,
                    #                                       model.logit_scale_blocks.exp())
                    loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2,
                                                          model.module.logit_scale_blocks.exp())
                else:
                    # 1. infoNCE
                    loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())

                    # 2. Classification
                    # loss_cls = cal_loss(features_cls_1, labels, criterion) + cal_loss(features_cls_2, labels, criterion)


                    # 3. Domian Space Alignment Loss
                    loss_DSA = loss_functions["DSA_loss"](features_dsa_1, features_dsa_2,
                                                          model.module.logit_scale_blocks.exp())


                lossall = train_config.weight_infonce * loss + train_config.weight_cls * loss_cls + train_config.weight_dsa * loss_DSA

                # lossall = 1.0 * loss + 0.0 * loss_cls + 0.0 * loss_DSA

                losses.update(lossall.item())

            # scaler.scale(loss).backward()  # -- 混合精度好像是这样用的
            scaler.scale(lossall).backward()  # -- 这里才是反向传播，上面就是记录一下

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        else:

            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                loss = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()

        if train_config.verbose:
            # tst = model.logit_scale
            monitor = {
                "loss": "{:.4f}".format(loss.item()),
                "loss_cls": "{:.4f}".format(train_config.weight_cls * loss_cls.item()),
                "loss_dsa": "{:.4f}".format(train_config.weight_dsa * loss_DSA.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

            if tensorboard is not None:
                steps = step + (epoch - 1) * train_steps_per
                tensorboard.add_scalar("Loss", lossall.item(), steps)
                tensorboard.add_scalar("Loss_Avg", losses.avg, steps)
                tensorboard.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], steps)
                tensorboard.add_scalar("Learning_Rate_Temp", optimizer.param_groups[-1]['lr'], steps)
                tensorboard.add_scalar("Temperature", model.module.logit_scale.detach().cpu().numpy(), steps)
                # tensorboard.add_scalar("Temperature", model.logit_scale.detach().cpu().numpy(), steps)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()
    ## 测试flops
    # If using DataParallel, extract the module for FLOPs and params calculation
    # if isinstance(model, torch.nn.DataParallel):
    #     model_to_profile = model.module  # Unwrap the model from DataParallel
    # else:
    #     model_to_profile = model
    # dummy_input = torch.randn(1,3,384,384).to(train_config.device)
    # dummy_input2 = torch.randn(1,3,384,384).to(train_config.device)
    # flops, params = profile(model_to_profile, inputs=(dummy_input, dummy_input2))
    # print(f"FLOPs:{flops/1e9:.2f}")
    # wait before starting progress bar
    # time.sleep(0.1)
    # num_batches = 0
    # totle_time = 0.0
    # -- draw visualization for ablation study
    draw_vis = False
    if draw_vis:
        import cv2
        import os
        import numpy as np
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 2)
        for i in range(iterations):
            uav_ori = cv2.imread(rf"{pic_path}/{i}_uav.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.jpg")

            uav_shape = uav_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            uav = cv2.resize(uav_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            uav = torch.tensor(uav).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            uav = normalize(uav)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    uav = uav.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_uav = model(uav)[1]
                    img_feature_uav = F.normalize(img_feature_uav, dim=1)

                    img_feature_sat = model(sat)[1]
                    img_feature_sat = F.normalize(img_feature_sat, dim=1)

                    heat_map_uav = img_feature_uav[0].permute(1, 2, 0)
                    heat_map_uav = torch.mean(heat_map_uav, dim=2).detach().cpu().numpy()
                    heat_map_uav = (heat_map_uav - heat_map_uav.min()) / (heat_map_uav.max() - heat_map_uav.min())
                    heat_map_uav = cv2.resize(heat_map_uav, [uav_shape[1], uav_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_uav = cv2.applyColorMap((heat_map_uav * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_uav = cv2.addWeighted(uav_ori, alpha, colored_image_uav, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)


                    out_path = "/SUES-D-S-Rank"
                    cv2.imwrite(rf"{out_path}/{i}_uav_vis.jpg", blended_image_uav)
                    cv2.imwrite(rf"{out_path}/{i}_sat_vis.jpg", blended_image_sat)

        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []
    path_list = []
    ids_list = []
    with torch.no_grad():

        for img, ids,img_paths in bar:
            start_time = time.time()
            ids_list.append(ids)
            path_list.extend(img_paths)
            with autocast():
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    img_feature = model(img)[-2]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
        # end_time = time.time()
        # elapsed_time = end_time -start_time
        # totle_time += elapsed_time
        # num_batches+=1
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
    # avg_fps = num_batches/totle_time if totle_time>0 else 0
    # print(f"Average FPS:{avg_fps:.2f}")
    if train_config.verbose:
        bar.close()

    return img_features, ids_list,path_list
