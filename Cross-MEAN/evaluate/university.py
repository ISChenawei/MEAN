import os.path
import scipy.io
import torch
import numpy as np
from tqdm import tqdm
import gc
from ..trainer import predict
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.manifold import TSNE
from matplotlib import cm
def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("Extract Features:")
    img_features_gallery, ids_gallery,paths_gallery = predict(config, model, gallery_loader)
    img_features_query, ids_query,paths_query = predict(config, model, query_loader)


    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    # save_features_to_mat(img_features_query,img_features_gallery,ql,gl,paths_query, paths_gallery)
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i],img_features_gallery,gl ,paths_query[i],paths_gallery )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]


def eval_query(qf, ql, gf, gl,query_path,paths_gallery):
    score = gf @ qf.unsqueeze(-1)
    top_k = 10
    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)
    top_k_results = index[:top_k]
    top_k_path =[paths_gallery[i] for i in top_k_results]
    # print(f"Query Image:{query_path}")
    # save_folder='/home/hk/PAPER/DAC-main/SUES-D-S-150-Rank'
    # #
    # # print(f"Query ID:{ql},Top {top_k} Gallery Results:{gl[top_k_results]}")
    # for rank,gallery_path in enumerate(top_k_path,1):
    #     plot_query_and_gallery(query_path, top_k_path, save_folder)
    #     print(f"Top{rank} Gallery Image:{gallery_path}")
    ## T-tsne
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def save_features_to_mat(query_features, gallery_features, query_labels, gallery_labels, paths_query, paths_gallery, save_path='SUES-D-S-150-features.mat'):
    """
    保存提取的特征和标签到 .mat 文件
    """
    result = {
        'query_features': query_features.cpu().numpy(),
        'gallery_features': gallery_features.cpu().numpy(),
        'query_labels': query_labels,
        'gallery_labels': gallery_labels,
        'paths_query': paths_query,
        'paths_gallery': paths_gallery
    }
    scipy.io.savemat(save_path, result)
    print(f'Features and labels saved to {save_path}')

# def plot_query_and_gallery(query_image_path,gallery_image_path,save_folder):
#     fig,axes =plt.subplots(1,len(gallery_image_path) + 1,figsize = (20,5))
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     query_folder = os.path.basename(os.path.dirname(query_image_path))
#     split_path = query_image_path.split('query_satellite',1)[-1]
#     query_filename = split_path.strip(os.sep)
#
#     output_filename = query_filename.replace(os.sep,"_")
#     output_filename = f"{output_filename}_result.png"
#     query_img = cv.imread(query_image_path)
#     query_img = cv.cvtColor(query_img,cv.COLOR_BGR2RGB)
#     axes[0].imshow(query_img)
#     # axes[0].set.title('Query Image')
#     axes[0].axis('off')
#
#     # rect = patches.Rectangle((0,0),query_img.shape[1],query_img.shape[0],
#     #                          linewidth=5 , edgecolor='orange',facecolor='none')
#     # axes[0].add_patch(rect)
#
#     for i,gallery_image_path in enumerate(gallery_image_path):
#         gallery_img = cv.imread(gallery_image_path)
#         gallery_img = cv.cvtColor(gallery_img,cv.COLOR_BGR2RGB)
#         axes[i + 1].imshow(gallery_img)
#         # axes[i + 1].set_totle(f'Gallery {i +1 }')
#         axes[i + 1].axis('off')
#         gallery_folder = os.path.basename(os.path.dirname(gallery_image_path))
#         is_match = query_folder == gallery_folder
#         rect_color ='green' if is_match else 'red'
#         rect = patches.Rectangle((0, 0), query_img.shape[1], query_img.shape[0],
#                                  linewidth=8, edgecolor=rect_color, facecolor='none')
#         axes[i + 1].add_patch(rect)
#     plt.tight_layout()
#     out_path = os.path.join(save_folder,output_filename)
#     plt.savefig(out_path)
#     plt.close()
#     print(f"Image saved to : {out_path}")
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
