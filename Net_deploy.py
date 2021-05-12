# @Time    : 11/05/2021
# @Author  : Wei Chen
# @Project : Pycharm

from __future__ import print_function


import os
from uti_tool import compute_3d_IoU
import argparse
import numpy as np
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
import torch
import torch.nn as nn
import cv2

from uti_tool import load_ply, draw_cors_withsize, draw_cors, get_3D_corner, trans_3d, gettrans,get6dpose1

def load_models(cat):
    classifier_seg3D = GCN3D_segR(class_num=2, vec_num = 1,support_num= 7, neighbor_num= 10)
    classifier_ce = Point_center_res_cate() ## translation estimation
    classifier_Rot_red = Rot_red(F=1296, k= 6)  ## rotation red
    classifier_Rot_green = Rot_green(F=1296, k=6)### rotation green


    # optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    classifier_seg3D = nn.DataParallel(classifier_seg3D)
    classifier_ce = nn.DataParallel(classifier_ce)
    classifier_Rot_red = nn.DataParallel(classifier_Rot_red)
    classifier_Rot_green = nn.DataParallel(classifier_Rot_green)


    classifier_seg3D = classifier_seg3D.eval()
    classifier_ce = classifier_ce.eval()
    classifier_Rot_red = classifier_Rot_red.eval()
    classifier_Rot_green = classifier_Rot_green.eval()
    #

    classifier_seg3D.cuda()
    classifier_ce.cuda()
    classifier_Rot_green.cuda()
    classifier_Rot_red.cuda()

    outf = 'trained_models/'

    Seg3d = '%s/Seg3D_last_obj%s.pth' % (outf, cat)
    Tes = '%s/Tres_last_obj%s.pth' % (outf, cat)
    Rot = '%s/Rot_g_last_obj%s.pth' % (outf, cat)
    Rot_res = '%s/Rot_r_last_obj%s.pth' % (outf, cat)

    classifier_seg3D.load_state_dict(torch.load(Seg3d))
    classifier_ce.load_state_dict(torch.load(Tes))
    classifier_Rot_green.load_state_dict(torch.load(Rot))
    classifier_Rot_red.load_state_dict(torch.load(Rot_res))
    model_sizes = np.array(
        [[87, 220, 89], [165, 80, 165], [88, 128, 156], [68, 146, 72], [346, 200, 335], [146, 83, 114]])  ## 6x3

    cats = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    cate_id0 = np.where(np.array(cats) == cat)[0][0]
    model_size = model_sizes[cate_id0]

    return classifier_seg3D, classifier_ce, classifier_Rot_green,classifier_Rot_red, model_size,cate_id0
def FS_Net_Test(points, pc, rgb, Rt, Tt, classifier_seg3D, classifier_ce, classifier_Rot_green,classifier_Rot_red,
                cat, model_size,cate_id0,num_cor=3):

    OR, x_r, y_r, z_r = get_3D_corner(pc)
    points = torch.from_numpy(points).unsqueeze(0)

    Rt0 = Rt[0].numpy()
    Tt = Tt[0].numpy().reshape(3,1)

    ptsori = points.clone()
    points= points.numpy().copy()

    res = np.mean(points[0],0)
    points[0, :, 0:3] = points[0, :, 0:3] - np.array([res[0], res[1], res[2]])


    points = torch.from_numpy(points).cuda()

    pointsf = points[:, :, 0:3].unsqueeze(2) ##128 1500 1 12

    points = pointsf.transpose(3, 1)
    points_n = pointsf.squeeze(2)

    obj_idh = torch.zeros((1, 1))

    if obj_idh.shape[0] == 1:
        obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
    else:
        obj_idh = obj_idh.view(-1, 1)

    one_hot = torch.zeros(points.shape[0], 16).scatter_(1, obj_idh.cpu().long(), 1)

    one_hot = one_hot.cuda()

    pred_seg, point_recon, feavecs = classifier_seg3D(points_n, one_hot)

    pred_choice = pred_seg.data.max(2)[1]

    p = pred_choice

    ptsori=ptsori.cuda()
    pts_ = torch.index_select(ptsori[0, :, 0:3], 0, p[0,:].nonzero()[:,0])  ##Nx3

    feat = torch.index_select(feavecs[0, :, :], 0, p[0, :].nonzero()[:, 0])

    if len(pts_)<10:
        print('No object pts')
    else:
        pts_s = pts_[:, :].unsqueeze(0).float()
            # print(ib)

        # p[0, 10:31]
            # feas = torch.index_select(feass[ib, :, :], 0, indexs[ib, :].nonzero()[:, 0])

        if num_cor == 3:
            corners0 = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]]))
        else:
            corners0 = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0]]))

        pts_s=pts_s.cuda()
        feat = feat.cuda()
        corners0 = corners0.cuda()


        pts_s=pts_s.transpose(2,1)

        cen_pred,obj_size = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)),torch.Tensor([cate_id0]))
        T_pred = pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2) ## 1x3x1


        # feavec = torch.cat([box_pred, feat.unsqueeze(0)], 2)  ##
        feavec = feat.unsqueeze(0).transpose(1, 2)
        kp_m = classifier_Rot_green(feavec)

        if num_cor == 3:
            corners_ = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        else:
            corners_ = np.array([[0, 0, 0], [0, 1, 0]])



        kpm_gt = (trans_3d(corners_, Rt0, np.array([0, 0, 0]).T).T).flatten()



        bbx_3D = model_size+obj_size.detach().cpu().numpy()
        model_3D = np.array([x_r, y_r, z_r])



        box_pred_gan = classifier_Rot_red(feat.unsqueeze(0).transpose(1, 2))

        pred_axis = np.zeros((num_cor,3))

        pred_axis[0:2,:] = kp_m.view((2, 3)).detach().cpu().numpy()
        if num_cor==3:
            pred_axis[2,:] = box_pred_gan.view((2, 3)).detach().cpu().numpy()[1,:]

        box_pred_gan=box_pred_gan.detach().cpu().numpy()
        box_pred_gan = box_pred_gan / np.linalg.norm(box_pred_gan)
        cor0 = corners0.cpu().numpy()
        cor0= cor0/np.linalg.norm(cor0)
        kpm_gt = kpm_gt.reshape((num_cor,3))
        kpm_gt = kpm_gt/np.linalg.norm(kpm_gt)


        pred_axis = pred_axis/np.linalg.norm(pred_axis)

        pose_gt = gettrans(cor0.reshape((num_cor, 3)), kpm_gt.reshape((num_cor, 1, 3)))
        Rt = pose_gt[0][0:3, 0:3]

        pose = gettrans(cor0.reshape((num_cor, 3)), pred_axis.reshape((num_cor, 1, 3)))
        R = pose[0][0:3, 0:3]


        T = (pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)).view(1,3).detach().cpu().numpy()
        # T = res[0:3]+( cen_pred.unsqueeze(2)).view(1, 3).detach().cpu().numpy()
        #noise_batch_drop_numofloss_loss__cls_model_epoch.pth
        torch.cuda.empty_cache()

        show = 1
        if show == 1:
            R_loss, T_loss = get6dpose1(Rt, Tt, R, T, cat)
            size_2 = bbx_3D.reshape(3)
            K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

            rgb0 = rgb
            rgb0 = draw_cors(rgb0, pc, K, Rt, Tt, [255, 255, 255])
            rgb0 = draw_cors_withsize(rgb0, K, R, T, [255, 0, 0], xr=size_2[0], yr=size_2[1], zr=size_2[2])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb0, 'R_loss: %s' % (R_loss), (10, 20), font, 0.5, (0, 0, 0), 1, 0)
            cv2.putText(rgb0, 'T_loss(mm): %s' % (T_loss), (10, 40), font, 0.5, (0, 0, 0), 1, 0)
            cv2.imshow('show', rgb0 / 255)
            cv2.waitKey(10)
        eva = 1
        # if eva==1:
        #
        #     sRT_1 = np.eye(4)
        #     sRT_1[0:3, 0:3] = Rt
        #     sRT_1[0:3, 3:4] = Tt
        #     sRT_2 = np.eye(4)
        #     sRT_2[0:3, 0:3] = R
        #     sRT_2[0:3, 3:4] = T.reshape(3,1)
        #     size_2= bbx_3D.reshape(3)
        #     size_1 = model_3D
        #
        #     # size_2 = size_1
        #     class_name_1 = cat
        #     class_name_2 = cat
        #     iou3d = compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2,
        #                            handle_visibility=1)
        #
        #     return iou3d, R_loss, T_loss

