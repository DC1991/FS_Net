# @Time    : 12/05/2021
# @Author  : Wei Chen
# @Project : Pycharm



from __future__ import print_function

import os
import argparse
import torch.optim as optim
from torch.autograd import Variable

import torch
from Net_archs import GCN3D_segR, Rot_green, Rot_red, Point_center_res_cate
from data_loader_fsnet import load_pts_train_cate
import torch.nn as nn
import numpy as np
import time
from uti_tool import data_augment

from pyTorchChamferDistance.chamfer_distance import ChamferDistance

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=14, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='models', help='output folder')
parser.add_argument('--outclass', type=int, default=2, help='point class')
parser.add_argument('--model', type=str, default='', help='model path')

opt = parser.parse_args()


kc = opt.outclass
num_cor = 3
num_vec = 8
nw=0 # number of cpu
localtime = (time.localtime(time.time()))
year = localtime.tm_year
month = localtime.tm_mon
day = localtime.tm_mday
hour = localtime.tm_hour

cats = ['bottle','bowl','can','camera','laptop','mug']

for cat in ['laptop']:


    classifier_seg3D = GCN3D_segR(class_num=2, vec_num = 1,support_num= 7, neighbor_num= 10)
    classifier_ce = Point_center_res_cate() ## translation estimation
    classifier_Rot_red = Rot_red(F=1296, k= 6)  ## rotation red
    classifier_Rot_green = Rot_green(F=1296, k=6)### rotation green


    num_classes = opt.outclass

    Loss_seg3D = nn.CrossEntropyLoss()
    Loss_func_ce = nn.MSELoss()
    Loss_func_Rot1 = nn.MSELoss()
    Loss_func_Rot2 = nn.MSELoss()
    Loss_func_s = nn.MSELoss()




    classifier_seg3D = nn.DataParallel(classifier_seg3D)
    classifier_ce = nn.DataParallel(classifier_ce)
    classifier_Rot_red = nn.DataParallel(classifier_Rot_red)
    classifier_Rot_green = nn.DataParallel(classifier_Rot_green)


    classifier_seg3D = classifier_seg3D.train()
    classifier_ce = classifier_ce.train()
    classifier_Rot_red = classifier_Rot_red.train()
    classifier_Rot_green = classifier_Rot_green.train()



    Loss_seg3D.cuda()
    Loss_func_ce.cuda()
    Loss_func_Rot1.cuda()
    Loss_func_Rot2.cuda()
    Loss_func_s.cuda()

    classifier_seg3D.cuda()
    classifier_ce.cuda()
    classifier_Rot_red.cuda()
    classifier_Rot_green.cuda()


    opt.outf = 'models/FS_Net_%s'%(cat)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    sepoch  = 0

    batch_size = 12  #

    lr = 0.001

    epochs = opt.nepoch

    optimizer = optim.Adam([{'params': classifier_seg3D.parameters()},{'params': classifier_ce.parameters()},{'params': classifier_Rot_red.parameters()},{'params': classifier_Rot_green.parameters()}], lr=lr, betas=(0.9, 0.99))

    bbxs = 0
    K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    data_path = 'your data path'
    dataloader = load_pts_train_cate(data_path, batch_size, K,cat, lim=1, rad=300, shuf=True, drop=True, corners=0,nw=nw)

    for epoch in range(sepoch,epochs):

        if epoch > 0 and epoch % (epochs // 5) == 0:
            lr = lr / 4


        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * 10
        optimizer.param_groups[2]['lr'] = lr * 20
        optimizer.param_groups[3]['lr'] = lr * 20

        for i, data in enumerate(dataloader):

            points, target_, Rs, Ts, obj_id,S, imgp= data['points'], data['label'], data['R'], data['T'], data['cate_id'], data['scale'], data['dep']
            ptsori = points.clone()

            target_seg = target_[:, :, 0]  ###seg_target

            points_ = points.numpy().copy()

            points, corners, centers, pts_recon = data_augment(points_[:, :, 0:3], Rs, Ts,num_cor, target_seg,a=15.0)

            points, target_seg, pts_recon = Variable(torch.Tensor(points)), Variable(target_seg), Variable(pts_recon)

            points, target_seg,pts_recon = points.cuda(), target_seg.cuda(), pts_recon.cuda()

            pointsf = points[:, :, 0:3].unsqueeze(2)

            optimizer.zero_grad()
            points = pointsf.transpose(3, 1)
            points_n = pointsf.squeeze(2)

            obj_idh = torch.zeros((1,1))

            if obj_idh.shape[0] == 1:
                obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
            else:
                obj_idh = obj_idh.view(-1, 1)

            one_hot = torch.zeros(points.shape[0], 16).scatter_(1, obj_idh.cpu().long(), 1)
            one_hot = one_hot.cuda() ## the pre-defined category ID



            pred_seg, box_pred_, feavecs = classifier_seg3D(points_n, one_hot)


            pred_choice = pred_seg.data.max(2)[1]  ## B N
            # print(pred_choice[0])
            p = pred_choice  # [0].cpu().numpy() B N
            N_seg = 1000
            pts_s = torch.zeros(points.shape[0], N_seg, 3)

            box_pred = torch.zeros(points.shape[0], N_seg, 3)


            pts_sv = torch.zeros(points.shape[0], N_seg, 3)

            feat = torch.zeros(points.shape[0], N_seg, feavecs.shape[2])


            corners0 = torch.zeros((points.shape[0], num_cor, 3))
            if torch.cuda.is_available():
                ptsori = ptsori.cuda()

            Tt = np.zeros((points.shape[0], 3))
            for ib in range(points.shape[0]):
                if len(p[ib, :].nonzero()) < 10:
                    continue

                pts_ = torch.index_select(ptsori[ib, :, 0:3], 0, p[ib, :].nonzero()[:, 0])  ##Nx3


                box_pred__ = torch.index_select(box_pred_[ib, :, :], 0, p[ib, :].nonzero()[:, 0])
                feavec_ = torch.index_select(feavecs[ib, :, :], 0, p[ib, :].nonzero()[:, 0])

                choice = np.random.choice(len(pts_), N_seg, replace=True)
                pts_s[ib, :, :] = pts_[choice, :]

                box_pred[ib] = box_pred__[choice]
                feat[ib, :, :] = feavec_[choice, :]

                corners0[ib] = torch.Tensor(np.array([[0,0,0],[0,200,0],[200,0,0]]))






            pts_s = pts_s.cuda()



            pts_s = pts_s.transpose(2, 1)
            cen_pred,obj_size = classifier_ce((pts_s - pts_s.mean(dim=2, keepdim=True)), obj_id)


            feavec = feat.transpose(1, 2)

            kp_m = classifier_Rot_green(feavec)


            centers = Variable(torch.Tensor((centers)))


            corners = Variable(torch.Tensor((corners)))




            if torch.cuda.is_available():
                box_pred = box_pred.cuda()
                centers = centers.cuda()
                S = S.cuda()
                corners = corners.cuda()
                feat = feat.cuda()
                corners0 = corners0.cuda()

            loss_seg = Loss_seg3D(pred_seg.reshape(-1, pred_seg.size(-1)), target_seg.view(-1,).long())
            loss_res = Loss_func_ce(cen_pred, centers.float())

            loss_size = Loss_func_s(obj_size,S.float())


            def loss_recon(a, b):
                if torch.cuda.is_available():
                    chamferdist = ChamferDistance()
                    dist1, dist2 = chamferdist(a, b)
                    loss = torch.mean(dist1) + torch.mean(dist2)
                else:
                    loss=torch.Tensor([100.0])
                return loss
            loss_vec = loss_recon(box_pred, pts_recon)



            kp_m2 = classifier_Rot_red(feat.transpose(1,2))  # .detach())

            green_v = corners[:, 0:6].float().clone()
            red_v = corners[:, (0, 1, 2, 6, 7, 8)].float().clone()
            target = torch.tensor([[1]], dtype=torch.float).cuda()


            loss_rot_g= Loss_func_Rot1(kp_m, green_v)
            loss_rot_r = Loss_func_Rot2(kp_m2, red_v)






            symme=1
            if cat in ['bottle','bowl','can']:
                symme=0.0


            Loss = loss_seg*20.0+loss_res/20.0+loss_vec/200.0+loss_size/20.0+symme*loss_rot_r/100.0+loss_rot_g/100.0
            Loss.backward()
            optimizer.step()

            print(cat)
            print('[%d: %d] train loss_seg: %f, loss_res: %f, loss_recon: %f, loss_size: %f, loss_rot_g: %f, '
                  'loss_rot_r: %f' % (
            epoch, i, loss_seg.item(), loss_res.item(), loss_vec.item(), loss_size.item(), loss_rot_g.item(),
            loss_rot_r.item()))


            print()

            torch.save(classifier_seg3D.state_dict(), '%s/Seg3D_last_obj%s.pth' % (opt.outf,
                                                                                    cat))
            torch.save(classifier_ce.state_dict(), '%s/Tres_last_obj%s.pth' % (opt.outf, cat))
            torch.save(classifier_Rot_green.state_dict(),
                       '%s/Rot_g_last_obj%s.pth' % (opt.outf, cat))
            torch.save(classifier_Rot_red.state_dict(),
                       '%s/Rot_r_last_obj%s.pth' % (opt.outf, cat))
            if epoch>0 and epoch %(epochs//5)== 0: ##save mid checkpoints

                torch.save(classifier_seg3D.state_dict(), '%s/Seg3D_epoch%d_obj%s.pth' % (opt.outf,
                                                                                          epoch, cat))
                torch.save(classifier_ce.state_dict(), '%s/Tres_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_Rot_green.state_dict(),
                           '%s/Rot_g_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))
                torch.save(classifier_Rot_red.state_dict(),
                           '%s/Rot_r_epoch%d_obj%s.pth' % (opt.outf, epoch, cat))




