# @Time    : 10/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov3_fsnet.models.experimental import attempt_load
from yolov3_fsnet.utils.datasets import LoadStreams, LoadImages, LoadImages_fsnet
from yolov3_fsnet.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov3_fsnet.utils.plots import plot_one_box
from yolov3_fsnet.utils.torch_utils import select_device, load_classifier, time_synchronized
from uti_tool import getFiles_ab_cate, depth_2_mesh_bbx, load_ply
from Net_deploy import load_models, FS_Net_Test
from torch.utils.data import DataLoader

def detect(opt,data_path, classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red,
           model_size, cate_id0):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16


    # Set Dataloader
    dataset = LoadImages_fsnet(data_path, img_size=imgsz, stride=stride)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    for icc, data in enumerate(dataloader):
        path, img, im0s, depth_, Rt, Tt, pc =data

        img = img[0].to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred, cenxy = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                     agnostic=opt.agnostic_nms)
        # pred2 = pred[0][(np.where(pred[0][:,-1].cpu()==63))] ##labtop
        K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        DR = int(cenxy.cpu().numpy()[1])
        DC = int(cenxy.cpu().numpy()[0])
        depth = depth_[0].numpy()
        if depth[DR, DC] == 0:
            while depth[DR, DC] == 0:
                DR = min(max(0, DR + np.random.randint(-10, 10)), 480)
                DC = min(max(0, DC + np.random.randint(-10, 10)), 640)
        XC = [0, 0]
        XC[0] = np.float32(DC - K[0, 2]) * np.float32(depth[DR, DC] / K[0, 0])
        XC[1] = np.float32(DR - K[1, 2]) * np.float32(depth[DR, DC] / K[1, 1])
        cen_depth = np.zeros((1, 3))
        cen_depth[0, 0:3] = [XC[0], XC[1], depth[DR, DC]]

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0 = path[0], '', im0s[0].numpy()
            mode = 'image'
            p = Path(p)  # to Path

            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label='', color=colors[int(cls)], line_thickness=3)

                dep3d = depth_2_mesh_bbx(depth, [det[0][1],det[0][3],det[0][0],det[0][2]], K)
                dep3d = dep3d[np.where(dep3d[:, 2]>0.0)]
                # show_mulit_mesh([dep3d])
                dep3d = chooselimt_test(dep3d, 400, cen_depth)  ##3 *N
                choice = np.random.choice(len(dep3d), 1500, replace=True)
                dep3d = dep3d[choice, :]
                #

                FS_Net_Test(dep3d, pc[0].numpy(), im0, Rt, Tt, classifier_seg3D, classifier_ce,
                            classifier_Rot_green,
                            classifier_Rot_red,
                            'laptop', model_size, cate_id0, num_cor=3)



                print(icc)


def chooselimt_test(pts0, dia, cen): ##replace the 3D sphere with 3D cube

    pts = pts0.copy()
    pts = pts[np.where(pts[:, 2] > 20)[0], :]
    ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia)[0], :]
    if ptsn.shape[0] < 1000:
        ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 2)[0], :]
        if ptsn.shape[0] < 500:
            ptsn = pts[np.where(np.abs(pts[:, 2] - cen[:, 2].min()) < dia * 3)[0], :]
    return ptsn



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes',default=63, nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok',default='False', action='store_true', help='existing project/name ok, '
                                                                            'do not increment')
    opt = parser.parse_args()
    print(opt)

    cate = 'laptop'
    fold = 'FS_Net/yolov3_fsnet/data/test_scene_1/' ##should be absolute path

    classifier_seg3D, classifier_ce, classifier_Rot_green, classifier_Rot_red, model_size, cate_id0 = load_models(
        cate)
    with torch.no_grad():


        detect(opt, fold ,classifier_seg3D, classifier_ce, classifier_Rot_green,
               classifier_Rot_red, model_size, cate_id0)
