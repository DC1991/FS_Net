# @Time    : 12/05/2021
# @Author  : Wei Chen
# @Project : Pycharm

import cv2
import numpy as np
import os
import _pickle as pickle
from uti_tool import getFiles_cate, depth_2_mesh_all, depth_2_mesh_bbx
from prepare_data.renderer import create_renderer

def render_pre(model_path):
    renderer = create_renderer(640, 480, renderer_type='python')
    models = getFiles_ab_cate(model_path, '.ply') #model name example: laptop_air_1_norm.ply please adjust the
    # corresponding functions according to the model name.
    objs=[]
    for model in models:
        obj = model.split('.')[1]
        objs.append(obj)
        renderer.add_object(obj, model)
    return renderer

def getFiles_ab_cate(file_dir,suf):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.split('.')[1] == suf:
                L.append(os.path.join(root, file))
    return L

def get_dis_all(pc,dep,dd=15):

    N=pc.shape[0]
    M=dep.shape[0]
    depp=np.tile(dep,(1,N))

    depmm=depp.reshape((M,N,3))
    delta = depmm - pc
    diss=np.linalg.norm(delta,2, 2)

    aa=np.min(diss,1)
    bb=aa.reshape((M,1))

    ids,cc=np.where(bb[:]<dd)

    return ids


def get_one(depth, bbx, vispt, K, idx, objid, bp):
    save_path = bp + '/%s/points' % (objid)
    save_pathlab = bp + '/%s/points_labs' % (objid)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_pathlab):
        os.makedirs(save_pathlab)

    VIS = depth_2_mesh_all(vispt, K)
    VIS = VIS[np.where(VIS[:, 2] > 0.0)] * 1000.0

    numbs = 6000

    numbs2 = 1000
    if VIS.shape[0] > numbs2:
        choice2 = np.random.choice(VIS.shape[0], numbs2, replace=False)
        VIS = VIS[choice2, :]


    filename = save_path + ("/pose%08d.txt" % (idx))
    w_namei = save_pathlab + ("/lab%08d.txt" % (idx))

    dep3d_ = depth_2_mesh_bbx(depth, bbx, K, enl=0)

    if dep3d_.shape[0] > numbs:
        choice = np.random.choice(dep3d_.shape[0], numbs, replace=False)

        dep3d = dep3d_[choice, :]
    else:
        choice = np.random.choice(dep3d_.shape[0], numbs, replace=True)
        dep3d = dep3d_[choice, :]

    dep3d = dep3d[np.where(dep3d[:, 2] != 0.0)]


    threshold = 12

    ids = get_dis_all(VIS, dep3d[:, 0:3], dd=threshold) ## find the object points

    if len(ids) <= 10:
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(w_namei):
            os.remove(w_namei)

    if len(ids) > 10:

        np.savetxt(filename, dep3d, fmt='%f', delimiter=' ')
        lab = np.zeros((dep3d.shape[0], 1), dtype=np.uint)
        lab[ids, :] = 1
        np.savetxt(w_namei, lab, fmt='%d')




def get_point_wise_lab(basepath, fold, renderer, sp):
    base_path = basepath + '%d/' % (fold)


    depths = getFiles_cate(base_path, '_depth', 4, -4)

    labels = getFiles_cate(base_path, '_label2', 4, -4)


    L_dep = depths

    Ki = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    Lidx = 1000
    if fold == 1:
        s = 0
    else:
        s = 0
    for i in range(s, len(L_dep)):

        lab = pickle.load(open(labels[i], 'rb'))

        depth = cv2.imread(L_dep[i], -1)
        img_id = int(L_dep[i][-14:-10])
        for ii in range(len(lab['class_ids'])):


            obj = lab['model_list'][ii]

            seg = lab['bboxes'][ii].reshape((1, 4))  ## y1 x1 y2 x2  (ori x1,y1,w h)

            idx = (fold - 1) * Lidx + img_id

            R = lab['rotations'][ii]  # .reshape((3, 3))

            T = lab['translations'][ii].reshape((3, 1))  # -np.array([0,0,100]).reshape((3, 1))


            if T[2] < 0:
                T[2] = -T[2]
            vis_part = renderer.render_object(obj, R, T, Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2])['depth']

            bbx = [seg[0, 0], seg[0, 2], seg[0, 1], seg[0, 3]]

            if vis_part.max() > 0:
                get_one(depth, bbx, vis_part, Ki, idx, obj, sp)




if __name__ == '__main__':
    path = 'your own object model path '
    render_pre(path)






