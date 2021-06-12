# @Time    : 25/09/2020 18:02
# @Author  : Wei Chen
# @Project : Pycharm
import torch
from torch.utils.data import Dataset, DataLoader
import _pickle as pickle
from uti_tool import *
import random


def getFiles(file_dir,suf):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        #print('root: ',dirs)
        for file in files:
            if os.path.splitext(file)[1] == suf:
                L.append(os.path.join(root, file))
        L.sort(key=lambda x:int(x[-11:-4]))
    return L

def getDirs(file_dir):
    L=[]

    dirs = os.listdir(file_dir)

    return dirs


def load_depth(depth_path):
    """ Load depth image from img_path. """

    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def chooselimt(pts0, lab, zmin, zmax):


    pts = pts0.copy()
    labs = lab.copy()

    pts1=pts[np.where(pts[:,2]<zmax)[0],:]
    lab1 = labs[np.where(pts[:,2]<zmax)[0], :]

    ptsn = pts1[np.where(pts1[:, 2] > zmin)[0], :]
    labs = lab1[np.where(pts1[:, 2] > zmin)[0],:]

    return ptsn,labs

def circle_iou(pts,lab, dia):
    # fx = K[0, 0]
    # ux = K[0, 2]
    # fy = K[1, 1]
    # uy = K[1, 2]
    a = pts[lab[:, 0] == 1, :]
    ptss = pts[lab[:, 0] == 1, :]
    idx = np.random.randint(0, a.shape[0])

    zmin = max(0,ptss[idx,2]-dia)
    zmax = ptss[idx,2]+dia

    return zmin, zmax


class CateDataset(Dataset):
    def __init__(self, root_dir, K, cate,lim=1,transform=None,corners=0, temp=None):

        cats = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

        objs = os.listdir(root_dir)
        self.objs_name = objs
        self.objs = np.zeros((len(objs),1),dtype=np.uint)

        for i in range(len(objs)):
            if cate in objs[i]:
                self.objs[i]=1

        self.cate_id = np.where(np.array(cats)==cate)[0][0]+1
        self.ids = np.where(self.objs==1)

        self.root_dir = root_dir
        self.lim=lim
        self.transform=transform
        self.cate = cate
        self.K = K
        self.corners = corners
        self.rad=temp
        if cate=='labtop':
            self.rad = 600
        if cate == 'bottle':
            self.rad = 400



        datapath = 'Real/train/scene_' ## file path of train scenes
        model_path = 'real_train/plys/'  ##object model

        self.data = datapath
        self.c = random.randint(0, len(self.ids) - 1)
        self.model_path = model_path
    def __len__(self):


        return 1500 ##


    def __getitem__(self, index):


        c = random.randint(0, len(self.ids[0])-1)

        obj_id = self.ids[0][c]
        cate = self.objs_name[obj_id]

        pc = load_ply(self.model_path+'/%s.ply'%(cate))['pts']*1000.0


        root_dir = self.root_dir + '/%s/' % (cate)
        pts_ps = getFiles_ab(root_dir+'points/','.txt',-12,-4)
        idx = random.randint(0, len(pts_ps) - 1)
        pts_name = pts_ps[idx]
        lab_name = getFiles_ab(root_dir+'points_labs/','.txt',-12,-4)[idx]



        scene_id = int(pts_name[-12:-4])//1000+1 ## you can change according to your own name rules

        img_id = int(pts_name[-12:-4])-(scene_id-1)*1000

        depth_p  = self.data+'%d'%(scene_id)+'/%04d_depth.png'%(img_id)
        label_p = self.data+'%d'%(scene_id)+'/%04d_label.pkl'%(img_id)

        gts = pickle.load(open(label_p, 'rb'))
        idin = np.where(np.array(gts['model_list']) == cate)


        if len(idin[0])==0: ## fix some wrong cases
            bbx = np.array([1,2,3,4]).reshape((1, 4))
            R = np.eye(3)
            T = np.array([0,0,0]).reshape(1,3)
        else:
            bbx = gts['bboxes'][idin[0]].reshape((1, 4)) ## y1 x1 y2 x2
            R = gts['rotations'][idin[0]].reshape(3,3)
            T = gts['translations'][idin[0]].reshape(1,3)*1000.0

        self.pc = pc
        self.R = R
        self.T = T
        depth = cv2.imread(depth_p,-1)
        # pts_name = bpp + 'pose%08d.txt' % (idx)

        label = np.loadtxt(lab_name)


        label_ = label.reshape((-1, 1))
        points_ = np.loadtxt(pts_name)



        points_, label_,sx,sy,sz = self.aug_pts_labs(depth,points_,label_,bbx)

        Scale = np.array([sx,sy,sz])


        if  points_.shape[0]!=label_.shape[0]:
            print(self.root_dir[idx])

        choice = np.random.choice(len(points_), 2000, replace=True)
        points = points_[choice, :]
        label = label_[choice, :]

        sample = {'points': points, 'label': label, 'R':R, 'T':T,'cate_id':self.cate_id,'scale':Scale,'dep':depth_p}

        return sample

    def aug_pts_labs(self, depth,pts,labs,bbx):

        ## 2D bounding box augmentation and fast relabeling
        bbx_gt = [bbx[0,1], bbx[0,3],bbx[0,0],bbx[0,2]]#x1,x2, y1 , y2
        bbx = shake_bbx(bbx_gt) ## x1,x2,y1,y2
        depth, bbx_iou = depth_out_iou(depth, bbx, bbx_gt)

        mesh = depth_2_mesh_bbx(depth, [bbx[2], bbx[3], bbx[0], bbx[1]], self.K)
        mesh = mesh[np.where(mesh[:, 2] > 0.0)]
        mesh = mesh[np.where(mesh[:, 2] < 5000.0)]

        if len(mesh) > 1000:
            choice = np.random.choice(len(mesh), len(mesh)//2, replace=True)
            mesh = mesh[choice, :]

        pts_a, labs_a = pts_iou(pts.copy(), labs.copy(), self.K, bbx_iou)

        assert pts_a.shape[0]==labs_a.shape[0]

        if len(pts_a[labs_a[:, 0] == 1, :])<50: ## too few points in intersection region
            pts_=pts_a.copy()
            labs_ = labs_a.copy()
        else:
            pts_ = pts.copy()
            labs_ = labs.copy()

        N = pts_.shape[0]
        M = mesh.shape[0]
        mesh = np.concatenate([mesh, pts_], axis=0)
        label = np.zeros((M + N, 1), dtype=np.uint)
        label[M:M + N, 0] = labs_[:, 0]
        points = mesh

        if self.lim == 1:
            zmin, zmax = circle_iou(points.copy(), label.copy(), self.rad)
            points, label = chooselimt(points, label,zmin, zmax)



        ### 3D deformation
        Rt = get_rotation(180,0,0)
        self.pc = np.dot(Rt, self.pc.T).T ## the object 3D model is up-side-down along the X axis in our case, you may not need this code to reverse


        s  = 0.8
        e = 1.2
        pointsn, ex,ey, ez,s = defor_3D(points,label, self.R, self.T, self.pc, scalex=(s, e),scalez=(s, e),
                                        scaley=(s, e), scale=(s, e), cate=self.cate)
        sx,sy,sz = var_2_norm(self.pc, ex, ey, ez, c=self.cate)
        return pointsn, label.astype(np.uint8), sx,sy,sz


def load_pts_train_cate(data_path ,bat,K,cate,lim=1,rad=400,shuf=True,drop=False,corners=0,nw=0):

    data=CateDataset(data_path, K, cate,lim=lim,transform=None,corners=corners, temp=rad)

    dataloader = DataLoader(data, batch_size=bat, shuffle=shuf, drop_last=drop,num_workers=nw)

    return dataloader










