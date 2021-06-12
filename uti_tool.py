# @Time    : 06/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import numpy as np
import torch
import cv2
import math
import struct
import os
import random



def get_rotation(x_,y_,z_):

    # print(math.cos(math.pi/2))
    x=float(x_/180)*math.pi
    y=float(y_/180)*math.pi
    z=float(z_/180)*math.pi
    R_x=np.array([[1, 0, 0 ],
                 [0, math.cos(x), -math.sin(x)],
                 [0, math.sin(x), math.cos(x)]])

    R_y=np.array([[math.cos(y), 0, math.sin(y)],
                 [0, 1, 0],
                 [-math.sin(y), 0, math.cos(y)]])

    R_z=np.array([[math.cos(z), -math.sin(z), 0 ],
                 [math.sin(z), math.cos(z), 0],
                 [0, 0, 1]])
    return np.dot(R_z,np.dot(R_y,R_x))


def trans_3d(pc,Rt,Tt):
    Tt=np.reshape(Tt,(3,1))
    pcc=np.zeros((4,pc.shape[0]),dtype=np.float32)
    pcc[0:3,:]=pc.T
    pcc[3,:]=1

    TT=np.zeros((3,4),dtype=np.float32)
    TT[:,0:3]=Rt
    TT[:,3]=Tt[:,0]
    trans=np.dot(TT,pcc)
    return trans


def data_augment(points, Rs, Ts, num_c, target_seg,ax=5, ay=5, az=25, a=15):

    centers = np.zeros((points.shape[0], 3))
    corners = np.zeros((points.shape[0], 3 * num_c))
    pts_recon= torch.zeros(points.shape[0], 1000, 3)
    pts_noTs = points.copy()
    pts0 = points.copy()
    for ii in range(points.shape[0]):




        # idx = idxs[ii].item()
        Rt = Rs[ii].numpy().reshape(3,3)
        Tt = Ts[ii].numpy().reshape(1,3)

        # res = np.mean(points[ii], 0)
        res = np.mean(points[ii], 0)
        points[ii, :, 0:3] = points[ii, :, 0:3] - np.array([res[0], res[1], res[2]])


        points[ii, :, 0:3] = cv2.ppf_match_3d.addNoisePC(np.float32(points[ii, :, 0:3]), 0.1)

        dx = np.random.randint(-ax, ax)
        dy = np.random.randint(-ay, ay)
        dz = np.random.randint(-az, az)

        points[ii, :, 0] = points[ii, :, 0] + dx
        points[ii, :, 1] = points[ii, :, 1] + dy
        points[ii, :, 2] = points[ii, :, 2] + dz






        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        points[ii, :, 0:3] = np.dot(Rm, points[ii, :, 0:3].T).T

        if target_seg[0].sum().item()<1:
            print('target_seg.numpy()[ii, :] == 1)[0].sum()',(target_seg.numpy()[ii, :] ==1)[0].sum())

        pts_seg = pts0[ii, np.where(target_seg.numpy()[ii, :] == 1)[0], 0:3]
        centers[ii,:]=Tt-np.mean(pts_seg,0)

        Tt_c = np.array([0, 0, 0]).T


        corners_ = np.array([[0,0,0],[0,200, 0],[200, 0, 0]])

        pts_rec = pts_seg - Tt

        choice = np.random.choice(len(pts_rec), 1000, replace=True)
        pts_rec = pts_rec[choice, :]
        pts_recon[ii, :] = torch.Tensor(np.dot(Rm, pts_rec.T).T)

        pts_nT = pts_noTs[ii, :] - Tt
        pts_noTs[ii, :] = np.dot(Rm, pts_nT.T).T
        # corners_=kps2
        corners[ii, :] = (trans_3d(corners_, np.dot(Rm, Rt), Tt_c).T).flatten()



    return points,corners, centers, pts_recon

def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: A path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    text_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split(' ')[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split(' ')[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split(' ')[-1], line.split(' ')[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            face_props.append(('n_corners', elems[2]))
            for i in range(face_n_corners):
                face_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('property2 list') and header_face_section:
            elems = line.split(' ')
            # (name of the property, data type)
            text_props.append(('n_corners', elems[2]))
            for i in range(3):
                text_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print ('Error: Only triangular faces are supported.')
                        print ('Number of face corners:', val)
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:

            elems = f.readline().rstrip('\n').rstrip('\r').split(' ')

            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print ('Error: Only triangular faces are supported.')
                        print ('Number of face corners:', int(elems[prop_id]))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]
        #print(prop_vals.keys())
        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    f.close()

    return model

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]
    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]
    Returns:
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates
def compute_3d_IoU(sRT_1, sRT_2, size_1, size_2, class_name_1, class_name_2, handle_visibility):
    '''

    Args:
        sRT_1: 4x4
        sRT_2: 4x4
        size_1: 3x8
        size_2: 3
        class_name_1: str
        class_name_2: str
        handle_visibility: bool

    Returns:

    '''
    """ Computes IoU overlaps between two 3D bboxes. """
    def asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2):
        noc_cube_1 = get_3d_bbox(size_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, sRT_1)
        noc_cube_2 = get_3d_bbox(size_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, sRT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if sRT_1 is None or sRT_2 is None:
        return -1

    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):
        def y_rotation_matrix(theta):
            return np.array([[ np.cos(theta), 0, np.sin(theta), 0],
                             [ 0,             1, 0,             0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [ 0,             0, 0,             1]])
        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = sRT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, sRT_2, size_1, size_2))
    else:
        max_iou = asymmetric_3d_iou(sRT_1, sRT_2, size_1, size_2)

    return max_iou

def get_change_3D(x_r,y_r,z_r):
    ext1=np.array([0,x_r,y_r,z_r])
    or1=np.array([-ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or2=np.array([ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or3=np.array([ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or4=np.array([-ext1[1]/2,ext1[2]/2,ext1[3]/2])

    or5=np.array([-ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or6=np.array([ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or7=np.array([ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or8=np.array([-ext1[1]/2,ext1[2]/2,-ext1[3]/2])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])
    return OR

def get_3D_corner(pc):
    # pc=move_2_C(pc)
    x_r=max(pc[:,0])-min(pc[:,0])
    y_r=max(pc[:,1])-min(pc[:,1])
    z_r=max(pc[:,2])-min(pc[:,2])

    # print(max(pc[:,0]))
    # pdb.set_trace()

    ext1=np.array([0,x_r,y_r,z_r])
    or1=np.array([-ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or2=np.array([ext1[1]/2,-ext1[2]/2,ext1[3]/2])
    or3=np.array([ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or4=np.array([-ext1[1]/2,ext1[2]/2,ext1[3]/2])

    or5=np.array([-ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or6=np.array([ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or7=np.array([ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or8=np.array([-ext1[1]/2,ext1[2]/2,-ext1[3]/2])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])

    return OR, x_r,y_r,z_r


def draw_cors_withsize(img_,K,R_,T_,color,xr,yr,zr, lindwidth=2):
    T_=T_.reshape((3,1))
    img=np.zeros(img_.shape)
    np.copyto(img,img_)

    R=R_


    OR=get_change_3D(xr,yr,zr)
    OR_temp=OR



    OR[:,0]=OR_temp[:,0]
    OR[:,1]=OR_temp[:,1]
    OR[:,2]=OR_temp[:,2]


    pcc=np.zeros((4,len(OR)),dtype='float32')
    pcc[0:3,:]=OR.T
    pcc[3,:]=1


    TT=np.zeros((3,4),dtype='float32')
    TT[:,0:3]=R
    TT[:,3]=T_[:,0]

    camMat=K

    pc_t = np.dot(TT, pcc)  # 3xN
    pc_tt = np.dot(camMat, pc_t)

    pc_t=np.transpose(pc_tt)
    x=pc_t[:,0]/pc_t[:,2]
    y=pc_t[:,1]/pc_t[:,2]





    cv2.line(img, (np.int(x[0]),np.int(y[0])), (np.int(x[1]), np.int(y[1])), color, lindwidth)
    cv2.line(img, (np.int(x[1]),np.int(y[1])), (np.int(x[2]), np.int(y[2])), color, lindwidth)
    cv2.line(img, (np.int(x[2]),np.int(y[2])), (np.int(x[3]), np.int(y[3])), color, lindwidth)
    cv2.line(img, (np.int(x[3]),np.int(y[3])), (np.int(x[0]), np.int(y[0])), color, lindwidth)

    cv2.line(img, (np.int(x[0]),np.int(y[0])), (np.int(x[4]), np.int(y[4])), color, lindwidth)
    cv2.line(img, (np.int(x[1]),np.int(y[1])), (np.int(x[5]), np.int(y[5])), color, lindwidth)
    cv2.line(img, (np.int(x[2]),np.int(y[2])), (np.int(x[6]), np.int(y[6])), color, lindwidth)
    cv2.line(img, (np.int(x[3]),np.int(y[3])), (np.int(x[7]), np.int(y[7])), color, lindwidth)

    cv2.line(img, (np.int(x[4]),np.int(y[4])), (np.int(x[5]), np.int(y[5])), color, lindwidth)
    cv2.line(img, (np.int(x[5]),np.int(y[5])), (np.int(x[6]), np.int(y[6])), color, lindwidth)
    cv2.line(img, (np.int(x[6]),np.int(y[6])), (np.int(x[7]), np.int(y[7])), color, lindwidth)
    cv2.line(img, (np.int(x[7]),np.int(y[7])), (np.int(x[4]), np.int(y[4])), color, lindwidth)


    return img

def move_2_C(pc):
    x_c=(max(pc[:,0])+min(pc[:,0]))/2
    y_c=(max(pc[:,1])+min(pc[:,1]))/2
    z_c=(max(pc[:,2])+min(pc[:,2]))/2
    pc_t=pc
    pc[:,0]=pc_t[:,0]-x_c
    pc[:,1]=pc_t[:,1]-y_c
    pc[:,2]=pc_t[:,2]-z_c
    return pc


def draw_cors(img_,pc,K,R_,T_,color, lindwidth=2):
    pc = move_2_C(pc)
    T_=T_.reshape((3,1))
    img=np.zeros(img_.shape)
    np.copyto(img,img_)

    R=R_

    # R_m=get_rotation(0,0,-90)
    R_m=get_rotation(0,0,0)
    # print(R_m)
    R=np.dot(R,R_m)
    # pc_temp=pc
    # pc[:,0]=pc_temp[:,0]
    # pc[:,1]=pc_temp[:,1]
    #print(pc.shape)
    OR1,xr,yr,zr=get_3D_corner(pc)
    #print(xr,yr,zr)
    #dfd
    OR=get_change_3D(xr,yr,zr)
    OR_temp=OR



    OR[:,0]=OR_temp[:,0]
    OR[:,1]=OR_temp[:,1]
    OR[:,2]=OR_temp[:,2]


    # OR[:,0]=OR_temp[:,0]
    # OR[:,1]=OR_temp[:,1]
    # OR[:,2]=OR_temp[:,2]

    pcc=np.zeros((4,len(OR)),dtype='float32')
    pcc[0:3,:]=OR.T
    pcc[3,:]=1


    TT=np.zeros((3,4),dtype='float32')
    TT[:,0:3]=R
    TT[:,3]=T_[:,0]
    #print('s: ',TT)
    # etrs

    #aa=TT*pcc
    #print(aa.shape)
    camMat=K
    #pdb.set_trace()
    pc_tt=np.dot(camMat,np.dot(TT,pcc))

    pc_t=np.transpose(pc_tt)
    x=pc_t[:,0]/pc_t[:,2]
    y=pc_t[:,1]/pc_t[:,2]





    cv2.line(img, (np.int(x[0]),np.int(y[0])), (np.int(x[1]), np.int(y[1])), color, lindwidth)
    cv2.line(img, (np.int(x[1]),np.int(y[1])), (np.int(x[2]), np.int(y[2])), color, lindwidth)
    cv2.line(img, (np.int(x[2]),np.int(y[2])), (np.int(x[3]), np.int(y[3])), color, lindwidth)
    cv2.line(img, (np.int(x[3]),np.int(y[3])), (np.int(x[0]), np.int(y[0])), color, lindwidth)

    cv2.line(img, (np.int(x[0]),np.int(y[0])), (np.int(x[4]), np.int(y[4])), color, lindwidth)
    cv2.line(img, (np.int(x[1]),np.int(y[1])), (np.int(x[5]), np.int(y[5])), color, lindwidth)
    cv2.line(img, (np.int(x[2]),np.int(y[2])), (np.int(x[6]), np.int(y[6])), color, lindwidth)
    cv2.line(img, (np.int(x[3]),np.int(y[3])), (np.int(x[7]), np.int(y[7])), color, lindwidth)

    cv2.line(img, (np.int(x[4]),np.int(y[4])), (np.int(x[5]), np.int(y[5])), color, lindwidth)
    cv2.line(img, (np.int(x[5]),np.int(y[5])), (np.int(x[6]), np.int(y[6])), color, lindwidth)
    cv2.line(img, (np.int(x[6]),np.int(y[6])), (np.int(x[7]), np.int(y[7])), color, lindwidth)
    cv2.line(img, (np.int(x[7]),np.int(y[7])), (np.int(x[4]), np.int(y[4])), color, lindwidth)
    # plt.imshow(img)
    # plt.plot([x[0],x[1]],[y[0],y[1]],marker = 'o',color='red')
    # plt.plot([x[1],x[2]],[y[1],y[2]],marker = 'o',color='red')
    # plt.plot([x[2],x[3]],[y[2],y[3]],marker = 'o',color='red')
    # plt.plot([x[3],x[0]],[y[3],y[0]],marker = 'o',color='red')
    #
    # plt.plot([x[0],x[4]],[y[0],y[4]],marker = 'o',color='red')
    # plt.plot([x[1],x[5]],[y[1],y[5]],marker = 'o',color='red')
    # plt.plot([x[2],x[6]],[y[2],y[6]],marker = 'o',color='red')
    # plt.plot([x[3],x[7]],[y[3],y[7]],marker = 'o',color='red')
    #
    # plt.plot([x[4],x[5]],[y[4],y[5]],marker = 'o',color='red')
    # plt.plot([x[5],x[6]],[y[5],y[6]],marker = 'o',color='red')
    # plt.plot([x[6],x[7]],[y[6],y[7]],marker = 'o',color='red')
    # plt.plot([x[7],x[4]],[y[7],y[4]],marker = 'o',color='red')

    return img

def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    #print(P.shape,Q.shape)
    # print(np.mean(P,0))
    # P= P-np.mean(P,0)
    # Q =Q - np.mean(Q, 0)
    # print(P)
    # tests
    C = np.dot(P.T, Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    U, S, V = np.linalg.svd(C)
    #S=np.diag(S)
    #print(C)
    # print(S)
    #print(np.dot(U,np.dot(S,V)))
    d = (np.linalg.det(V.T) * np.linalg.det(U.T)) <0.0

    # d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    # E = np.diag(np.array([1, 1, 1]))
    # if d:
    #     S[-1] = -S[-1]
    #     V[:, -1] = -V[:, -1]
    E = np.diag(np.array([1, 1, (np.linalg.det(V.T) * np.linalg.det(U.T))]))


    # print(E)

    # Create Rotation matrix U
    #print(V)
    #print(U)
    R = np.dot(V.T ,np.dot(E,U.T))

    return R



def gettrans(kps,h):
    # print(kps.shape) ##N*3
    # print(h.shape)##N,100,3
    # tess
    hss=[]
    # print(h)
    # print(kps.shape) ##3*N
    # kps
    # print(kps.shape)
    # tess
    kps=kps.reshape(-1,3)
    for i in range(h.shape[1]):
        # print(i)
        # print(h[:,i,:].shape #N*3
        # tss

        P = kps.T - kps.T.mean(1).reshape((3, 1))
        #
        Q= h[:,i,:].T - h[:,i,:].T.mean(1).reshape((3,1))
        # print(P.shape,Q.shape)
        # print(kps,h[:,i,:])
        # tess

        # print(P.T,Q.T)
        R=kabsch(P.T,Q.T) ##N*3, N*3

        T=h[:,i,:]-np.dot(R,kps.T).T

        # print(np.mean(T,0))
        # tess
        # print(T.shape)
        hh = np.zeros((3, 4), dtype=np.float32)
        hh[0:3,0:3]=R
        hh[0:3,3]=np.mean(T,0)
        # print(R)
        hss.append(hh)
        # print(hh)
        # if i==3:
        #     tess
    # print(hss)
    return hss

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id,hv=0):

    R1 = RT_1


    R2 = RT_2

    #     try:
    #         assert np.abs(np.linalg.det(R1) - 1) < 0.01
    #         assert np.abs(np.linalg.det(R2) - 1) < 0.01
    #     except AssertionError:
    #         print(np.linalg.det(R1), np.linalg.det(R2))

    if class_id in ['bottle', 'can', 'bowl']:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_id == 'mug' and hv==0:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_id in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    # shift = np.linalg.norm(T1 - T2) * 100
    result = theta

    return result
def calcAngularDistance(Rt, R):

    #print(np.transpose(Rt))

    #t1=np.ones((3,3))
    rotDiff = np.dot(Rt.T, R/np.linalg.det(R))
    #print(rotDiff)


    trace = np.trace(rotDiff)
    #print(np.arccos(1)/math.pi)


    trace2 = np.min([float(3.0000), np.max([float(-1.0000), float(trace)])])


    #print((float(trace2) - 1.0) / 2.0)
    #pdb.set_trace()

    return float(180 * np.arccos((float(trace2) - 1.0) / 2.0) / math.pi)


def get6dpose1(Rt,Tt, R, T, sy=0,class_id='',hv=0):

    if sy==1:
        R_loss = compute_RT_degree_cm_symmetry(Rt, R,class_id,hv)
    else:
        R_loss=calcAngularDistance(Rt, R)


    Tt = Tt.reshape(3,1)
    T = T.reshape(3,1)

    t_loss=cv2.norm(Tt-T,normType=cv2.NORM_L2)
    #print(cv2.norm(Tt-T))
    #print(np.linalg.norm(Tt-T))
    #pdb.set_trace()

    return R_loss, t_loss

def getFiles_cate(file_dir,suf,a,b, sort=1):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        #print('root: ',dirs)
        for file in files:
            if os.path.splitext(file)[0][4:] == suf:

                L.append(os.path.join(root, file))
        if sort==1:
            L.sort(key=lambda x: int(x[b-len(suf)-a:b-len(suf)]))#0000.png
    return L

def getFiles_ab_cate(file_dir,suf,a,b, sort=1):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        #print('root: ',dirs)
        for file in files:
            if file.split('_')[1]== suf:
                #print(os.path.join(root, file))
                #sdf
                L.append(os.path.join(root, file))
        # L.sort(key=lambda x:int(x[-9:-4])) # 0000
        if sort==1:
            L.sort(key=lambda x: int(x[a:b]))#0000.png
    return L



def depth_2_pc(depth, K, bbx=[1, 2, 3, 4], step=1):
    x1 = bbx[0]
    x2 = bbx[1]
    y1 = bbx[2]
    y2 = bbx[3]

    fx = K[0, 0]
    ux = K[0, 2]
    fy = K[1, 1]
    uy = K[1, 2]
    W = y2 - y1 + 1
    H = x2 - x1 + 1

    xw0 = np.arange(y1, y2, step)
    xw0 = np.expand_dims(xw0, axis=0)
    xw0 = np.tile(xw0.T, 2)
    uu0 = np.zeros_like(xw0, dtype=np.float32)
    uu0[:, 0] = ux
    uu0[:, 1] = uy

    mesh = np.zeros((len(range(0, H, step)) * xw0.shape[0], 3))
    c = 0
    for i in range(x1, x2, step):
        xw = xw0.copy()
        uu = uu0.copy()
        xw[:, 0] = i  ### W 2

        z = depth[xw[:, 0], xw[:, 1]]  ##W 1

        xw[:, 0] = xw[:, 0] * z
        xw[:, 1] = xw[:, 1] * z

        uu[:, 0] = uu[:, 0] * z
        uu[:, 1] = uu[:, 1] * z

        X = (xw[:, 1] - uu[:, 0]) / fx
        Y = (xw[:, 0] - uu[:, 1]) / fy
        mesh[xw.shape[0] * c:xw.shape[0] * (c + 1), 0] = X
        mesh[xw.shape[0] * c:xw.shape[0] * (c + 1), 1] = Y
        mesh[xw.shape[0] * c:xw.shape[0] * (c + 1), 2] = z
        c += 1

    return mesh
def depth_2_mesh_bbx(depth,bbx,K, step=1, enl=0):
    #bbx: r1,r2, c1,c2

    x1 = int(max(bbx[0],0))-enl
    x2 = int(min(bbx[1],depth.shape[0]))+enl
    y1 = int(max(bbx[2],0))-enl
    y2 = int(min(bbx[3],depth.shape[1]))+enl


    mesh = depth_2_pc(depth, K, bbx = [x1,x2,y1,y2], step=step)

    return mesh

def depth_2_mesh_all(depth,K):

    #dep=np.logical_and(depth[:,:],depth[:,:])


    r,c=np.where(depth>0)

    r1 = r.min()
    r2 = r.max()
    c1 = c.min()
    c2 = c.max()

    mesh=depth_2_mesh_bbx(depth, [r1,r2,c1,c2], K, step=1, enl=0)

    return mesh

def getFiles_ab(file_dir,suf,a,b):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        #print('root: ',dirs)
        for file in files:
            if os.path.splitext(file)[1] == suf:
                #print(os.path.join(root, file))
                #sdf
                L.append(os.path.join(root, file))
        # L.sort(key=lambda x:int(x[-9:-4])) # 0000
        L.sort(key=lambda x: int(x[a:b]))#0000.png
    return L

def shake_bbx(bbx, degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1.2), shear=(0, 0),W=640,H=480):

    #### bbx: x1,x2,y1,y2
    # random.seed(1092)
    # print(random.random())
    bw = bbx[1]-bbx[0]
    cx = bbx[0]+bw//2
    bh = bbx[3]-bbx[2]
    cy = bbx[2]+bh//2
    targets = np.array([bbx[0], bbx[2], bbx[1], bbx[3]])
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R = np.eye(3)

    # R[:2] = cv2.getRotationMatrix2D(angle=a, center=(W / 2, H / 2), scale=s)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(int(cx), int(cy)), scale=s)
    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * bw  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * bh   # y translation (pixels)
    # T[0, 2] = translate[0] * bw  # x translation (pixels)
    # T[1, 2] = translate[1] * bh  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    # M = np.eye(3)
    # if len(targets) > 0:
    n = 1
    points = targets.copy().reshape(1,4)
    # print(random.random())


    # warp points
    xy = np.ones((n * 4, 3))
    xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(n, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

    # apply angle-based reduction of bounding boxes
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

    # reject warped points outside of image
    x1 = int(np.clip(xy[0][0], 0, W))
    x2 = int(np.clip(xy[0][2], 0, W))
    y1 = int(np.clip(xy[0][1], 0, H))
    y2 = int(np.clip(xy[0][3], 0, H))

    return np.array([x1, x2,y1, y2])


def depth_out_iou(depth, box1_yolo, box2_gt, z=0):
    ioubb = get_bbxs_iou(box1_yolo, box2_gt)

    if z == 0:
        depth[ioubb[2]:ioubb[3], ioubb[0]:ioubb[1]] = 0

    return depth, ioubb

def get_bbxs_iou(box1, box2):
    '''

    :param box1: x1,x2, y1 , y2
    :param box2: x1 x2 y1 y2
    :return:
    '''
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[2], box1[1], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[2], box2[1], box2[3]

    x2 = min(b1_x2, b2_x2)
    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    y2 = min(b1_y2, b2_y2)

    return int(x1),int(x2),int(y1),int(y2)

def defor_3D(pts,lab, R,T,pc ,scalex=(1,1),scalez=(1,1),scaley=(1,1),scale=(1,1), cate='',OR=None,bm=0.5):

    ## follow by xyz order
    # x_nr = x_r + x_r*np.random.uniform(scalex[0], scalex[1])
    # y_nr = y_r + y_r*np.random.uniform(scaley[0], scaley[1])
    # z_nr = z_r + z_r*np.random.uniform(scalez[0], scalez[1])
    ptsn = pts.copy()
    ptsn =np.dot(R.T ,(ptsn-T).T).T # Nx3
    # ptsn = np.dot(R.T, ptsn.T).T  # Nx3
    # show_mulit_mesh([ptsn, pts-T])
    ex = random.uniform(scalex[0], scalex[1])
    ez = random.uniform(scalez[0], scalez[1])
    ey = random.uniform(scaley[0], scaley[1])
    OR,lx,ly,lz ,miny,maxy= get_3D_corner_def(pc)
    # show_mulit_mesh([pc])

    ptsn[:, 0] = ptsn[:, 0] * ex
    ptsn[:, 1] = ptsn[:, 1] * ey
    ptsn[:, 2] = ptsn[:, 2] * ez

    s=0
    if cate =='mug' or cate=='bowl':
        s = random.uniform(-(1 - scale[0]), -(1 - scale[1]))
        mb = bm
        # print(mb, s)
        #
        if mb>0.5:
            s = s * ((maxy-ptsn[np.where(lab == 1)[0], 1]) / ly) ## bottle change (screen down)
        else:
            s = s * ((ptsn[np.where(lab == 1)[0], 1] - miny) / ly) ## mouth change (screen up)
        ptsn[np.where(lab==1)[0], 0] = ptsn[np.where(lab==1)[0], 0]*(1+s)

        ptsn[np.where(lab==1)[0], 2] = ptsn[np.where(lab==1)[0], 2]*(1+s)




    ptsn = np.dot(R, ptsn.T).T+T

    # print('inside func')
    # show_mulit_mesh([ptsn])





    return ptsn, ex,ey, ez,s

def pts_iou(pts,label, K, seg):
    '''

    :param pts: N,3
    label: N,1
    :param K: 3,3
    :param seg: x1, x2, y1 y2
    :return:
    '''
    fx = K[0,0]
    ux = K[0,2]
    fy = K[1,1]
    uy = K[1,2]
    # dep3d = dep3d[np.where(dep3d[:, 0] != 0.0)]
    # dep3d = dep3d[np.where(dep3d[:, 1] != 0.0)]
    # dep3d = dep3d[np.where(dep3d[:, 2] != 0.0)]
    x1 = seg[0]
    x2 = seg[1]
    y1 = seg[2]
    y2 = seg[3]

    pts1 = pts[np.where(((pts[:,0]*fx+ux*pts[:,2])/pts[:,2])>x1)]
    label1 = label[np.where(((pts[:,0]*fx+ux*pts[:,2])/pts[:,2])>x1)]

    pts2 = pts1[np.where(((pts1[:, 0] * fx + ux * pts1[:, 2]) / pts1[:, 2]) < x2)]
    label2 = label1[np.where(((pts1[:, 0] * fx + ux * pts1[:, 2]) / pts1[:, 2]) < x2)]

    pts3 = pts2[np.where(((pts2[:, 1] * fy + uy * pts2[:, 2]) / pts2[:, 2]) > y1)]
    label3 = label2[np.where(((pts2[:, 1] * fy + uy * pts2[:, 2]) / pts2[:, 2]) > y1)]


    pts4 = pts3[np.where(((pts3[:, 1] * fy + uy * pts3[:, 2]) / pts3[:, 2]) < y2)]
    label4 = label3[np.where(((pts3[:, 1] * fy + uy * pts3[:, 2]) / pts3[:, 2]) < y2)]

    return pts4, label4

def get_3D_corner_def(pc):
    pc=move_2_C(pc)
    x_r=max(pc[:,0])-min(pc[:,0])
    y_r=max(pc[:,1])-min(pc[:,1])
    z_r=max(pc[:,2])-min(pc[:,2])

    ext1=np.array([0,x_r,y_r,z_r])
    or1=np.array([ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or2=np.array([ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or3=np.array([ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or4=np.array([ext1[1]/2,-ext1[2]/2,ext1[3]/2])

    or5=np.array([-ext1[1]/2,-ext1[2]/2,-ext1[3]/2])
    or6=np.array([-ext1[1]/2,ext1[2]/2,-ext1[3]/2])
    or7=np.array([-ext1[1]/2,ext1[2]/2,ext1[3]/2])
    or8=np.array([-ext1[1]/2,-ext1[2]/2,ext1[3]/2])

    OR=np.array([or1,or2,or3,or4,or5,or6,or7,or8])

    return OR, x_r,y_r,z_r,min(pc[:,1]),max(pc[:,1])

def var_2_norm(pc,ex,ey, ez,c=''):
    ## cats = ['bottle','bowl','camera','can','laptop','mug']
    # cats = ['02876657', '02880940', '02942699', '02946921', '03642806', '03797390']
    if c=='bottle':
        unitx = 87
        unity = 220
        unitz = 89
    elif c=='bowl':
        unitx = 165
        unity = 80
        unitz = 165
    elif c == 'camera':
        unitx = 88
        unity = 128
        unitz = 156
    elif c=='can':
        unitx = 68
        unity = 146
        unitz = 72
    elif c=='laptop':
        unitx = 346
        unity = 200
        unitz = 335
    elif c=='mug':
        unitx = 146
        unity = 83
        unitz = 114
    elif c=='02876657':
        unitx = 324/4
        unity = 874/4
        unitz = 321/4
    elif c == '02880940':
        unitx = 675/4
        unity = 271/4
        unitz = 675/4
    elif c=='02942699':
        unitx = 464/4
        unity = 487/4
        unitz = 702/4
    elif c=='02946921':
        unitx = 450/4
        unity = 753/4
        unitz = 460/4
    elif c=='03642806':
        unitx = 581/4
        unity = 445/4
        unitz = 672/4
    elif c=='03797390':
        unitx = 670/4
        unity = 540/4
        unitz = 497/4
    else:
        print('This category is not recorded in my little brain.')



    OR, lx, ly, lz, miny, maxy = get_3D_corner_def(pc)

    lx = lx * ex
    ly = ly * ey
    lz = lz * ez



    return lx-unitx,ly-unity, lz-unitz