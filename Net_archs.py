# @Time    : 06/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import torch.nn as nn
import gcn3d
import torch
import torch.nn.functional as F


class GCN3D_segR(nn.Module):
    def __init__(self, class_num,vec_num, support_num, neighbor_num):
        super(GCN3D_segR, self).__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


        self.classnum = class_num
        self.vecnum = vec_num*3
        dim_fuse = sum([128, 128, 256, 256, 512, 512, 16])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num+vec_num*3, 1),
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                onehot: "tensor (bs, cat_num)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)


        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1,2)).transpose(1,2), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                     v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4,onehot], dim= 2)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global, onehot], dim= 2)

        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch) ## B N 50?
        seg = pred[:,:,0:self.classnum]
        vecs = pred[:,:, self.classnum:self.classnum+self.vecnum]
        return seg, vecs, feat


class Point_center(nn.Module):
    def __init__(self):
        super(Point_center, self).__init__()

        # self.conv1 = torch.nn.Conv2d(12, 64, 1) ##c
        self.conv1 = torch.nn.Conv1d(3, 128, 1) ## no c
        self.conv2 = torch.nn.Conv1d(128, 256, 1)

        ##here
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        # self.conv4 = torch.nn.Conv1d(1024,1024,1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        # self.bn4 = nn.BatchNorm1d(1024)
        # self.global_feat = global_feat

    def forward(self, x,obj):## 5 6 30 1000
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x))) ## 5 64 30 1000
        x = F.relu(self.bn2(self.conv2(x))) ## 5 64 1 1000
        x = (self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x2 = torch.max(x, -1, keepdim=True)[0]#5 512 1
        # x2=torch.mean(x, -1, keepdim=True)
        obj = obj.view(-1, 1)
        one_hot = torch.zeros(batchsize, 16).scatter_(1, obj.cpu().long(), 1)
        # print(one_hot[1,:])
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        one_hot2 = one_hot.unsqueeze(2)
        return torch.cat([x2, one_hot2],1)
        #
        # return x2
    # return pointfeat2

class Point_center_res_cate(nn.Module):
    def __init__(self):
        super(Point_center_res_cate, self).__init__()

        # self.feat = Point_vec_edge()
        self.feat = Point_center()
        self.conv1 = torch.nn.Conv1d(512+16, 256,1)
        self.conv2 = torch.nn.Conv1d(256, 128,1)
        # self.drop1 = nn.Dropout(0.1)
        self.conv3 = torch.nn.Conv1d(128, 6,1 )


        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x, obj):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # print(x.size())
        # tes
        x = self.feat(x, obj) ## Bx1024x1xN
        T_feat = x
        # x=x.squeeze(2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = (self.bn2(self.conv2(x)))

        x=self.drop1(x)
        x = self.conv3(x)



        x = x.squeeze(2)
        x=x.contiguous()##Bx6
        xt = x[:,0:3]
        xs = x[:,3:6]

        return xt,xs

class Rot_green(nn.Module):
    def __init__(self, k=24,F=1036):
        super(Rot_green, self).__init__()
        self.f=F
        self.k = k


        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256,256,1)
        self.conv4 = torch.nn.Conv1d(256,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x=self.drop1(x)
        x = self.conv4(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x


class Rot_red(nn.Module):
    def __init__(self, k=24,F=1036):
        super(Rot_red, self).__init__()
        self.f=F
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256,256,1)
        self.conv4 = torch.nn.Conv1d(256,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x=self.drop1(x)
        x = self.conv4(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x