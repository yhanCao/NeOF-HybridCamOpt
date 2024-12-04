from scipy.spatial.transform import Rotation as R
import math
import torch
import numpy as np
from dataset.utils import *
class FarthestSampler:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    def __call__(self, pts, k):
        index=np.zeros(k, dtype=int)
        farthest_pts = np.zeros((k, 6), dtype=np.float32)
        farthest_pts[0] = pts[0]
        distances = self._calc_distances(farthest_pts[0,:3], pts[:,:3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            index[i]=np.argmax(distances)
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i,:3], pts[:,:3]))
        return farthest_pts,index
class KNNaverageCamera:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    def __call__(self, selectposition, pointnormals,k):
        selectnormal=np.zeros_like(selectposition)
        for i in range(len(selectposition)):
            distance=self._calc_distances(selectposition[i],pointnormals[:,:3])
            index=np.argsort(distance)[:k]
            selectnormal[i]=np.mean(pointnormals[index,3:],axis=0)
        return selectnormal
class KnnPoints:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    def __call__(self,pointnormals,k):
        index_all=torch.zeros([len(pointnormals),k],dtype=int)
        for i in range(len(pointnormals)):
            normali=pointnormals[i,3:].unsqueeze(-1)
            mm=torch.mm(pointnormals[:,3:],normali).squeeze(-1)
            index1=(mm>0).nonzero().squeeze(-1)
            pointselect=torch.index_select(pointnormals,0,index1)
            distance=self._calc_distances(pointnormals[i,:3],pointselect[:,:3])
            _,index=torch.sort(distance,descending=False)
            index_all[i]=index1[index[:k]]
        return index_all
#####################################init with existed camera parameter###################
def read_CameraParams(num,camerapath):
    cameraExlist={}
    for i in range(num):
        # name=camerapath+"CCD"+str(i+1)+".cal"
        name=camerapath+"CCD"+str(i)+".txt"
        intrinsic = np.genfromtxt(name,dtype=float,delimiter=' ',skip_footer=3)[:,:3]
        extrinsic = np.genfromtxt(name,dtype=float,delimiter=' ',skip_header=4)[:,:4]
        cameraExlist[i]=extrinsic
    return intrinsic,cameraExlist
def cameraextrinc(num,camerapath):
    intrinsic,cameraExlist=read_CameraParams(num,camerapath)
    Rs={}
    Cs={}
    for key,value in cameraExlist.items():
        R=value[:,0:3]
        T=value[:,3:]
        C=np.dot(-R.transpose(),T).reshape(3)
        # print(C)
        Rs[key]=R
        Cs[key]=C
    return intrinsic,Rs,Cs

############################################init with random camera parameter##############
def getQuaternion(fromVector, toVector):
        fromVector = np.array(fromVector)
        fromVector_e = fromVector / np.linalg.norm(fromVector)
        toVector = np.array(toVector)
        toVector_e = toVector / np.linalg.norm(toVector)
        cross = np.cross(toVector_e, fromVector_e)
        cross_e = cross / np.linalg.norm(cross)
        dot = np.dot(fromVector_e, toVector_e)
        angle = math.acos(dot)
        # print("angle",angle,"dot:",dot,"tovector:",toVector)
        if angle == 0 :
            return 1
        elif angle == math.pi:
            return -1
        else:
            return [cross_e[0]*math.sin(angle/2), cross_e[1]*math.sin(angle/2), cross_e[2]*math.sin(angle/2), math.cos(angle/2)]
def QuatToRotateMatrix(quat):
    if quat == 1:
        I=np.identity(3)
        for i in range(len(I)):
            for j in range(len(I[0])):
                if I[i][j]==0 :
                    I[i][j]=1e-8
        return np.identity(3)
    elif quat == -1 :
        I=np.identity(3)
        for i in range(len(I)):
            for j in range(len(I[0])):
                if I[i][j]==0 :
                    I[i][j]=1e-8
        I[2,2]=-1
        return I
    else:
        r1=R.from_quat(quat)
        rotate=r1.as_matrix()
    return rotate
def initRandomCameras(cameranum,pointnormals,height):
    # fs=FarthestSampler()
    # camera,index=fs(pointnormals,cameranum)
    # knn=KNNaverageCamera()
    # camera[:,3:]=knn(camera[:,:3],pointnormals,30)
    index=np.random.randint(low=0,high=len(pointnormals)-1,size=cameranum)
    camera=pointnormals[index]
    Cs={}
    Rs={}
    for i in range(len(camera)):
        zaxis=-camera[i,3:]
        Cs[i]=camera[i,:3]+height*camera[i,3:]
        print("camera xyz:",camera[i,:3],camera[i,3:])
        Rs[i]=QuatToRotateMatrix(getQuaternion(np.array([0,0,1]),zaxis))
    return Rs,Cs
##################################get standard camera parameter from intrinsic,Rs,Cs##############################
# generate camerapose matrix (cameranum,9)
def getCameraPose(Rs,Cs):
    camerapose=np.empty(shape=[0,9])
    for i in range(len(Cs)):
        cpose=np.append(Cs[i].reshape(1,3),np.append(Rs[i][0,:].reshape(1,3),Rs[i][2,:].reshape(1,3),axis=1),axis=1)
        camerapose=np.append(camerapose,cpose,axis=0)
    return camerapose
def intrinsicTotensor(cameranum,intrinsic):
    I=np.tile(intrinsic,(cameranum,1,1))
    return torch.tensor(I,dtype=torch.float,requires_grad=False).to(device)

