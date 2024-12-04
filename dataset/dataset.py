import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from .init_camera import *
from .pcd import *

def readCamerapose(path):
  if os.path.exists(path):
    camerapose=np.load(path)
    print(camerapose.shape)
    position=camerapose[:,:3]
    rotation=camerapose[:,3:].reshape(-1,3,3)
    return position,rotation

def Initdatafromrandom(args):
    pointspath="data/"+args.modelname
    if args.voxelnum is not None:
        pcd,pointnormals,voxelnormals,minbound,size=getPointNormalfromPly(pointspath,args.voxelnum,args.voxelsize,args.kcoverage)
    else :
        pointnormals=getPointNormalfromFile(pointspath)
    Rs,Cs=initRandomCameras(args.cameranum,pointnormals,args.height)
    camerapose=getCameraPose(Rs,Cs)
    return pcd,pointnormals,voxelnormals,minbound,npToTensor(camerapose,dtype=torch.float),size

def VisReadData(camerapath,pointspath,voxelnum=None):
    if voxelnum is not None:
        pcd,pointnormals,size=getPointNormalfromPly(pointspath,voxelnum,normalize=False,centeral=False)
    else :
        pointnormals=getPointNormalfromFile(pointspath)
    Rs,Cs=readCamerapose(camerapath)
    return pcd,npToTensor(pointnormals),Cs,Rs,size

def generateP(num):
    Cmin=np.array([-1,-1])
    Cmax=np.array([1,1])
    points=np.random.random((num,2))*(Cmax-Cmin)+Cmin
    ones=np.ones((num,1))*0.8
    points=np.append(points,ones,1)
    normals=np.zeros_like(points)
    normals[:,2]=1
    pointnormals=np.append(points,normals,1) 
    print(pointnormals)
    return pointnormals
