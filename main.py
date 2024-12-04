import argparse
from dataset.dataset import *
from visualization.visual_cam_points import *
from torch.utils.tensorboard import SummaryWriter
from optimization import CameraLayerOpt
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str, default='random/moto/')
    parser.add_argument('--lr1',type=float,default=1e-3)
    parser.add_argument('--lr2',type=float,default=1e-3)
    parser.add_argument('--height',type=float,default=2)
    parser.add_argument('--cameranum',type=int,default=10)
    parser.add_argument('--epoches',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--decay',type=float,default=1e-4)
    parser.add_argument('--optimizer',type=str,default="Adam")
    parser.add_argument('--kcoverage',type=int,default=3)
    parser.add_argument('--isscene',type=int,default=False)
    parser.add_argument('--modelname',type=str,default='scene/room_0.ply')
    parser.add_argument('--voxelnum',type=int,default=30000)
    parser.add_argument('--voxelsize',type=float,default=0.02)
    args = parser.parse_args()
    #######################filePath###########################################
    tensorboarddir="tensorboardModel/"
    modelpath="modelpath/"+args.path
    posepath="resultModel/"+args.path+"pose/"
    pcdpath="resultModel/"+args.path
    if os.path.exists(posepath)==False:
        os.makedirs(posepath)
    tensorboardpath=tensorboarddir+args.path
    if os.path.exists(tensorboardpath)==False:
        os.makedirs(tensorboardpath) 
    writer=SummaryWriter(tensorboardpath)
    ###########################################################################   
    pcd,pointnormals,voxelnormals,minbound,camerapose,scale=Initdatafromrandom(args)
    camlayopt = CameraLayerOpt(args,pointnormals,voxelnormals,minbound,scale,posepath,writer)
    position=camerapose[:,:3]
    rotation=compute_rotation_matrix_from_ortho6d(camerapose[:,3:])
    posename=posepath+str(0)+".npy"
    saveTrainingResult(posename,position,rotation,scale)
    ###visualization mesh/ply with camera placement
    visMesh(pcd,position.cpu().numpy(),rotation.cpu().numpy(),flag=1)
    #############################################################################
    position,rotation = camlayopt.opt(camerapose)
    visMesh(pcd,position,rotation,flag=1)
