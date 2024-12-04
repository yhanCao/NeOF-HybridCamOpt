import torch
import numpy as np
from scipy.spatial.transform import Rotation as R 
w,h=640,480
intrinsic=np.array([[320.0,0,319.5],
                   [0,320.0,239.5], 
                   [0,0,1]])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###########calculate rotation matrix from parameter 6D
# batch*n
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    z_raw = ortho6d[:,3:6]#batch*3
        
    z = normalize_vector(z_raw) #batch*3
    y = cross_product(z,x_raw) #batch*3
    y = normalize_vector(y)#batch*3
    x = cross_product(y,z)#batch*3
        
    x = x.view(-1,1,3)
    y = y.view(-1,1,3)
    z = z.view(-1,1,3)
    matrix = torch.cat((x,y,z), 1) #batch*3*3
    return matrix.to(torch.float)
####rotationTransform################
def RotationToEuler(Rmnumpy):
    euler=np.zeros(shape=(len(Rmnumpy),3))
    for i in range(len(Rmnumpy)):
        r=R.from_matrix(Rmnumpy[i])
        euler[i]=r.as_euler('xyz',degrees=True)
    return euler
def EulerToRotaionMatrix(euler):
    rotate=np.zeros(shape=(len(euler),3,3))
    for i in range(len(euler)):
        r=R.from_euler('zyx',euler[i],degrees=True)
        rotate[i]=r.as_matrix()
    return rotate
#######################numpy to tensor in device#########################
def npToTensor(matrix,dtype=torch.float):
	return torch.tensor(matrix,dtype=dtype,requires_grad=False).to(device)
def saveTrainingResult(path,position,rotation,scale=1):
    rotate=rotation.reshape(-1,9)
    cameraposedata=torch.cat((position,rotate),1)
    cameraposedata=cameraposedata.detach().cpu().numpy()
    cameraposedata[:,:3]=cameraposedata[:,:3]*scale[0]+scale[1]
    np.save(path,cameraposedata)