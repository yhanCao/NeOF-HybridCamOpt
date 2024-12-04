from dataset.dataset import *
from torch.nn import Parameter
from field.field_attribute import get_visiblep_opt
def angle_between_vectors(vec, mat):
    return (torch.acos(torch.mm(mat, vec.unsqueeze(-1)).squeeze(-1) / (torch.norm(mat,dim=1) * torch.norm(vec)))) * (180 / torch.pi)
class GenerateP3d(torch.nn.Module):
    def __init__(self,cameranum):
        super().__init__()
        self.weightp=Parameter(torch.zeros(size=[cameranum,3]))
        self.weightr1=Parameter(torch.zeros(size=[cameranum,3]))
        self.weightr2=Parameter(torch.zeros(size=[cameranum,3]))
        self.softmax=torch.nn.Softmax(dim=1)
        self.softplus=torch.nn.Softplus(beta=0.1)
    def reconstruction3D(self,P,point):
        point=npToTensor(point)
        A=torch.zeros(3*len(P),4+len(P)).to(device)
        for i in range(len(P)):
            A[3*i:3*(i+1),:4]=-P[i]
            A[3*i:3*(i+1),4+i]=point[i]
        U, S, V = torch.svd(A)
        output0 = (V[0][-1]/V[3][-1]).to(torch.float32)
        output1 = (V[1][-1]/V[3][-1]).to(torch.float32)
        output2 = (V[2][-1]/V[3][-1]).to(torch.float32)
        return [output0,output1,output2]
    def getReconstructionLoss(self,camerapose,voxelnormals,height):
        rotate=camerapose[:,3:]*(1+torch.cat((self.weightr1,self.weightr2),1))
        position=camerapose[:,:3]+(camerapose[:,:3]*self.weightp)
        rotation=compute_rotation_matrix_from_ortho6d(rotate)
        t=torch.bmm(-rotation,position.unsqueeze(-1))
        E=torch.cat((rotation,t),-1)
        I=npToTensor(intrinsic).repeat(len(rotation),1,1)
        P=torch.bmm(I,E)
        relation=np.zeros([0,5])
        for i in range(len(position)):
            with torch.no_grad():
                x_select,index=get_visiblep_opt(voxelnormals,position[i].data.cpu().numpy(),rotation[i].data.cpu().numpy(),200,min_h=height*0.5,max_h=height*1.5)
                point3d=np.dot(intrinsic,x_select.T).T
                point3d=point3d/point3d[:,-1][:,None]
                relation=np.append(relation,np.append(index[:,None],np.append(np.ones([len(index),1])*i,point3d,1),1),0)
        reconstruction_loss = 0
        for j in range(len(voxelnormals)):
            index=(relation[:,0]==j).nonzero()[0]
            if len(index) >=2:
                output=self.reconstruction3D(P[relation[index,1]],relation[index,2:])
                reconstruction_loss += torch.sqrt((output[0]-voxelnormals[j,0])**2+(output[1]-voxelnormals[j,1])**2+(output[2]-voxelnormals[j,2])**2)
        return reconstruction_loss
    def forward(self,camerapose,voxelnormals,voxelmodel,fieldmodel,height):
        torch.cuda.empty_cache()
        ###get max z of current camera
        rotate=camerapose[:,3:]*(1+torch.cat((self.weightr1,self.weightr2),1))
        zaxis=rotate[:,3:]
        zaxis=zaxis/torch.linalg.norm(zaxis,dim=1,keepdim=True)
        ##################沿着垂直于z轴的平面移动#####################
        ##########test############
        position=camerapose[:,:3]+(camerapose[:,:3]*self.weightp)-torch.bmm((camerapose[:,:3]*self.weightp).unsqueeze(1),zaxis.unsqueeze(-1)).squeeze(-1)*zaxis
        # position=camerapose[:,:3]+(self.weightp)-torch.bmm((self.weightp).unsqueeze(1),zaxis.unsqueeze(-1)).squeeze(-1)*zaxis
        # position=camerapose[:,:3]+(camerapose[:,:3]*self.weightp)
        rotation=compute_rotation_matrix_from_ortho6d(rotate)
        ##########################################################
        voxelAll=torch.zeros([len(camerapose),3],requires_grad=True).to(device)
        for i in range(len(position)):
            with torch.no_grad():
                x_select,_=get_visiblep_opt(voxelnormals,position[i].data.cpu().numpy(),rotation[i].data.cpu().numpy(),200,min_h=height*0.5,max_h=height*1.5)
            x_world=torch.mm(rotation[i].T,npToTensor(x_select.T)).T+position[i].unsqueeze(0) 
            # print(i,x_world.shape)
            judge=(torch.abs(angle_between_vectors(-zaxis[i],voxelmodel[:,3:6]))<90).nonzero().squeeze()
            ### use correct voxel to calcuate loss
            x_world_field=fieldmodel(x_world.unsqueeze(1),voxelmodel[judge,:3],voxelmodel[judge,3:6],voxelmodel[judge,6:])

            voxelAll[i] +=torch.sum(x_world_field.squeeze(-1),dim=0)
            ###################calculate each voxel uncoverage num ####################################
        return voxelAll,position.data,rotation.data