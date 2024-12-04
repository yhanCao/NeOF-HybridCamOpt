import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter

class Direct(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.d_a=torch.ones(size=[1],dtype=torch.float,requires_grad=False)
    self.d_b=torch.ones(size=[1],dtype=torch.float,requires_grad=False)
  def forward(self,x_world,voxel_point,voxel_normal,score):
    #x_world (m,1,3)
    #voxel_point(1,n,3)
    #voxel_normal(n,3)
    #score(n,1)
    distance=torch.linalg.norm(x_world-voxel_point[:,:,:3],dim=-1)
    with torch.no_grad():
        x_normal=torch.mean(voxel_normal[torch.sort(distance,1)[1][:,:8]],1).unsqueeze(1) #(m,1,3)
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        output=cos(x_normal.repeat(1,len(voxel_normal),1),voxel_normal.unsqueeze(0).repeat(x_normal.shape[1],1,1))
        score_num=torch.sum(output>0.75,dim=1)
        nonzerojudge=(score_num!=0).nonzero().squeeze()
    score_sum=torch.sum(score.repeat(x_world.shape[0],1)*(output>0.75)/(torch.exp(distance)),1)
    x_world_field=score_sum[nonzerojudge]/score_num[nonzerojudge]
    output.cpu().numpy(),score_num.cpu().numpy(),x_normal.cpu().numpy()
    del output,score_num
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return x_world_field,nonzerojudge
class Addparam(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.d_a=Parameter(torch.ones(size=[1],dtype=torch.float,requires_grad=True))
    self.d_b=Parameter(torch.ones(size=[1],dtype=torch.float,requires_grad=True))
  def forward(self,x_world,voxel_point,voxel_normal,score):
    #x_world (m,1,3)
    #voxel_point(1,n,3)
    #voxel_normal(n,3)
    #score(n,1)
    distance=torch.linalg.norm(x_world-voxel_point[:,:,:3],dim=-1)
    with torch.no_grad():
        x_normal=torch.mean(voxel_normal[torch.sort(distance,1)[1][:,:8]],1).unsqueeze(1) #(m,1,3)
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        output=cos(x_normal.repeat(1,len(voxel_normal),1),voxel_normal.unsqueeze(0).repeat(x_normal.shape[1],1,1))
        score_num=torch.sum(output>0.8,dim=1)
        nonzerojudge=(score_num!=0).nonzero().squeeze()
    score_sum=torch.sum(score.repeat(x_world.shape[0],1)*(output>0.8)/(self.d_a*torch.exp(self.d_b*distance)),1)
    x_world_field=score_sum[nonzerojudge]/score_num[nonzerojudge]
    output.cpu().numpy(),score_num.cpu().numpy(),x_normal.cpu().numpy()
    del output,score_num
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return x_world_field,nonzerojudge
class AddAttention(torch.nn.Module):
  def __init__(self, input_channel,output_channel,num_heads):
    super(AddAttention, self).__init__()
    self.num_heads = num_heads
    self.output_channel = output_channel

    assert output_channel % num_heads == 0

    self.depth = output_channel // num_heads

    self.Wq = nn.Linear(output_channel, output_channel)
    self.Wk = nn.Linear(output_channel, output_channel)
    self.fc = nn.Linear(input_channel, output_channel)
    # self.position_encoding=GeometryPositionEncodingSine(1)
  def calculate_x(self,x_world,voxel_point,voxel_normal,density=None):
    #x_world(m,1,3) voxel_point(1,n,3) voxel_normal(n,3)
    with torch.no_grad():
      distance=torch.linalg.norm(x_world-voxel_point[:,:,:3],dim=-1)
      x_normal=torch.mean(voxel_normal[torch.sort(distance,1)[1][:,:8]],1).unsqueeze(1) #(m,1,3)
      cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
      cos_similarity=cos(x_normal.repeat(1,len(voxel_normal),1),voxel_normal.unsqueeze(0).repeat(x_normal.shape[1],1,1)).unsqueeze(-1)
    position_relative=x_world-voxel_point
    x=torch.cat((position_relative,cos_similarity),-1)
    if density != None:
      density=density.repeat(x.shape[0],1)
      x=torch.cat((x,density.unsqueeze(-1)),-1)
    del distance,x_normal,cos_similarity,position_relative
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return x
  def calculate_x_voxelselect(self,x_world,voxel_point,voxel_normal,v,density=None):
    #x_world(m,1,3) voxel_point(1,n,3) voxel_normal(n,3)
    with torch.no_grad():
      distance=torch.linalg.norm(x_world-voxel_point.unsqueeze(0)[:,:,:3],dim=-1)
      voxel_point_select=voxel_point[torch.sort(distance,1)[1][:,:30]]
      voxel_normal_select=voxel_normal[torch.sort(distance,1)[1][:,:30]]
      v=v[torch.sort(distance,1)[1][:,:30]]
      x_normal=torch.mean(voxel_normal[torch.sort(distance,1)[1][:,:8]],1).unsqueeze(1) #(m,1,3)
      normal_relative=x_normal-voxel_normal_select
    position_relative=x_world-voxel_point_select
    x=torch.cat((position_relative,normal_relative),-1)
    if density != None:
      density=density.repeat(x.shape[0],1)
      x=torch.cat((x,density.unsqueeze(-1)),-1)
    del distance,x_normal,normal_relative,position_relative
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return x,v
    
  def forward(self,x_world,voxel_point,voxel_normal,v,density=None,mask=None):
    #x_world (m,1,3)
    #voxel_point (1,n,3)
    #voxel_normal (n,3)
    #v (n,1)
    ###get attention input x
    x,v=self.calculate_x_voxelselect(x_world,voxel_point,voxel_normal,v,density)
    # print(x.shape)
    x=self.fc(x)
    # x=self.position_encoding(x)
    # print(x.shape)
    batch_size = x.size(0)

    # Perform linear operation and split into h heads
    Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)
    K = self.Wk(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)
    # V = self.Wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)
    # Scaled Dot-Product Attention
    # print(Q.shape,K.shape)
    scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(self.depth)
    # print(scores.shape,v.shape)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(scores,dim=-1)
    attention=attention.squeeze(1)
    # apply attention to value
    out = torch.matmul(attention,v).squeeze(-1)
    # average heads
    out = torch.mean(out,dim=1)
    # out= torch.mean(out,dim=1)
    return out