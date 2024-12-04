import time
from dataset.dataset import *
from field.visibility_field import *
from field.optimization import GenerateP3d
from field.field_attribute import voxel_model,get_visible_points

class CameraLayerOpt:
    def __init__(self,args,pointnormals,voxelnormals,minbound,scale,posepath,writer):
        self.args = args
        self.log_writer = writer
        self.pointnormals = pointnormals
        self.voxelnormals = voxelnormals
        self.minbound = minbound
        self.scale = scale
        self.posepath = posepath
        self.model=GenerateP3d(args.cameranum).to(device)
        self.fieldmodel = AddAttention(6,16,1).to(device)
    def judgeDistance(self,position,points):
        distance=np.linalg.norm(position-points,axis=1)
        judge=(distance<=0.02).nonzero()[0]
        if len(judge)==0:
            return True
        else:
            return False
    def judgeBoundingBox(self,p,points):
        min=np.min(points,axis=0)
        max=np.max(points,axis=0)
        if len((p>min).nonzero()[0]) ==3 and len((p<max).nonzero()[0]) ==3 :
            return True
        else:
            return False
    def resetNewforScene(self,position,rotation,min_camera):
        voxelmodel,voxel_visibility=voxel_model(self.args,self.voxelnormals,rotation,position,min_h=self.args.height*0.5,max_h=self.args.height*1.5)
        ############get current score#######
        with torch.no_grad():
            uncoverage_mean=self.fieldmodel(voxelmodel[:,:3].unsqueeze(1),voxelmodel[:,:3],voxelmodel[:,3:6],voxelmodel[:,6:]).cpu().numpy()
            origin_score=np.sum(np.square(uncoverage_mean[:,0]))
        rr=rotation[min_camera]
        rp=position[min_camera]
        if self.judgeDistance(rp,self.voxelnormals[:,:3]) == False :
            origin_score=1e6
        sample=100 
        # print("origin",origin_score)
        uncoverage_new=np.argsort(uncoverage_mean[:,0])[::-1][:sample]
        center=np.mean(self.voxelnormals[:,:3],axis=0)
        for i in range(sample):
            voxel_visibility[:,min_camera]=0
            zaxis=-self.voxelnormals[uncoverage_new[i],3:]
            r=QuatToRotateMatrix(getQuaternion(np.array([0,0,1]),zaxis))
            p=self.voxelnormals[uncoverage_new[i],:3]-self.args.height*zaxis
            if self.judgeBoundingBox(p,self.voxelnormals[:,:3]) == False or self.judgeDistance(p,self.voxelnormals[:,:3]) == False:
                zaxis=-zaxis
                p=self.voxelnormals[uncoverage_new[i],:3]-self.args.height*zaxis
                r=QuatToRotateMatrix(getQuaternion(np.array([0,0,1]),zaxis))
            if self.judgeBoundingBox(p,self.voxelnormals[:,:3]) == False or self.judgeDistance(p,self.voxelnormals[:,:3]) == False:
                zaxis=(self.voxelnormals[uncoverage_new[i],:3]-center)/np.linalg.norm(self.voxelnormals[uncoverage_new[i],:3]-center)
                p=self.voxelnormals[uncoverage_new[i],:3]-self.args.height*zaxis
                r=QuatToRotateMatrix(getQuaternion(np.array([0,0,1]),zaxis))
            if self.judgeDistance(p,self.voxelnormals[:,:3]) == False or self.judgeDistance(p,self.voxelnormals[:,:3]) == False :
                p=self.voxelnormals[uncoverage_new[i],:3]-self.args.height*0.5*zaxis
                r=QuatToRotateMatrix(getQuaternion(np.array([0,0,1]),zaxis))
            if self.judgeDistance(p,self.voxelnormals[:,:3]) == False or self.judgeDistance(p,self.voxelnormals[:,:3]) == False :
                continue
            judge,_=get_visible_points(self.voxelnormals,p,r,min_h=self.args.height*0.5,max_h=self.args.height*1.5)

            voxel_visibility[judge,min_camera]=1
            voxelmodel[:,-3]=npToTensor(self.args.kcoverage-np.clip(np.sum(voxel_visibility,axis=1),0,self.args.kcoverage))
            with torch.no_grad():
                uncoverage_mean=self.fieldmodel(npToTensor(self.voxelnormals[:,:3]).unsqueeze(1),voxelmodel[:,:3],voxelmodel[:,3:6],voxelmodel[:,6:]).cpu().numpy()
                current_score=np.sum(np.square(uncoverage_mean[:,0]))
                # print(current_score)
            if current_score < origin_score :
                rr=r
                rp=p
                origin_score=current_score
                print("current min:",current_score)
        # if min_camera==args.cameranum-1:
        #     vis(voxelmodelfinal,uncoverage_final,common_rows)
        rotation[min_camera]=rr
        position[min_camera]=rp
        torch.cuda.empty_cache()
        return  npToTensor(position),npToTensor(rotation)
    def resetNewforModel(self,position,rotation,min_camera):
        voxelmodel,voxel_visibility=voxel_model(self.args,self.voxelnormals,rotation,position,min_h=self.args.height*0.5,max_h=self.args.height*1.5)
        ############get current score#######
        with torch.no_grad():
            uncoverage_mean=self.fieldmodel(voxelmodel[:,:3].unsqueeze(1),voxelmodel[:,:3],voxelmodel[:,3:6],voxelmodel[:,6:]).cpu().numpy()
            origin_score=np.sum(np.square(uncoverage_mean[:,0]))
        rr=rotation[min_camera]
        rp=position[min_camera]
        sample=20 
        uncoverage_new=np.argsort(uncoverage_mean[:,0])[::-1][:sample]
        for i in range(sample):
            voxel_visibility[:,min_camera]=0
            zaxis=-self.voxelnormals[uncoverage_new[i],3:]

            r=QuatToRotateMatrix(getQuaternion(np.array([0,0,1]),zaxis))
            p=self.voxelnormals[uncoverage_new[i],:3]-self.args.height*zaxis

            judge,_=get_visible_points(self.voxelnormals,p,r,200,min_h=self.args.height*0.5,max_h=self.args.height*1.5)
            voxel_visibility[judge,min_camera]=1
            voxelmodel[:,-3]=npToTensor(self.args.kcoverage-np.clip(np.sum(voxel_visibility,axis=1),0,self.args.kcoverage))
            with torch.no_grad():
                uncoverage_mean=self.fieldmodel(npToTensor(self.voxelnormals[:,:3]).unsqueeze(1),voxelmodel[:,:3],voxelmodel[:,3:6],voxelmodel[:,6:]).cpu().numpy()
                current_score=np.sum(np.square(uncoverage_mean[:,0]))
                # print(current_score)
            if current_score < origin_score :
                rr=r
                rp=p
                origin_score=current_score
        rotation[min_camera]=rr
        position[min_camera]=rp
        torch.cuda.empty_cache()
        return  npToTensor(position),npToTensor(rotation)
    def metric(self,position,rotation):
        _,voxel_visibility=voxel_model(self.args,self.voxelnormals,rotation,position,min_h=self.args.height*0.8,max_h=self.args.height*1.2)
        _,point_visibility=voxel_model(self.args,self.pointnormals,rotation,position,min_h=self.args.height*0.8,max_h=self.args.height*1.2)
        p_coverage = np.sum(point_visibility,axis=-1)
        v_coverage = np.sum(voxel_visibility,axis=-1)
        rate_p=np.sum(np.square(np.maximum(self.args.kcoverage-p_coverage,np.zeros(len(self.pointnormals)))))/(self.args.kcoverage*self.args.kcoverage*len(self.pointnormals))
        rate_v=np.sum(np.square(np.maximum(self.args.kcoverage-v_coverage,np.zeros(len(v_coverage)))))/(self.args.kcoverage*self.args.kcoverage*len(v_coverage))
        return rate_p,rate_v
    def opt(self,camerapose):

        bestposition=camerapose[:,:3]
        bestrotation=compute_rotation_matrix_from_ortho6d(camerapose[:,3:])
        mmin=1
        bbp=bestposition
        bbr=bestrotation
        with torch.no_grad():
            rate_p,rate_v=self.metric(bbp.cpu().numpy(),bbr.cpu().numpy())
            print(f'Epoch: 0, RateP: {rate_p:.4f}')
            print(f'Epoch: 0, RateV: {rate_v:.4f}')
        for epoch in range(self.args.epoches):
            minRatev=1
            #get voxel visibility in each epoch
            position=copy.deepcopy(bestposition.data)
            rotation=copy.deepcopy(bestrotation.data)
            if  epoch % 5  ==0:
                for min_camera in range(self.args.cameranum):
                    if self.args.scene:
                        position,rotation=self.resetNewforScene(position.cpu().numpy(),rotation.cpu().numpy(),min_camera)
                    else:
                        position,rotation=self.resetNewforModel(position.cpu().numpy(),rotation.cpu().numpy(),min_camera)
                    saveTrainingResult(self.posepath+str(epoch)+"_"+str(min_camera+1)+".npy",position,rotation,self.scale)
            camerapose=torch.cat((position,torch.cat((rotation[:,0,:],rotation[:,2,:]),1)),1)
            if epoch <=10:
                start=time.time()
                voxelmodel,_=voxel_model(self.args,self.voxelnormals,rotation.cpu().numpy(),position.cpu().numpy(),min_h=self.args.height*0.8,max_h=self.args.height*1.2)
                fieldoptimizer = torch.optim.Adam(self.fieldmodel.parameters(),lr=1e-3*pow(0.99,epoch))
                fieldscheduler=torch.optim.lr_scheduler.StepLR(fieldoptimizer,step_size=1000,gamma=1e-4)
                # 训练模型
                num_epochs = 1000
                for iter in range(num_epochs):
                    # 前向传播
                    uncoverage_mean=self.fieldmodel(voxelmodel[:,:3].unsqueeze(1),voxelmodel[:,:3],voxelmodel[:,3:6],voxelmodel[:,6:]).squeeze(-1)
                    loss = torch.sum(torch.abs(voxelmodel[:,-3]-uncoverage_mean[:,0]))/len(uncoverage_mean)
                    # 反向传播和优化
                    fieldoptimizer.zero_grad()
                    loss.backward()
                    fieldoptimizer.step()
                    fieldscheduler.step()
                    if (iter+1) % 100 == 0:
                        print(f'Epoch [{iter+1}/{num_epochs}], Loss: {loss.item():.4f}')
                end =time.time()
                print("voxelmodel time:",end-start)
            self.minbound=npToTensor(self.minbound)
            #init optimizer model
            
            if self.args.optimizer == 'SGD':
                optimizer = torch.optim.SGD([{'params':self.model.weightp,'lr':self.args.lr1*pow(0.9,epoch)},{'params':self.model.weightr2,'lr':self.args.lr2*pow(0.9,epoch)},{'params':self.model.weightr1,'lr':self.args.lr1*pow(0.9,epoch)}])
            elif self.args.optimizer == 'Adam':
                optimizer = torch.optim.Adam([{'params':self.model.weightp,'lr':self.args.lr1*pow(0.9,epoch)},{'params':self.model.weightr2,'lr':self.args.lr2*pow(0.9,epoch)},{'params':self.model.weightr1,'lr':self.args.lr1*pow(0.9,epoch)}]
                )
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=self.args.decay)
            #optimizer
            
            for iters in range(self.args.iterations):
                start=time.time()
                optimizer.zero_grad()
                attribute,position,rotation=self.model(camerapose,self.voxelnormals,voxelmodel,self.fieldmodel,self.args.height)
                uperbound = torch.tensor([self.args.kcoverage,1,np.pi/2],dtype=torch.float,requires_grad=False).to(device)
                L=(uperbound - torch.mean(attribute/len(self.voxelnormals),dim=0))[0]
                L.backward()
                optimizer.step()
                scheduler.step()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    _,voxel_visibility=voxel_model(self.args,self.voxelnormals,rotation.data.cpu().numpy(),position.data.cpu().numpy(),min_h=self.args.height*0.8,max_h=self.args.height*1.2)
                    v_coverage = np.sum(voxel_visibility,axis=1)
                    rate_v=np.sum(np.square(np.maximum(self.args.kcoverage-v_coverage,np.zeros(len(v_coverage)))))/(self.args.kcoverage*self.args.kcoverage*len(v_coverage))
                    num_v=np.sum(np.sign(v_coverage))
                if rate_v < minRatev :
                    minRatev=rate_v
                    bestposition=copy.deepcopy(position)
                    bestrotation=copy.deepcopy(rotation)
                    if rate_v < mmin :
                        mmin=rate_v
                        bbp=copy.deepcopy(position)
                        bbr=copy.deepcopy(rotation)
                self.log_writer.add_scalar("loss",L,epoch*self.args.iterations+iters)

                self.log_writer.add_scalar("voxel_uncoverage_rate",rate_v,epoch*self.args.iterations+iters)
                self.log_writer.add_scalar("voxel_coverage_num",num_v,epoch*self.args.iterations+iters)
                print(f'Epoch: {epoch*self.args.iterations+iters:03d}, Loss: {L:.4f}')
                print(f'Epoch: {epoch*self.args.iterations+iters:03d}, RateV: {rate_v:.4f}')
                end=time.time()
                print(f'Epoch: {epoch*self.args.iterations+iters:03d}, Time: {(end-start):.4f}')
        with torch.no_grad():
            rate_p,rate_v=self.metric(bbp.cpu().numpy(),bbr.cpu().numpy())
            print(f'Epoch: {epoch*self.args.iterations+iters:03d}, RateP: {rate_p:.4f}')
            print(f'Epoch: {epoch*self.args.iterations+iters:03d}, RateV: {rate_v:.4f}')
        saveTrainingResult(self.posepath+"after.npy",bbp,bbr,self.scale)
        return bbp.cpu().numpy(),bbr.cpu().numpy()