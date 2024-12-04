from dataset.utils import *
import open3d as o3d
def get_fov_from_intrinsic_matrix(K):
    fx = K[0, 0]
    fy = K[1, 1]
    fov_x = 2 * np.arctan(K[0,2]/K[0,0])
    fov_y = 2 * np.arctan(K[1,2]/K[1,1])
    return fov_x, fov_y
def is_point_in_fov(point, rotation, position,min_h,max_h):
    #get aov
    # aov_x,aov_y=get_fov_from_intrinsic_matrix(intrinsic)
    # 将点从世界坐标系转换到相机坐标系
    point_c=np.dot(rotation,(point-position).T).T
    # 判断点是否在相机的视锥体内
    # judge_h=np.where(point_c[:,2]>=min_h)[0]
    judge_h=np.where((point_c[:,2]>=min_h)*(point_c[:,2]<=max_h))[0]
    tan_aov_x = intrinsic[0,2]/intrinsic[0,0]
    tan_aov_y = intrinsic[1,2]/intrinsic[1,1]
    x = point_c[judge_h,0] / point_c[judge_h,2]
    y = point_c[judge_h,1] / point_c[judge_h,2]
    judge= (x  >= -tan_aov_x) * (x <= tan_aov_x) * (y >= -tan_aov_y) * (y <= tan_aov_y)  
    judge_aov=np.where(judge)[0]
    return judge_h[judge_aov]
def get_visible_points(pointnormals,position,rotation,radius=300,min_h=0,max_h=10):
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(pointnormals[:,:3])
    point_cloud.normals=o3d.utility.Vector3dVector(pointnormals[:,3:])
    # point_cloud.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,size=(len(pointnormals), 3)))
    _, pt_map = point_cloud.hidden_point_removal(position, radius)
    #correspond
    point=pointnormals[pt_map,:3]
    judge=is_point_in_fov(point,rotation,position,min_h,max_h)
    return np.array(pt_map)[judge],pointnormals[np.array(pt_map)[judge],:3]
def get_visiblep_opt(pointnormals,position,rotation,radius,min_h=0,max_h=10):
    points=np.dot(rotation,(pointnormals[:,:3]-position).T).T
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(points)
    point_cloud.normals=o3d.utility.Vector3dVector(pointnormals[:,3:])
    # point_cloud.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,size=(len(pointnormals), 3)))
    _, pt_map = point_cloud.hidden_point_removal(np.ones(shape=[3])*1e-5, radius)
    # pcd_down=pcd.voxel_down_sample(voxel_size=0.001)
    # print(np.asarray(pcd.points).shape,np.asarray(pcd_down.points).shape)
    #correspond
    
    point=points[pt_map]
    
    judge_h=np.where((point[:,2]>=min_h)*(point[:,2]<=max_h))[0]
    tan_aov_x = intrinsic[0,2]/intrinsic[0,0]
    tan_aov_y = intrinsic[1,2]/intrinsic[1,1]
    x = point[judge_h,0] / point[judge_h,2]
    y = point[judge_h,1] / point[judge_h,2]
    judge= (x  > -tan_aov_x) * (x < tan_aov_x) * (y > -tan_aov_y) * (y < tan_aov_y)  
    judge_aov=np.where(judge)[0]
    judge=judge_h[judge_aov]
    return point[judge],np.asarray(pt_map)[judge]
def calculateCOCC(cameraposition,pointnormal):
    ray=cameraposition-pointnormal[:3]
    raymean=np.mean(ray,axis=0)
    co=1-np.sum(raymean * pointnormal[3:]) / (np.linalg.norm(raymean) * np.linalg.norm(pointnormal[3:])) if np.linalg.norm(raymean)!= 0 and np.linalg.norm(pointnormal[3:])!=0 else 1
    cc=0
    num=0
    for i in range(len(cameraposition)-1):
        for j in range(i+1,len(cameraposition)):
            angle=np.arccos(np.clip(np.sum(ray[i]*ray[j])/ (np.linalg.norm(ray[i]) * np.linalg.norm(ray[j])),0,1))
            cc+=(np.pi/2-angle)
            num+=1
    cc=cc*2/len(cameraposition)/(len(cameraposition)-1)
    return co,cc
def voxel_model(args,voxelnormals,rotation,position,min_h=0,max_h=1):
    #################################get voxels in how many cameras################### 

    #voxel center vis
    voxel_visibility = np.zeros([len(voxelnormals),len(position)])
    for i in range(len(position)):
        judge,_=get_visible_points(voxelnormals,position[i],rotation[i],200,min_h,max_h)
        voxel_visibility[judge,i]=1
    voxel_unvis=args.kcoverage-np.clip(np.sum(voxel_visibility,axis=1),0,args.kcoverage)
    
    #add voxel_need_vis attribute
    grid_return=np.append(voxelnormals,voxel_unvis[:,None],1)
    grid_return=np.append(grid_return,np.zeros([len(grid_return),2]),1)
    #voxel angle
    # angle_dif=np.zeros([len(voxelnormals)])
    # angle_sim=np.zeros([len(voxelnormals)])

    # for i in range(len(voxelnormals)):
    #     cameraindex=np.where(voxel_unvis[i]!=0)[0]
    #     if len(cameraindex) ==0:
    #         angle_sim[i],angle_dif[i]=1,np.pi/2
    #     elif len(cameraindex) ==1:
    #         angle_sim[i]=1-np.sum((position[cameraindex]-voxelnormals[i,:3]) * voxelnormals[i,3:]) / (np.linalg.norm(position[cameraindex]-voxelnormals[i,:3]) * np.linalg.norm(voxelnormals[i,3:])) if np.linalg.norm(position[cameraindex]-voxelnormals[i,:3])!= 0 and np.linalg.norm(voxelnormals[i,3:])!=0 else 1
    #         angle_dif[i]=np.pi/2
    #     elif len(cameraindex) >=2 :
    #         angle_sim[i],angle_dif[i] = calculateCOCC(position[cameraindex],voxelnormals[i])
    #         # print(angle_sim[i])
    #         # print(cameraindex,angle_dif[i])
    # grid_return=np.append(grid_return,angle_sim[:,None],1)
    # grid_return=np.append(grid_return,angle_dif[:,None],1)

    return npToTensor(grid_return),voxel_visibility