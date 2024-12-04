from dataset.utils import *
import itertools
# generate standard pointnormals matrix(n,6) from file
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    print(centroid,m)
    return pc,m,centroid
def getPointNormalfromFile(csvfile):
    pointnormals=np.loadtxt(csvfile,dtype=float,delimiter=',')
    return pointnormals
def getPointNormalfromPly(path,num,voxelsize,kcoverage):
    ending=path.split(".")[-1]
    if ending=="obj":
        mesh = o3d.io.read_triangle_mesh(path,True)
        mesh.compute_vertex_normals()
        pcd=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=num)
    elif ending=="ply":
        pcd = o3d.io.read_point_cloud(path)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)

    maxbound=pcd.get_max_bound()
    minbound=pcd.get_min_bound()
    center=pcd.get_center()
    size=maxbound-minbound
    #normalize
    points=np.asarray(pcd.points-center)/np.max(size)
    scale=[np.max(size),center]
    pcd.points=o3d.utility.Vector3dVector(points)
    if ending=='obj':
        mesh.vertices=o3d.utility.Vector3dVector((np.asarray(mesh.vertices)-center)/scale[0])
    #generatevoxelgrid
    voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxelsize)
    boundbox=np.asarray(voxelgrid.get_axis_aligned_bounding_box().get_box_points())
    minbound=np.min(boundbox,axis=0)
    maxbound=np.max(boundbox,axis=0)
    minvoxel=voxelgrid.get_voxel(minbound)
    maxvoxel=voxelgrid.get_voxel(maxbound)
    voxelnum=maxvoxel-minvoxel
    voxelnum+=1
    voxelsurface=np.zeros(shape=(voxelnum[0],voxelnum[1],voxelnum[2],1),dtype=int)
    point_index=np.floor((points-minbound)//voxelsize).astype(int)
    voxelsurface[point_index[:,0],point_index[:,1],point_index[:,2]]=kcoverage
    index=np.asarray(list(itertools.product(np.arange(voxelnum[0]),np.arange(voxelnum[1]),np.arange(voxelnum[2]))))
    judge_surface=np.where(voxelsurface[index[:,0],index[:,1],index[:,2]]!=0)[0]
    voxelindex=index[judge_surface]
    ###voxelposition
    voxelcenter_surface=(voxelindex+0.5)*voxelsize+minbound
    ###voxelnormal
    normals=np.asarray(pcd.normals)
    voxelnormals_surface = np.zeros_like(voxelindex,dtype=float)
    for i in range(len(voxelindex)):
        voxelnormals_surface[i]=np.mean(normals[np.where((point_index == voxelindex[i]).all(1))[0]],axis=0)

    pointnormals=np.append(points,np.asarray(pcd.normals),axis=1)
    voxelnormals = np.append(voxelcenter_surface,voxelnormals_surface,axis=1)
    pointnormals.astype(float)
    return mesh if ending=="obj" else pcd,pointnormals,voxelnormals,minbound,scale

