import numpy as np 
import open3d as o3d
import numpy as np
from dataset.init_camera import *
import matplotlib.pyplot as plt
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0,1,0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(lines) if lines is not None else self.lines_from_ordered_points(self.points)
        # print(self.lines,self.lines.shape)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            cylinder_segment.paint_uniform_color(self.colors[0,:])
            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def getCameraVis(camera_before,color,connection,R,C):
    T=np.dot(-R,C)
    R=R.T
    T=np.dot(-R,T)
    camera_rotate=np.dot(R,camera_before)
    camera_after=camera_rotate+T
    camera_points=camera_after.T
    tracking_mesh=LineMesh(camera_points,connection.astype(np.int32),color,radius=0.001)
    line_mesh_geo = tracking_mesh.cylinder_segments
    return line_mesh_geo

def visMesh(mesh,camerapose,camerarotation,flag=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=50)

    ##############control view ############

    camera_before=np.array([[0,0.2,-0.2,-0.2,0.2],
                            [0,0.1,0.1,-0.1,-0.1],
                            [0,0.2,0.2,0.2,0.2]])
    camera_before=camera_before*0.2
    camera_color=np.array( [[165/255 ,42/255 ,42/255],
                                [1,0,0],
                                [0,0,0],
                                [0,0,1],
                                [0,0,0]])
    point_connection=np.array([[0,1],
                                    [0,2],
                                    [0,3],
                                    [0,4],
                                    [1,2],
                                    [2,3],
                                    [3,4],
                                    [1,4]])

    line_mesh_all=[]
    for j in range(len(camerapose)):
            line_mesh_geo=getCameraVis(camera_before,camera_color,point_connection,camerarotation[j],camerapose[j].reshape(3,1))
            line_mesh_all.append(line_mesh_geo)
    for i in range(len(line_mesh_all)):
        for j in range(0,len(line_mesh_all[0])):
            vis.add_geometry(line_mesh_all[i][j])
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    if flag==1 :
        vis.run()
    vis.close()

def visFieldColor(pointnormals,coverage_num,name=None,scale=None):
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=pointnormals[:,:3]
    normal_array=pointnormals[:,3:]
    if scale != None:
        points_array = points_array*scale[0]+scale[1]
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    label=5
    print(np.max(coverage_num),np.min(coverage_num))
    colors=plt.get_cmap("coolwarm")(np.minimum(label,coverage_num)/(label if label > 0 else 1))
    point_cloud.colors=o3d.utility.Vector3dVector(colors[:,:3])
    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(name,point_cloud)

def visualization(pointnormals,camerapose,camerarotation,extrinsic=None,coverage_num=None,imagep=None,flag=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=50)
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=pointnormals[:,:3]
    normal_array=pointnormals[:,3:]
    ##############control view ############
    crt=vis.get_view_control()
    pinholecameracurrent=crt.convert_to_pinhole_camera_parameters()
    if len(camerapose)==1 and extrinsic is not None:
        pinholecamera=o3d.camera.PinholeCameraParameters()
        pinholecamera.extrinsic=extrinsic
        pinholecamera.intrinsic=pinholecameracurrent.intrinsic
    ########################################
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    # if coverage_num!= None:
    label=4
    colors=plt.get_cmap("plasma")(1-(np.minimum(label,coverage_num)/(label if label > 0 else 1)))
    point_cloud.colors=o3d.utility.Vector3dVector(colors[:,:3])
    # o3d.io.write_point_cloud(imagep,point_cloud)
    camera_before=np.array([[0,0.2,-0.2,-0.2,0.2],
                            [0,0.1,0.1,-0.1,-0.1],
                            [0,0.2,0.2,0.2,0.2]])
    camera_before=camera_before*0.2
    camera_color=np.array( [[0,0,0],
                                [0,0,1],
                                [0,0,1],
                                [0,0,1],
                                [165/255 ,42/255 ,42/255]])
    point_connection=np.array([[0,1],
                                    [0,2],
                                    [0,3],
                                    [0,4],
                                    [1,2],
                                    [2,3],
                                    [3,4],
                                    [1,4]])

    line_mesh_all=[]
    for j in range(len(camerapose)):
            line_mesh_geo=getCameraVis(camera_before,camera_color,point_connection,camerarotation[j],camerapose[j].reshape(3,1))
            line_mesh_all.append(line_mesh_geo)
    for i in range(len(line_mesh_all)):
        for j in range(0,len(line_mesh_all[0])):
            vis.add_geometry(line_mesh_all[i][j])
    
    vis.add_geometry(point_cloud)
    if len(camerapose)==1 and extrinsic is not None:
        crt.convert_from_pinhole_camera_parameters(pinholecamera,allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    if flag==1 :
        vis.run()
    if imagep !=None:
        vis.capture_screen_image(imagep+".png")
    vis.close()
def generateline(index,points,camerapose):
    camera_points=np.append(camerapose,points,axis=0)
    point_connection=np.zeros(shape=[0,2])
    for i in range(len(index[0])):
        point_connection=np.append(point_connection,np.array([[index[0][i]+len(camerapose),index[1][i]]]),axis=0)
    return camera_points,point_connection
def visaddline(pointnormals,camerapose,camerarotation,coverage_num,index,imagepath=None,flag=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=50)
    crt=vis.get_view_control()
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=pointnormals[:,:3]
    normal_array=pointnormals[:,3:]
    color_array=np.zeros_like(points_array)
    for i in range(len(coverage_num)):
        if coverage_num[i] == 0:
            color_array[i]=[255./255,0./255,0./255]
        elif coverage_num[i] == 1:
            color_array[i]=[255./255,255./255,0./255]
        elif coverage_num[i] == 2 :
            color_array[i]=[0./255,0./255,255./255]
        elif coverage_num[i] == 3 :
            color_array[i]=[0./255,255./255,0./255]
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    point_cloud.colors=o3d.utility.Vector3dVector(color_array)
    camera_before=np.array([[0,0.2,-0.2,-0.2,0.2],
                            [0,0.1,0.1,-0.1,-0.1],
                            [0,0.2,0.2,0.2,0.2]])
    camera_before=camera_before*0.1
    camera_color1=np.array( [[0,0,1],
                                [0,0,1],
                                [0,0,1],
                                [0,0,1],
                                [0,0,1]])
    camera_color2=np.array( [[0,0,0],
                                [0,0,0],
                                [0,0,0],
                                [0,0,0],
                                [0,0,0]])
    point_connection=np.array([[0,1],
                                    [0,2],
                                    [0,3],
                                    [0,4],
                                    [1,2],
                                    [2,3],
                                    [3,4],
                                    [1,4]])
    line_mesh_all=[]
    for j in range(len(camerapose)):
            line_mesh_geo=getCameraVis(camera_before,camera_color1,point_connection,camerarotation[j],camerapose[j].reshape(3,1))
            line_mesh_all.append(line_mesh_geo)
    for i in range(len(line_mesh_all)):
        for j in range(0,len(line_mesh_all[0])):
            vis.add_geometry(line_mesh_all[i][j])
    point,connection=generateline(index,pointnormals[:,:3],camerapose)
    connect_lineset=o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),lines=o3d.utility.Vector2iVector(connection))
    connect_lineset.colors = o3d.utility.Vector3dVector(camera_color2)
    vis.add_geometry(connect_lineset)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    if flag==1 :
        vis.run()
    if imagepath != None:
        vis.capture_screen_image(imagepath+".png")
    vis.close()
def visualPointselect(pointnormals):
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=np.append(pointnormals[:,:2],np.ones((len(pointnormals),1)),1)
    normal_array=np.zeros_like(points_array)
    normal_array[:,2]=1
    color_array=np.zeros_like(points_array)
    color_array[:,0]=255./255*pointnormals[:,2]
    max=np.argmax(pointnormals[:,2])
    color_array[max]=[0,1,0]
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    point_cloud.colors=o3d.utility.Vector3dVector(color_array)
    o3d.visualization.draw_geometries([point_cloud])

