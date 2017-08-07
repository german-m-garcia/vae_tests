import vtk
import numpy as np
from array import *
from matplotlib import pyplot as plt
import otn.transformations as tf
import os
import sys
import datetime

class RenderDepthMesh:
    
    def __init__(self, path_mesh, path_output='', 
                 param_tesselation_level=3, param_radius_sphere=1,
                 param_resolution_x=100, param_resolution_y=100,
                 param_view_angle=30, param_resolution_angle=100):
        self.path_mesh = path_mesh
        self.path_output = path_output
        self.param_tesselation_level = param_tesselation_level
        self.param_radius_sphere = param_radius_sphere
        self.param_resolution_x = param_resolution_x
        self.param_resolution_y = param_resolution_y
        self.param_view_angle = param_view_angle
        self.param_resolution_angle = param_resolution_angle
        print("init")
    
    def readMesh(self):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(self.path_mesh)
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(reader.GetOutput())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())
        mapper.Update()
        polydata = reader.GetOutput()
        
        # centre object
        com = [0]*3
        center = [0]*3
        cells = polydata.GetPolys()
        totalArea_com = 0
        cells.InitTraversal()
        for i in range(cells.GetNumberOfCells()):
            idx = vtk.vtkIdList()
            cells.GetNextCell(idx)    
            p1_com = [0]*3; p3_com = [0]*3; p2_com = [0]*3; 
            polydata.GetPoint(idx.GetId(0), p1_com)
            polydata.GetPoint(idx.GetId(1), p2_com)
            polydata.GetPoint(idx.GetId(2), p3_com)
            vtk.vtkTriangle.TriangleCenter(p1_com, p2_com, p3_com, center)
            area_com = vtk.vtkTriangle.TriangleArea(p1_com, p2_com, p3_com)
            com = [x + y*area_com for x, y in zip(com, center)]
            totalArea_com += area_com
        del cells
        com = [x / totalArea_com for x in com]
        global trans_filter_center 
        trans_filter_center = vtk.vtkTransformFilter()
        trans_center = vtk.vtkTransform()
        com = [-x for x in com]
        trans_center.Translate(com)
        trans_filter_center.SetTransform(trans_center)
        trans_filter_center.SetInputData(polydata)
        trans_filter_center.Update()
        
    def transformMesh(self,pose):  # px, py, pz, qw, qx, qy, qz
        global transformed_mesh
        transformed_mesh = vtk.vtkTransformFilter()
        transform = vtk.vtkTransform()
        mat = tf.quaternion_matrix([pose[3], pose[4], pose[5], pose[6]])
        mat[0:3,3] = pose[0:3]
        transform.SetMatrix((mat[0,0],mat[0,1],mat[0,2],mat[0,3],
                           mat[1,0],mat[1,1],mat[1,2],mat[1,3],
                           mat[2,0],mat[2,1],mat[2,2],mat[2,3],
                           mat[3,0],mat[3,1],mat[3,2],mat[3,3]))
        transformed_mesh.SetTransform(transform)
        transformed_mesh.SetInputData(trans_filter_center.GetOutput())
        transformed_mesh.Update()         
        return transformed_mesh.GetOutput()
                             
    def createCam(self):
        # create icosahedron for camera
        subdivide = vtk.vtkLoopSubdivisionFilter()
        ico = vtk.vtkPlatonicSolidSource()
        ico.SetSolidTypeToIcosahedron()
        ico.Update()

        # tesselate cells from icosahedron
        subdivide.SetNumberOfSubdivisions(self.param_tesselation_level)
        subdivide.SetInputConnection(ico.GetOutputPort())
        subdivide.Update()
        
        # camera using vertices
        sphere = subdivide.GetOutput()
        global N
        N = sphere.GetNumberOfPoints();
        global cam_positions
        cam_positions = np.empty((0,3))
        for i in range(N):
            cam_pos = [0]*3
            sphere.GetPoint(i, cam_pos)
            cam_positions = np.vstack((cam_positions, cam_pos))    
    
    def getTransformedMesh(self):
        return transformed_mesh
      
    def renderMaskTransformedMesh(self, pose):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformed_mesh.GetOutputPort());
        mapper.Update();        
        actor_view = vtk.vtkActor()
        actor_view.SetMapper(mapper)
        cam_pos = [0,0,0]
        cam = vtk.vtkCamera()
        cam.SetViewAngle(self.param_view_angle)
        
        test = np.cross(cam_pos, [0, -1, 0])  
        if np.dot(cam_pos, test) == 1:
            test = np.cross(cam_pos, [1, 0, 0]) 
        # print(test)
        cam.SetViewUp([0, 1, 0])
        cam.SetPosition([0,0,0])
        cam.SetFocalPoint(0,0,-0.35416024)
        cam.Modified()
        
        obj_tf = cam.GetViewTransformMatrix()        
        obj_mat = np.empty((0,4))
        for x in range(4):
            row = []
            for y in range(4):
                row.append(obj_tf.GetElement(x,y))
            obj_mat = np.vstack((obj_mat, row)) 
        #print(obj_mat)
        
        # create renderer and window
        render_win = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        render_win.AddRenderer(renderer)

        render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        render_win.SetOffScreenRendering(True)
        renderer.SetBackground(0,0,0)
        
        # create picker
        worldPicker = vtk.vtkWorldPointPicker()
        # render view
        renderer.SetActiveCamera(cam)
        renderer.AddActor(actor_view)
        renderer.Modified()
        render_win.Render()  
        z_buffer = vtk.vtkFloatArray()
        render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,z_buffer)  

        mask_map = np.zeros(shape=(self.param_resolution_y, self.param_resolution_x))
        for py in range(self.param_resolution_y):
            for px in range(self.param_resolution_x):
                pz = z_buffer.GetValue(py * self.param_resolution_x + px)
                if (pz == 1.):   mask_map[py,px] = 0
                else: mask_map[py,px]=1
                    
        return mask_map
    
    def renderTransformedMesh(self, pose):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformed_mesh.GetOutputPort());
        mapper.Update();        
        actor_view = vtk.vtkActor()
        actor_view.SetMapper(mapper)
        cam_pos = [0,0,0]
        cam = vtk.vtkCamera()
        cam.SetViewAngle(self.param_view_angle)
        
        test = np.cross(cam_pos, [0, -1, 0])  
        if np.dot(cam_pos, test) == 1:
            test = np.cross(cam_pos, [1, 0, 0]) 
        # print(test)
        cam.SetViewUp([0, 1, 0])
        cam.SetPosition([0,0,0])
        cam.SetFocalPoint(0,0,-0.35416024)
        cam.Modified()
        
        obj_tf = cam.GetViewTransformMatrix()        
        obj_mat = np.empty((0,4))
        for x in range(4):
            row = []
            for y in range(4):
                row.append(obj_tf.GetElement(x,y))
            obj_mat = np.vstack((obj_mat, row)) 
        #print(obj_mat)
        
        # create renderer and window
        render_win = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        render_win.AddRenderer(renderer)
        render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        render_win.SetOffScreenRendering(True)
        renderer.SetBackground(0,0,0)
        
        # create picker
        worldPicker = vtk.vtkWorldPointPicker()
        # render view
        renderer.SetActiveCamera(cam)
        renderer.AddActor(actor_view)
        renderer.Modified()
        render_win.Render()   
        z_buffer = vtk.vtkFloatArray()
        render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,z_buffer)  

        campos = cam_pos
        depth_map = np.zeros(shape=(self.param_resolution_y, self.param_resolution_x))
        for py in range(self.param_resolution_y):
            for px in range(self.param_resolution_x):
                pz = z_buffer.GetValue(py * self.param_resolution_x + px)
                if (pz == 1.):   depth = None
                else:
                    coords = vtk.vtkFloatArray()
                    worldPicker.Pick(px,py,pz,renderer)
                    coords = worldPicker.GetPickPosition()

                    depth = np.sqrt((coords[0]-campos[0])**2
                                    +(coords[1]-campos[1])**2
                                    +(coords[2]-campos[2])**2)                
                    depth_map[py,px] = depth       
        return depth_map
    
    def rendering_cam_orientation(self, j, n_view_up, n_translate, cam_pos):
        local_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))                          
        local_pose = np.empty((0,7))
        
        #print('.', end='', flush=True)
        view_up_ang = 2*np.pi/n_view_up*j
        view_up_x = np.cos(view_up_ang)
        view_up_y = np.sin(view_up_ang)
        view_up = [view_up_x, view_up_y, 0]

        v_n = [0,0,1]
        v_c = cam_pos/np.linalg.norm(cam_pos)
        v_v = np.cross(v_n,v_c)
        s = np.linalg.norm(v_v)
        c = np.dot(v_n,v_c)
        v_ss = np.array([[0, -v_v[2], v_v[1]], [v_v[2], 0, -v_v[0]], [-v_v[1], v_v[0], 0]])
        I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        R = I + v_ss + v_ss.dot(v_ss)*(1/1+c)
        view_up = R.dot(view_up)     
        
        cam.SetViewUp(view_up)
        cam.SetPosition(cam_pos*self.param_radius_sphere)
        cam.SetFocalPoint(0,0,0)
        cam.Modified()
        translated_renderer.SetActiveCamera(cam)
        
        # render view
        renderer.SetActiveCamera(cam)
        renderer.AddActor(actor_view)
        renderer.Modified()
        render_win.Render()   
        z_buffer = vtk.vtkFloatArray()
        render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,z_buffer)  

        obj_tf = cam.GetViewTransformMatrix()        
        obj_mat = np.empty((0,4))
        for x in range(4):
            row = []
            for y in range(4):
                row.append(obj_tf.GetElement(x,y))
            obj_mat = np.vstack((obj_mat, row))                                        
        
        campos = cam.GetPosition()        
        valid_map = np.empty((0,3))
        for py in range(self.param_resolution_y):
            for px in range(self.param_resolution_x):
                pz = z_buffer.GetValue(py * self.param_resolution_x + px)
                if (pz == 1.):   depth = None
                else:    
                    valid_map = np.vstack((valid_map,[px, py, pz]))
        
        randId = np.random.choice(valid_map.shape[0], n_translate, replace=False)
        translate_map = valid_map[randId, :]
        n_translation = translate_map.shape[0] 
 
        for k in range(n_translation):
            if k==0:
                px = int(self.param_resolution_x/2.)
                py = int(self.param_resolution_y/2.)
                pz = z_buffer.GetValue(py * self.param_resolution_x + px)
            else:
                [px, py, pz] = translate_map[k]
            translated_coords = vtk.vtkFloatArray()
            worldPicker.Pick(px,py,pz,renderer)
            translated_coords = worldPicker.GetPickPosition()

            # translate mesh                    
            translated_mesh = vtk.vtkTransformFilter()
            transform = vtk.vtkTransform()
            transform.SetMatrix((1,0,0,-translated_coords[0],
                                 0,1,0,-translated_coords[1],
                                 0,0,1,-translated_coords[2],
                                 0,0,0,1))
            translated_mesh.SetTransform(transform)
            translated_mesh.SetInputData(trans_filter_center.GetOutput())
            translated_mesh.Update()         

            # rendering                      
            translated_mapper.SetInputConnection(translated_mesh.GetOutputPort());
            translated_mapper.Update();        
            
            translated_renderer.Modified()
            translated_render_win.Render()  

            translated_z_buffer = vtk.vtkFloatArray()
            translated_render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,translated_z_buffer)  
            translated_depth_map = np.zeros(shape=(1,self.param_resolution_y, self.param_resolution_x))
            for py in range(self.param_resolution_y):
                for px in range(self.param_resolution_x):
                    pz = translated_z_buffer.GetValue(py * self.param_resolution_x + px)
                    if (pz == 1.):   depth = None
                    else:
                        coords = vtk.vtkFloatArray()
                        worldPicker.Pick(px,py,pz,translated_renderer)
                        coords = worldPicker.GetPickPosition()

                        depth = np.sqrt((coords[0]-campos[0])**2
                                        +(coords[1]-campos[1])**2
                                        +(coords[2]-campos[2])**2)                
                        translated_depth_map[0,py,px] = depth
            
            del translated_z_buffer
            translated_obj_mat = tf.concatenate_matrices(obj_mat, 
                                                         [[1, 0, 0, -translated_coords[0]],
                                                          [0, 1, 0, -translated_coords[1]],
                                                          [0, 0, 1, -translated_coords[2]],
                                                          [0,0,0,1]])
            #print(obj_mat, translated_obj_mat)

            translated_obj_q = tf.quaternion_from_matrix(translated_obj_mat)

            valid_map = translated_depth_map[translated_depth_map>0.]
            if(valid_map.size != 0):
                #print(local_depth_map.shape)
                #print(translated_depth_map.shape)
                local_depth_map = np.vstack((local_depth_map,translated_depth_map))
                local_pose = np.vstack((local_pose,[translated_obj_mat[0,3], translated_obj_mat[1,3], translated_obj_mat[2,3], 
                                   translated_obj_q[0], translated_obj_q[1], translated_obj_q[2], translated_obj_q[3]]))           
  #      local_depth_map = np.reshape(local_depth_map,(-1,self.param_resolution_y,self.param_resolution_x)) 
  #      local_pose = np.reshape(local_pose,(-1,7))
        return [local_depth_map, local_pose]
    
    def rendering_cam(self, i, n_view_up, n_translate):
        cam_pos = cam_positions[i]
        # create temporal virtual camera
        total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.empty((0,7))
        for j in range(n_view_up): 
            [local_depth_map, local_pose] = self.rendering_cam_orientation(j, n_view_up, n_translate, cam_pos)
            total_depth_map = np.vstack((total_depth_map,local_depth_map))
            total_pose = np.vstack((total_pose,local_pose))
        return [total_depth_map, total_pose]
            
    def rendering_translate_multiprocess(self, name, saveImgs):
        print("readmesh")
        self.readMesh()
        print("createCam")
        self.createCam()
        # create renderer and window
        global render_win
        global renderer
        render_win = vtk.vtkRenderWindow()
        render_win.SetOffScreenRendering(True)
        renderer = vtk.vtkRenderer()
        render_win.AddRenderer(renderer)
        render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        renderer.SetBackground(0,0,0)

        global translated_render_win
        global translated_renderer
        global translated_mapper        
        translated_render_win = vtk.vtkRenderWindow()
        translated_render_win.SetOffScreenRendering(True)
        translated_renderer = vtk.vtkRenderer()
        translated_render_win.AddRenderer(translated_renderer)
        translated_render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        translated_renderer.SetBackground(0,0,0)   
        
        translated_mapper = vtk.vtkPolyDataMapper()
        
        global translated_actor_view
        translated_actor_view = vtk.vtkActor()
        translated_renderer.AddActor(translated_actor_view)
        translated_actor_view.SetMapper(translated_mapper)  
        
        # create picker
        global worldPicker
        worldPicker = vtk.vtkWorldPointPicker()
        
        global cam
        cam = vtk.vtkCamera()
        cam.SetViewAngle(self.param_view_angle)
        
        global total_depth_map
        global total_pose
        total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.empty((0,7))
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(trans_filter_center.GetOutputPort());
        mapper.Update();     
        global actor_view
        actor_view = vtk.vtkActor()
        actor_view.SetMapper(mapper)
        
        # for each camera position, transform the object and render view        
        n_view_up = 8*self.param_tesselation_level        
        n_translate = 8*self.param_tesselation_level    
        print(N)
        # for test
        #N = 2
        n_view_up = 1
        n_translate = 1
        
        t_start = datetime.datetime.now().replace(microsecond=0)
        for i in range(N):
            t_now = datetime.datetime.now().replace(microsecond=0)
            if i==0: print(t_now-t_start,flush=True, end='')
            if i > 0:                
                elapse_total = N
                elapse_cur = i
                t_remain = (t_now-t_start)/(elapse_cur)*(elapse_total)-(t_now-t_start)            
                print(t_remain,flush=True)
                print(' '+str(i)+'/'+str(N)+' cam ',flush=True)
                print(t_now-t_start,flush=True)
                
            [local_depth_map, local_pose] = self.rendering_cam(i, n_view_up, n_translate) 
            total_depth_map = np.vstack((total_depth_map,local_depth_map))
            total_pose = np.vstack((total_pose,local_pose))
       #     if i>0:
       #         if i%5 == 0:                                        
       #             self.save(name+str(i), saveImgs)
       #             total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
       #             total_pose = np.empty((0,7))
       #         if i == N-1:
       #             self.save(name+str(i), saveImgs)
       #             total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
       #             total_pose = np.empty((0,7))
        self.save(name, saveImgs)
        total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.empty((0,7))
        
    def rendering_translate_multiprocess_num(self, name, nums, saveImgs):
        print("readmesh")
        self.readMesh()
        print("createCam")
        self.createCam()
        # create renderer and window
        global render_win
        global renderer
        render_win = vtk.vtkRenderWindow()
        render_win.SetOffScreenRendering(True)
        renderer = vtk.vtkRenderer()
        render_win.AddRenderer(renderer)
        render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        renderer.SetBackground(0,0,0)

        global translated_render_win
        global translated_renderer
        global translated_mapper        
        translated_render_win = vtk.vtkRenderWindow()
        translated_render_win.SetOffScreenRendering(True)
        translated_renderer = vtk.vtkRenderer()
        translated_render_win.AddRenderer(translated_renderer)
        translated_render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        translated_renderer.SetBackground(0,0,0)   
        
        translated_mapper = vtk.vtkPolyDataMapper()
        
        global translated_actor_view
        translated_actor_view = vtk.vtkActor()
        translated_renderer.AddActor(translated_actor_view)
        translated_actor_view.SetMapper(translated_mapper)  
        
        # create picker
        global worldPicker
        worldPicker = vtk.vtkWorldPointPicker()
        
        global cam
        cam = vtk.vtkCamera()
        cam.SetViewAngle(self.param_view_angle)
        
        global total_depth_map
        global total_pose
        total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.empty((0,7))
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(trans_filter_center.GetOutputPort());
        mapper.Update();     
        global actor_view
        actor_view = vtk.vtkActor()
        actor_view.SetMapper(mapper)
        
        # for each camera position, transform the object and render view        
        n_view_up = 8*self.param_tesselation_level        
        n_translate = 8*self.param_tesselation_level    
        
        # for test
        #N = 2
        
        t_start = datetime.datetime.now().replace(microsecond=0)
        N = len(nums)
        print(N)
        
        for i in range(N):
            t_now = datetime.datetime.now().replace(microsecond=0)
            if i==0: print(t_now-t_start,flush=True, end='')
            if i > 0:                
                elapse_total = N
                elapse_cur = i
                t_remain = (t_now-t_start)/(elapse_cur)*(elapse_total)-(t_now-t_start)            
                print(t_remain,flush=True)
                print(' '+str(i)+'/'+str(N)+' cam ',flush=True)
                print(t_now-t_start,flush=True)
                
            [local_depth_map, local_pose] = self.rendering_cam(nums[i], n_view_up, n_translate) 
            total_depth_map = np.vstack((total_depth_map,local_depth_map))
            total_pose = np.vstack((total_pose,local_pose))
       #     if i>0:
       #         if i%5 == 0:                                        
       #             self.save(name+str(i), saveImgs)
       #             total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
       #             total_pose = np.empty((0,7))
       #         if i == N-1:
       #             self.save(name+str(i), saveImgs)
       #             total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
       #             total_pose = np.empty((0,7))
        self.save(name, saveImgs)
        total_depth_map = np.empty((0,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.empty((0,7))
        
    def rendering_translate(self):
        self.readMesh()
        self.createCam()
        # create renderer and window
        render_win = vtk.vtkRenderWindow()
        render_win.SetOffScreenRendering(True)
        renderer = vtk.vtkRenderer()
        render_win.AddRenderer(renderer)
        render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        renderer.SetBackground(0,0,0)

        translated_render_win = vtk.vtkRenderWindow()
        translated_render_win.SetOffScreenRendering(True)
        translated_renderer = vtk.vtkRenderer()
        translated_render_win.AddRenderer(translated_renderer)
        translated_render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        translated_renderer.SetBackground(0,0,0)            
        translated_mapper = vtk.vtkPolyDataMapper()
        
        # create picker
        worldPicker = vtk.vtkWorldPointPicker()
        global total_depth_map
        global total_pose
        total_depth_map = []
        total_pose = []
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(trans_filter_center.GetOutputPort());
        mapper.Update();        
        actor_view = vtk.vtkActor()
        actor_view.SetMapper(mapper)
        
        # for each camera position, transform the object and render view        
        n_view_up = 8*self.param_tesselation_level        
        n_translate = 8*self.param_tesselation_level        
        n=0
        # for test
        #N = 1
        #n_view_up = 1
        t_start = datetime.datetime.now().replace(microsecond=0)
        for i in range(N):
            t_now = datetime.datetime.now().replace(microsecond=0)
            #if i==0: print(t_now-t_start,flush=True, end='')
            if i > 0:                
                elapse_total = N
                elapse_cur = i
                t_remain = (t_now-t_start)/(elapse_cur)*(elapse_total)-(t_now-t_start)            
                #print(t_remain,flush=True)
                #print(' '+str(i)+'/'+str(N)+' cam ',flush=True)
                #print(t_now-t_start,flush=True, end='')
            cam_pos = cam_positions[i]
            # create temporal virtual camera
            cam = vtk.vtkCamera()
            cam.SetViewAngle(self.param_view_angle)

            for j in range(n_view_up): 
                #print('.', end='', flush=True)
                view_up_ang = 2*np.pi/n_view_up*j
                view_up_x = np.cos(view_up_ang)
                view_up_y = np.sin(view_up_ang)
                view_up = [view_up_x, view_up_y, 0]
                
                v_n = [0,0,1]
                v_c = cam_pos/np.linalg.norm(cam_pos)
                v_v = np.cross(v_n,v_c)
                s = np.linalg.norm(v_v)
                c = np.dot(v_n,v_c)
                v_ss = np.array([[0, -v_v[2], v_v[1]], [v_v[2], 0, -v_v[0]], [-v_v[1], v_v[0], 0]])
                I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                R = I + v_ss + v_ss.dot(v_ss)*(1/1+c)
                view_up = R.dot(view_up)     
                cam.SetViewUp(view_up)
                cam.SetPosition(cam_pos*self.param_radius_sphere)
                cam.SetFocalPoint(0,0,0)
                cam.Modified()

                # render view
                renderer.SetActiveCamera(cam)
                renderer.AddActor(actor_view)
                renderer.Modified()
                render_win.Render()   
                z_buffer = vtk.vtkFloatArray()
                render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,z_buffer)  
                
                obj_tf = cam.GetViewTransformMatrix()
                obj_mat = np.empty((0,4))
                for x in range(4):
                    row = []
                    for y in range(4):
                        row.append(obj_tf.GetElement(x,y))
                    obj_mat = np.vstack((obj_mat, row))                                        

                campos = cam.GetPosition()
                #depth_map = np.zeros(shape=(self.param_resolution_y, self.param_resolution_x))
                valid_map = []
                for py in range(self.param_resolution_y):
                    for px in range(self.param_resolution_x):
                        pz = z_buffer.GetValue(py * self.param_resolution_x + px)
                        if (pz == 1.):   depth = None
                        else:    
                            valid_map.append([px, py, pz])                                                
                            
                valid_map = np.reshape(valid_map, [-1,3])
                randId = np.random.choice(valid_map.shape[0], n_translate, replace=False)
                translate_map = valid_map[randId, :]
                               
                n_translation = translate_map.shape[0] 
                #n_translation = 1
                for k in range(n_translation):
                    [px, py, pz] = translate_map[k]
                    translated_coords = vtk.vtkFloatArray()
                    worldPicker.Pick(px,py,pz,renderer)
                    translated_coords = worldPicker.GetPickPosition()
                    
                    # translate mesh                    
                    translated_mesh = vtk.vtkTransformFilter()
                    transform = vtk.vtkTransform()
                    transform.SetMatrix((1,0,0,-translated_coords[0],
                                       0,1,0,-translated_coords[1],
                                       0,0,1,-translated_coords[2],
                                       0,0,0,1))
                    translated_mesh.SetTransform(transform)
                    translated_mesh.SetInputData(trans_filter_center.GetOutput())
                    translated_mesh.Update()         
                    
                    # rendering                      
                    translated_mapper.SetInputConnection(
                                translated_mesh.GetOutputPort());
                    translated_mapper.Update();        
                    translated_actor_view = vtk.vtkActor()
                    translated_actor_view.SetMapper(translated_mapper)
                    translated_renderer.SetActiveCamera(cam)
                    translated_renderer.AddActor(translated_actor_view)
                    translated_renderer.Modified()
                    translated_render_win.Render()  
                    
                    translated_z_buffer = vtk.vtkFloatArray()
                    translated_render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,translated_z_buffer)  
                    translated_depth_map = np.zeros(shape=(self.param_resolution_y, self.param_resolution_x))
                    for py in range(self.param_resolution_y):
                        for px in range(self.param_resolution_x):
                            pz = translated_z_buffer.GetValue(py * self.param_resolution_x + px)
                            if (pz == 1.):   depth = None
                            else:
                                coords = vtk.vtkFloatArray()
                                worldPicker.Pick(px,py,pz,translated_renderer)
                                coords = worldPicker.GetPickPosition()

                                depth = np.sqrt((coords[0]-campos[0])**2
                                                +(coords[1]-campos[1])**2
                                                +(coords[2]-campos[2])**2)                
                                translated_depth_map[py,px] = depth
                    translated_obj_mat = tf.concatenate_matrices(obj_mat, 
                                                                 [[1, 0, 0, -translated_coords[0]],
                                                                  [0, 1, 0, -translated_coords[1]],
                                                                  [0, 0, 1, -translated_coords[2]],
                                                                  [0,0,0,1]])
                    
                    translated_obj_q = tf.quaternion_from_matrix(translated_obj_mat)
                
                    valid_map = translated_depth_map[translated_depth_map>0.]
                    if(valid_map.size != 0):
                        total_depth_map.append(translated_depth_map)            
                        total_pose.append([translated_obj_mat[0,3], translated_obj_mat[1,3], translated_obj_mat[2,3], 
                                        translated_obj_q[0], translated_obj_q[1], translated_obj_q[2], translated_obj_q[3]])           
                        n = n+1
                #print('  memsize = '+str(sys.getsizeof(total_depth_map)))
                        
        total_depth_map = np.reshape(total_depth_map,(-1,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.reshape(total_pose,(-1,7))        
    
    def rendering(self):
        # create renderer and window
        render_win = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        render_win.AddRenderer(renderer)
        render_win.SetSize(self.param_resolution_x, self.param_resolution_y)
        renderer.SetBackground(0,0,0)

        # create picker
        worldPicker = vtk.vtkWorldPointPicker()
        global total_depth_map
        global total_pose
        total_depth_map = []
        total_pose = []
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(trans_filter_center.GetOutputPort());
        mapper.Update();        
        actor_view = vtk.vtkActor()
        actor_view.SetMapper(mapper)
        
        # for each camera position, transform the object and render view        
        #n_view_up = 6*self.param_tesselation_level
        #n=0
        for i in range(N):
            cam_pos = cam_positions[i]
            # create temporal virtual camera
            cam = vtk.vtkCamera()
            cam.SetViewAngle(self.param_view_angle)

            # NEED TO BE UPDATED TO PRODUCE VARIOUS SETVIEWUP
            #test = np.cross(cam_pos, [0, 1, 0])  
            #if np.dot(cam_pos, test) == 1:
            #    test = np.cross(cam_pos, [1, 0, 0]) 
            # print(test)
            for j in range(n_view_up):
                view_up_ang = 2*np.pi/n_view_up*j
                view_up_x = np.cos(view_up_ang)
                view_up_y = np.sin(view_up_ang)
                view_up = [view_up_x, view_up_y, 0]
                
                v_n = [0,0,1]
                v_c = cam_pos/np.linalg.norm(cam_pos)
                v_v = np.cross(v_n,v_c)
                s = np.linalg.norm(v_v)
                c = np.dot(v_n,v_c)
                v_ss = np.array([[0, -v_v[2], v_v[1]], [v_v[2], 0, -v_v[0]], [-v_v[1], v_v[0], 0]])
                I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                R = I + v_ss + v_ss.dot(v_ss)*(1/1+c)
                view_up = R.dot(view_up)     
                cam.SetViewUp(view_up)
                cam.SetPosition(cam_pos*self.param_radius_sphere)
                cam.SetFocalPoint(0,0,0)
                #cam.OrthogonalizeViewUp()
                cam.Modified()

                # render view
                renderer.SetActiveCamera(cam)
                renderer.AddActor(actor_view)
                renderer.Modified()
                render_win.Render()   
                z_buffer = vtk.vtkFloatArray()
                render_win.GetZbufferData(0,0,self.param_resolution_x-1,self.param_resolution_y-1,z_buffer)  

                campos = cam.GetPosition()
                depth_map = np.zeros(shape=(self.param_resolution_y, self.param_resolution_x))
                for py in range(self.param_resolution_y):
                    for px in range(self.param_resolution_x):
                        pz = z_buffer.GetValue(py * self.param_resolution_x + px)
                        if (pz == 1.):   depth = None
                        else:
                            coords = vtk.vtkFloatArray()
                            worldPicker.Pick(px,py,pz,renderer)
                            coords = worldPicker.GetPickPosition()

                            depth = np.sqrt((coords[0]-campos[0])**2
                                            +(coords[1]-campos[1])**2
                                            +(coords[2]-campos[2])**2)                
                            depth_map[py,px] = depth
                
                #cam_p = cam.GetPosition()
                #cam_q = cam.GetOrientationWXYZ()
                #print(cam_q)
                #cam_mat=tf.quaternion_matrix([cam_q[1], cam_q[2], cam_q[3], cam_q[0]])
                #cam_mat[0:3,3] = cam_p[0:3]

                obj_tf = cam.GetViewTransformMatrix()
                obj_mat = np.empty((0,4))
                for x in range(4):
                    row = []
                    for y in range(4):
                        row.append(obj_tf.GetElement(x,y))
                    obj_mat = np.vstack((obj_mat, row))                        
                #cflip = np.array([[1, 0, 0, 0],[0, -1, 0, 0 ],[0, 0, -1, 0 ],[0, 0, 0, 1]])
                #obj_mat = tf.concatenate_matrices(cflip, obj_mat)
                #obj_mat = np.linalg.inv(cam_mat)
                obj_q = tf.quaternion_from_matrix(obj_mat)
                #obj_p = ([obj_mat[0,3], obj_mat[1,3], obj_mat[2,3])
                #print(obj_mat)
                #print(obj_q)    
                valid_map = depth_map[depth_map>0.]
                if(valid_map.size != 0):
                    total_depth_map.append(depth_map)            
                    total_pose.append([obj_mat[0,3], obj_mat[1,3], obj_mat[2,3], 
                                    obj_q[0], obj_q[1], obj_q[2], obj_q[3]])           
                    n = n+1        
        total_depth_map = np.reshape(total_depth_map,(-1,self.param_resolution_y,self.param_resolution_x))
        total_pose = np.reshape(total_pose,(-1,7))
                                     
    def save(self, name, saveImgs):
        print(total_depth_map.shape)
        print(total_pose.shape)        
        savePath = self.path_output + name + ".npz"
        np.savez(savePath, depth_map = total_depth_map, obj_pose = total_pose)
        print("npz file is saved to %s" % (savePath))
        
        # save img
        if(saveImgs==True):
            print("Now saving img files...")
            directory = self.path_output+name+'_img/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(total_depth_map.shape[0]):
                depth_map = total_depth_map[i]
                plt.imshow(depth_map, cmap='Greys_r')                
                plt.savefig(directory+str(i)+'.png')     
                plt.clf()
        print("img files are saved to %s" % directory)
        
    def saveImgs(self, npz, nFrom):
        directory = self.path_output+npz+'_img/'
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        npzPath = self.path_output + npz + ".npz"
        depth_map = np.load(npzPath)['depth_map']
        print("# of maps: ", depth_map.shape[0])
        
        for i in range(nFrom, depth_map.shape[0]):
            plt.imshow(depth_map[i], cmap='Greys_r', antialiased=True)
            plt.savefig(directory+str(i)+'.png')   
            plt.clf()
        print("img files are saved to %s" % self.path_output+'/'+name+'/')  
        
    def generate(self, name, saveImgs=False):
        #self.readMesh()
        #self.createCam()
        
        #self.rendering() 
        self.rendering_translate()
        #self.rendering_translate_multiprocess(name, saveImgs)
        #self.save(name, saveImgs)
    def generate_num(self, name, nums, saveImgs=False):
        self.rendering_translate_multiprocess_num(name, nums, saveImgs)
        
