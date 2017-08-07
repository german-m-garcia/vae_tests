import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

class ColorizeDepthMap:
    
    def __init__(self, length, path_output=''):
        self.length = length
        self.path_output = path_output
    
    def colorize(self, im_depth, noise, dist):
        raw = np.copy(im_depth)
        global size_x
        global size_y
        size_x = raw.shape[1]
        size_y = raw.shape[0]
        
        # averaged depth at the img center 
        #depth_center = 0
        #n = 0
        #n_center = 20
        #for i in range(n_center):
        #    for j in range(n_center):
        #        depth = im_depth[int(size_y/2+(i-n_center/2)), int(size_x/2+(j-n_center/2))]
        #        if(depth != 0):
        #            n += 1
        #            depth_center += depth
        #if n == 0:
        #    im_depth[int(size_y/2),int(size_x/2)] = 1.
        #    plt.imshow(im_depth)                
        #    plt.show()
        #depth_center /= n            
        
        # non-zero depth near to the center
        idx = np.array(np.where(raw>0))
        if(idx.shape[1] > 0):
            idx_min = np.argmin(np.linalg.norm(idx-size_x/2, axis=0))
            idx_min = idx[:,idx_min]
            depth_center = raw[idx_min[0], idx_min[1]]
        
            depth_min = depth_center - self.length;
            depth_max = (2*self.length)+depth_min
            if(noise):
                n_min = len(np.where(raw<depth_min)[0])
                n_max = len(np.where(raw>depth_max)[0])

                raw[np.where(raw<depth_min)] = np.random.uniform(depth_min, depth_max, n_min)            
                raw[np.where(raw>depth_max)] = np.random.uniform(depth_min, depth_max, n_max)           
            else:
                raw[np.where(raw<depth_min)] = depth_min
                raw[np.where(raw>depth_max)] = depth_max
            raw = raw - depth_min
            raw = (raw * 255./(depth_max-depth_min))
        
        raw = raw.astype(np.uint8)
        im_color = cv2.applyColorMap(raw, cv2.COLORMAP_JET)
        
        if(dist == True):  
            im_dist = np.zeros([size_y,size_x,3], dtype='uint8')

            # dists
            center_2d = [int(size_y/2), int(size_x/2)]
            center_depth = raw[center_2d[0],center_2d[1]]    
            dzdx = raw[center_2d[0]+1, center_2d[1]] - raw[center_2d[0]-1, center_2d[1]] / 2.
            dzdy = raw[center_2d[0], center_2d[1]+1] - raw[center_2d[0], center_2d[1]-1] / 2.    
            center_normal = [-dzdx, -dzdy, 1.]
            center_normal = center_normal/np.linalg.norm(center_normal)
            for i in range(1,size_y-1):
                for j in range(1,size_x-1):
                    cur_dist_depth = abs(raw[i,j]-center_depth)
                    if(cur_dist_depth>self.length/2): cur_dist_depth = self.length/2
                    cur_dist_2d = np.linalg.norm(np.array([i,j])-np.array(center_2d))

                    cur_dzdx = raw[i+1, j] - raw[i-1, j] / 2.
                    cur_dzdy = raw[i, j+1] - raw[i, j-1] / 2.

                    cur_normal = [-cur_dzdx, -cur_dzdy, 1.]
                    cur_normal = cur_normal/np.linalg.norm(cur_normal)
                    cur_dist_normal = np.linalg.norm(cur_normal - center_normal)
                    if(cur_dist_normal>0.02): cur_dist_normal = 0.02
                    
                    im_dist[i,j,0] = cur_dist_2d/np.linalg.norm([size_y/2,size_x/2])*255.
                    im_dist[i,j,1] = cur_dist_depth/(self.length/2)*255.
                    im_dist[i,j,2] = cur_dist_normal/0.02*255.
                    
            return [im_color, im_dist]
        else:
            return im_color    
    
    def colorizeMaps(self, depth_maps, noise, dist):
        n = depth_maps.shape[0]
        global color_maps
        global dist_maps
        color_maps = []
        dist_maps = []
        for i in range(n):
            if(dist == True):
                [im_color, im_dist] = self.colorize(depth_maps[i], noise, dist)               
            else:
                im_color = self.colorize(depth_maps[i], noise, dist)
            color_maps.append(im_color)
            if(dist==True): 
                dist_maps.append(im_dist)
            if(i%1000 == 0):
                print(str(i) + '/' + str(n))
        color_maps = np.reshape(color_maps,(-1,size_y,size_x,3))
        if(dist==True): 
            dist_maps = np.reshape(dist_maps,(-1,size_y,size_x,3))
            return [color_maps, dist_maps]
        else:
            return color_maps
    
    def save(self, name, saveImgs, dist):
        savePath = self.path_output + name + ".npz"
        if(dist == True):
            np.savez(savePath, color_maps = color_maps, dist_maps = dist_maps)
        else:
            np.savez(savePath, color_maps = color_maps)
        print("npz file is saved to %s" % (savePath))
        
        # save img
        if(saveImgs==True):
            print("Now saving img files...")
            directory = self.path_output+name+'_img/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(color_maps.shape[0]):
                color_map = color_maps[i]
                plt.imshow(color_map)                
                plt.savefig(directory+str(i)+'.png')  
                plt.clf()
        print("img files are saved to %s" % self.path_output+'/'+name+'/')
    
    def generate(self, name, depth_maps, saveImgs=False, noise=True, dist=True):
        self.colorizeMaps(depth_maps, noise, dist)
        self.save(name, saveImgs, dist)
    
    def saveImgs(self, npz, nFrom):
        directory = self.path_output+npz+'_img/'
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        npzPath = self.path_output + npz + ".npz"
        color_maps = np.load(npzPath)['color_maps']
        print("# of maps: ", color_maps.shape[0])
        
        for i in range(nFrom, color_maps.shape[0]):
            color_map = color_maps[i]
            plt.imshow(color_map)                
            plt.savefig(directory+str(i)+'.png')   
            plt.clf()
        print("img files are saved to %s" % self.path_output+'/'+name+'/')  
        

# averaged depth at the img center 
def normalizeDepth(im_depth, length=0.1, noise=False):    
    size_x = im_depth.shape[1]
    size_y = im_depth.shape[0]
    depth_center = 0
    n = 0
    for i in range(5):
        for j in range(5):
            depth = im_depth[int(size_y/2+(i-2)), int(size_x/2+(j-2))]
            if(depth != 0):
                n += 1
                depth_center += depth
    if n == 0:
        im_depth[int(size_y/2),int(size_x/2)] = 1.
        plt.imshow(im_depth)                
        plt.show()    
    depth_center /= n                 
    depth_min = depth_center - length;
    depth_max = (2*length)+depth_min
    for i in range(size_y):
        for j in range(size_x):                
            if(im_depth[i,j] < depth_min): 
                im_depth[i,j] = depth_min
            if(im_depth[i,j] > depth_max): 
                im_depth[i,j] = depth_max
            if(noise and (im_depth[i,j] == depth_min or im_depth[i,j] == depth_max)):
                im_depth[i,j] = np.random.uniform(depth_min, depth_max, 1)
    im_depth = im_depth - depth_min
    im_depth = (im_depth * size_x/(depth_max-depth_min))
    return im_depth
