#Code for by-hand granules labelling of Sunrise/IMaX maps
#Author: Saida Diaz - KIS

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import warnings
from pylab import rcParams
import math
from scipy.io import readsav
import pandas as pd
pd.set_option('display.max_rows', 500)
import sys
import cv2
import scipy.ndimage as ndi 
from matplotlib.widgets import Button
from matplotlib.widgets import Cursor

import random 
from itertools import combinations as comb

from itertools import groupby
from operator import itemgetter

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def save_byt(m_grey, name:str):
    byte_test=bytearray(m_grey)
    f = open(name+".BYT", "wb")
    f.write(byte_test)
    f.close()

def var_lap(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

class manual_edition():
    def __init__(self, data_mlt4, data_o, frame):
        self.data_mlt4 = data_mlt4
        self.data_o = data_o
        self.frame = frame

        plt.rcParams.update({'font.size': 8})
        self.new_cmap = rand_cmap(800)
        
        self.fig, self.ax = plt.subplots(1,2, figsize=(10, 6), sharex=True, sharey=True)
        im1=self.ax[0].imshow(self.data_o, origin = 'lower', cmap='gray')
        im2=self.ax[1].imshow(self.data_mlt4, origin = 'lower', cmap=self.new_cmap)
        self.fig.suptitle('Close figure for save')
        self.partial_image = self.data_mlt4.copy()
        
        self.fig.tight_layout()

        self.cursor1 = Cursor(self.ax[0],
           horizOn = True,
           vertOn = True,
           color = 'red',
           linewidth = '1.0')
        
        self.ll = []
        self.partial_save = {}
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.select)
        self.cid1 = self.fig.canvas.mpl_connect("button_press_event",self.draw)
        self.cid2 = self.fig.canvas.mpl_connect('close_event', self.on_close)

    def select(self, event):
        x1, y1 = event.xdata, event.ydata
        global label
        label=self.data_mlt4[int(y1)][int(x1)]
        print(label)
            
    def draw(self, event):
        print(label)
        x, y = event.xdata, event.ydata
        self.ax[1].clear()
        for i in np.linspace(-2,2,5):
            for j in np.linspace(-2,2,5):
                self.partial_image[int(y)+int(i)][int(x)+int(j)] = label
        im2=self.ax[1].imshow(self.partial_image, origin = 'lower', cmap=self.new_cmap)
        self.ax[1].set_title('White star will make your cell selection')

    def on_close(self, event, save=False):
        print('Save/Close')
        mod_map = self.partial_image
        if save == True:
            np.savez(f'Modified_data_Frame_{self.frame}.npz', mod_map=mod_map)
        self.fig.canvas.mpl_disconnect(self.cid)
        return mod_map

class cell_labelling():
    def __init__(self, data_mlt4, data_o, Type):
        self.data_mlt4 = data_mlt4
        self.data_o = data_o
        self.type = Type
        self.new_cmap = rand_cmap(800)

        plt.rcParams.update({'font.size': 8})
        
        self.fig, self.ax = plt.subplots(1,2, figsize=(10, 6), sharex=True, sharey=True)
        im1=self.ax[0].imshow(self.data_o, origin = 'lower', cmap='gray')
        im2=self.ax[1].imshow(self.data_mlt4, origin = 'lower', cmap=self.new_cmap)
        self.ax[0].set_title('Click here for select/unselect cells')
        self.ax[1].set_title('White star will mark your cell selection')
        self.fig.suptitle('Close figure for save')
        
        self.fig.tight_layout()
        
        from matplotlib.widgets import Cursor
        self.cursor1 = Cursor(self.ax[0],
               horizOn = True,
               vertOn = True,
               color = 'red',
               linewidth = '1.0')
        
        self.ll = []
        self.partial_save = {}
        self.cid = self.fig.canvas.mpl_connect("button_press_event",self.pointing)
        self.cid2 = self.fig.canvas.mpl_connect('close_event', self.on_close)
        
    def pointing(self, event):
        global x1, y1
        x1, y1 = event.xdata, event.ydata
        global label
        label=self.data_mlt4[int(y1)][int(x1)]
        self.partial_save[label]=(x1,y1)
        if label in self.ll:
            self.ax[1].clear()
            im2=self.ax[1].imshow(self.data_mlt4, origin = 'lower', cmap=self.new_cmap)
            self.ax[1].set_title('White star will make your cell selection')
            del self.partial_save[label]
            if len(self.partial_save) != 0:
                pos_list=[self.partial_save[x] for x in self.partial_save.keys()]
                for pos in pos_list:
                    self.ax[1].scatter(pos[0],pos[1],marker='*',color='w')
            self.ll.remove(label)
        else:
            self.ax[1].scatter(x1,y1,marker='*',color='w')
            self.ll.append(label)

    def on_close(self, event):
        print('Save/Close')
        self.fig.canvas.mpl_disconnect(self.cid)
        
def df_label(ftag, data_original):
    label_list_mlt4=np.unique(ftag)
    df_tags=pd.DataFrame(index=label_list_mlt4[1:],columns=['Area[pix]', 
                                                        'X_size[pix]', 
                                                        'Y_size[pix]', 
                                                        'Mean_I', 
                                                        'X_pos', 
                                                        'Y_pos', 
                                                        'Label_features'])
    for l in label_list_mlt4[1:]:
        #Area
        num_pix=np.count_nonzero(ftag == l)
        #XY size
        x_array=np.count_nonzero(ftag == l, axis=1)
        y_array=np.count_nonzero(ftag == l, axis=0)
        x_s=np.count_nonzero(x_array)
        y_s=np.count_nonzero(y_array)
        #Mean
        mean_I_array=data_original[np.nonzero(ftag == l)]
        mean_I=mean_I_array.mean()
        #XY pos
        map_boll=data_original*(ftag == l).astype(int)
        cy, cx = ndi.center_of_mass(map_boll)
        df_tags.loc[l]=[num_pix,x_s,y_s, np.round(mean_I,2),int(cx),int(cy),0]
    return df_tags

class IMaX_maps():
    def __init__(self, file):
        self.file = file
        raw_data=readsav(self.file, python_dict=True)
        self.raw_data=raw_data['contmaps']
        self.maps = self.raw_data[:,70:raw_data['ny']-70,70:raw_data['ny']-70]/self.raw_data.mean()
        self.shape = self.maps.shape
        self.maps_grayscale = cv2.normalize(src=self.maps, dst=None, 
                                        alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                        dtype=cv2.CV_8UC1)
    def select_map(self, frame):
        self.frame = frame
        self.smap=self.maps_grayscale[frame][1:-1,1:-1]
        
    def open_MLT4_mask(self, mask_file):
        if str(self.frame) in mask_file:
            mask_save=readsav(mask_file, python_dict=True)
        else:
            raise Error('Mask file do not corresponds to selected frame:' +str(frame))
                        
        self.MLT4mask_map =mask_save['ftag']
        
        print(f'Mask_shape: {self.MLT4mask_map.shape}')
        print(f'Selected_map_shape: {self.smap.shape}')
                        
    def summary_cells(self):
        self.df_t=df_label(self.MLT4mask_map, self.smap)
                        
    def create_final_mask(self, l_DG, l_LG, l_AG, mask_map, plotting=True, save=False):
        self.l_DG = l_DG
        self.l_LG = l_LG
        self.l_AG = l_AG
        self.mask_map = mask_map

        self.df_t_extra=df_label(self.mask_map, self.smap)
        
        self.df_t_extra.loc[self.l_DG, 'Label_features'] = 1
        self.df_t_extra.loc[self.l_LG, 'Label_features'] = 2
        self.df_t_extra.loc[self.l_AG, 'Label_features'] = 3
        list_NG=self.df_t_extra[self.df_t_extra['Label_features'] == 0].index.values
        self.df_t_extra.loc[list_NG, 'Label_features'] = 4

        f_ll=np.zeros(self.mask_map.shape, dtype=np.int32)
        f_ll_bin=np.zeros((self.mask_map.shape[0],self.mask_map.shape[1],5), dtype=np.int32)
        for i in range(self.mask_map.shape[0]):
            for j in range(self.mask_map.shape[1]):
                l_value=self.mask_map[i,j]
                if l_value != 0:
                    fl_value=self.df_t_extra.loc[l_value, 'Label_features']
                    f_ll[i,j] = fl_value
                    if fl_value == 1:
                        f_ll_bin[i,j] = np.array([0,1,0,0,0])
                    elif fl_value == 2:
                        f_ll_bin[i,j] = np.array([0,0,1,0,0])
                    elif fl_value == 3:
                        f_ll_bin[i,j] = np.array([0,0,0,1,0])
                    elif fl_value == 4:
                        f_ll_bin[i,j] = np.array([0,0,0,0,1]) 
                else:
                    f_ll_bin[i,j] = np.array([1,0,0,0,0])
            
        self.c_smap=self.smap[13:self.smap.shape[0]-13,13:self.smap.shape[0]-13]
        self.c_mask_map=f_ll[13:f_ll.shape[0]-13,13:f_ll.shape[0]-13]
        self.c_mask_map_bin=f_ll_bin[13:f_ll.shape[0]-13,13:f_ll.shape[0]-13,:]
                        
        if plotting == True:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
            im1=ax[0].imshow(self.c_smap, origin='lower', cmap='gray')
            im2=ax[1].imshow(self.c_mask_map, origin='lower', cmap = plt.get_cmap('PiYG', 5))
            fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=ax[1], ticks=np.arange(0,5), fraction=0.046, pad=0.04)
            ax[0].set_title('Original Image')
            ax[1].set_title('Label Image')
            
        if save == True:
            smap = self.c_smap
            cmask_map = self.c_mask_map
            cmask_map_bin = self.c_mask_map_bin
            np.savez(f'Mask_data_Frame_{self.frame}.npz', 
                     smap=smap, cmask_map=cmask_map, cmask_map_bin=cmask_map_bin)