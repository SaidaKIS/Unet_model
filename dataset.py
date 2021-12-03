import numpy as np
import random
import torchvision.transforms as Ttorch
import torch
from glob import glob
import cv2
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_param(degree, size):
    """
    Generate random angle for rotation and define the extension box for define their
    center
    """
    angle = float(torch.empty(1).uniform_(float(degree[0]), float(degree[1])).item())
    extent = int(np.ceil(np.abs(size*np.cos(np.deg2rad(angle)))+np.abs(size*np.sin(np.deg2rad(angle))))/2)
    return angle, extent

def subimage(image, center, theta, width, height):
   """
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   """
   shape = (image.shape[1], image.shape[0]) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
   image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

   x = int(center[0] - width/2)
   y = int(center[1] - height/2)

   image = image[y:y+height, x:x+width]
   return image

class RandomRotation_crop(torch.nn.Module):
  def __init__(self, degrees, size):
       super().__init__()
       self.degree = [float(d) for d in degrees]
       self.size = int(size)

  def forward(self, img):
      """Rotate the image by a random angle.
         If the image is torch Tensor, it is expected
         to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        size (single value): size of the squared croped box
    """
      """
      Transformation that selects a randomly rotated region in the image within a specific 
      range of degrees and a fixed squared size.
      """
      angle, extent = get_param(self.degree, self.size)
      if isinstance(img, Tensor):
        d_1=img.size(dim=1)
        d_2=img.size(dim=2)
      else:
        raise TypeError("Img should be a Tensor")

      extent_1 = [float(extent), float(d_1-extent)]
      extent_2 = [float(extent), float(d_2-extent)]

      center_1 = float(torch.empty(1).uniform_(extent_1[0], extent_1[1]).item())
      center_2 = float(torch.empty(1).uniform_(extent_2[0], extent_2[1]).item())

      center = (int(center_1), int(center_2))

      img_raw=img.cpu().detach().numpy()

      cr_image_0 = subimage(img_raw[0], center, angle, self.size, self.size)
      cr_image_1 = subimage(img_raw[1], center, angle, self.size, self.size)

      return torch.Tensor(np.array([cr_image_0,cr_image_1]), device='cpu')

class Secuential_trasn(torch.nn.Module):
    """Generates a secuential transformation"""
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, img):
      t_list=[img]
      for t in self.transforms:
        t_list.append(t(t_list[-1]))
      return t_list[-1]

class segDataset(torch.utils.data.Dataset):
  def __init__(self, root, l=3000, s=96):
    super(segDataset, self).__init__()
    self.root = root
    self.size = s
    self.l = l
    self.classes = {'Intergranular lane' : 0,
                    'Granules with dots' : 1,
                    'Granules with lanes' : 2,
                    'Complex-shape granules' : 3, 
                    'Normal-shape granules' : 4}

    self.bin_classes = ['Intergranular lane', 'Granules with dots', 'Granules with lanes',
                        'Complex-shape granules', 'Normal-shape granules']

    self.transform_serie = Secuential_trasn([Ttorch.ToTensor(),
                                            RandomRotation_crop((0, 180), self.size),
                                            Ttorch.RandomHorizontalFlip(p=0.5),
                                            Ttorch.RandomVerticalFlip(p=0.5)
                                            ])
    
    file_list = sorted(glob(self.root+'*.npz'))
    self.images = []
    self.masks = []
    for f in file_list:
      file = np.load(f)
      self.map_f = file['smap'].astype(np.float32)
      self.mask_map_f = file['cmask_map'].astype(np.float32)
      for i in range(int(self.l/len(file_list))+1):
        img_t = self.transform_serie(np.array([self.map_f, self.mask_map_f]).transpose())
        self.images.append(img_t[0].unsqueeze(0))
        self.masks.append(img_t[1].type(torch.int64))
  
  def __getitem__(self, idx):
    return self.images[idx], self.masks[idx] 

  def __len__(self):
    return len(self.images)

  def choose(self): 
    idx = random.randint(len(self))
    return self.images[idx], self.masks[idx] 