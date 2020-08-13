import cv2
import numpy as np
from skimage.io import imread
from skimage.io import imsave
import os

def readImage(path, file):
    img = cv2.imread(path+file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cropImage(img, coor):
    #img: image read by io.imread
    #coor: coordinates in xlsx file
    return img[int(coor[1]):int(coor[3]), int(coor[0]):int(coor[2])]

def generateCropCoor(data, fov, disturb=False, dx=0, dy=0):
    x1,y1,x2,y2 = data
    x = (x1+x2)/2
    y = (y1+y2)/2
    nx1 = int(x-fov/2)
    nx2 = int(x+fov/2)
    ny1 = int(y-fov/2)
    ny2 = int(y+fov/2)
    if disturb == False:
    	return nx1, ny1, nx2, ny2
    else:
    	return nx1+dx, ny1+dy, nx2+dx, ny2+dy

def load_data(file_dir):
  train_imgs = []
  label_imgs = []
  for file in os.listdir(os.path.join(file_dir,'img/')):
    if file.endswith(".png"):
      tI=imread(os.path.join(file_dir,'img/',file))
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      train_imgs.append(tI)
      lI=imread(os.path.join(file_dir,'mask/',file))
      label_imgs.append(lI)
  return train_imgs, label_imgs

def load_data_noCrop(file_dir):
  train_imgs = []
  label_imgs = []
  names = []
  for file in os.listdir(os.path.join(file_dir,'img/')):
    if file.endswith(".png"):
      tI=imread(os.path.join(file_dir,'img/',file))
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      train_imgs.append(tI)
      lI=imread(os.path.join(file_dir,'mask/',file))
      label_imgs.append(lI)
      names.append(file)
  return train_imgs, label_imgs, names

def load_data_noCrop_new(img_dir, mask_dir):
  train_imgs = []
  label_imgs = []
  names = []
  for file in os.listdir(img_dir):
    if file.endswith(".png"):
      tI=imread(os.path.join(img_dir,file))
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      train_imgs.append(tI)
      lI=imread(os.path.join(mask_dir,file))
      label_imgs.append(lI)
      names.append(file)
  return train_imgs, label_imgs, names

def load_data_multitask(file_dir):
  train_imgs = []
  label_imgs = []
  for file in os.listdir(os.path.join(file_dir,'img/')):
    if file.endswith(".png"):
      tI=imread(os.path.join(file_dir,'img/',file))
      if (len(tI.shape)==2):
        tI=np.expand_dims(tI,axis=2)
      train_imgs.append(tI)
      lI=np.load(os.path.join(file_dir,'mask/',file[:-4]+'.data.npy'))
      label_imgs.append(lI)
  return train_imgs, label_imgs

def np_rotate(data, times):
    result = data.copy()
    for i in range(times):
        result = np.rot90(result)
    return result

def get_image_noCrop(train_imgs,label_imgs, names, fov, res, excel):
  if len(train_imgs) != len(names):
    print("Error: names doesn't match with training images!!!")
    return None
  # get image index and mouse index
  i_idx = np.random.randint(0, len(train_imgs)-1)
  fileName = names[i_idx]
  mouse_index = int(fileName[4:8])
  rawCoor = excel[excel['Mouses']==mouse_index].iloc[:,1:5].values[0]
  dx = np.random.randint(-20,20)
  dy = np.random.randint(-20,20)

  #Crop image
  coor = generateCropCoor(rawCoor, fov, disturb=True, dx=dx, dy=dy)
  train_sample = (train_imgs[i_idx]/255).copy()
  label_sample = label_imgs[i_idx].copy()
  train_sample_cropped = cropImage(train_sample, coor)
  label_sample_cropped = cropImage(label_sample, coor)

  #resize image
  if res != fov:
    train_sample_cropped_resized = cv2.resize(train_sample_cropped, (0,0), fx=res/fov, fy=res/fov)
    label_sample_cropped_resized = cv2.resize(label_sample_cropped, (0,0), fx=res/fov, fy=res/fov)
  else:
    train_sample_cropped_resized = train_sample_cropped
    label_sample_cropped_resized = label_sample_cropped

  #Rotate image
  nrotate = np.random.randint(0, 3)
  train_sample_final = np_rotate(train_sample_cropped_resized, nrotate)
  label_sample_final = np_rotate(label_sample_cropped_resized, nrotate)
  return train_sample_final, label_sample_final, coor

def get_image_noCrop_fast(train_imgs,label_imgs, num_images):
  # get image index and mouse index
  i_idx = np.random.randint(0, num_images-1)
  train_sample = (train_imgs[i_idx]/255).copy()
  label_sample = label_imgs[i_idx].copy()
  #Rotate image
  nrotate = np.random.randint(0, 3)
  train_sample_final = np_rotate(train_sample, nrotate)
  label_sample_final = np_rotate(label_sample, nrotate)
  return train_sample_final, label_sample_final

def get_image_noCrop_fast_gaussian(train_imgs,label_imgs, num_images):
  # get image index and mouse index
  i_idx = np.random.randint(0, num_images-1)
  train_sample = (train_imgs[i_idx]/255).copy()
  label_sample = (label_imgs[i_idx]/255).copy()
  #Rotate image
  #nrotate = np.random.randint(0, 3)
  #train_sample_final = np_rotate(train_sample, nrotate) #for end to end no rotate
  #label_sample_final = np_rotate(label_sample, nrotate) #for end to end no rotate
  #return train_sample_final, label_sample_final #for end to end no rotate
  return train_sample, label_sample

def get_batch_noCrop(train_imgs,label_imgs,batch_size, ins, imageType, names, fov, res, excel):
  #names: denote the sequence of names for img and mask
  train_samples = np.zeros((batch_size,ins,ins,imageType)) #<--- for RGB image, their are three channels.
  label_samples = np.zeros((batch_size,ins,ins)) #<--- Zheng's modification
  for i in range(batch_size):
    train_sample, label_sample, _ = get_image_noCrop(train_imgs,label_imgs, names, fov, res, excel)
    train_samples[i,:,:,:] = train_sample
    label_samples[i,:,:,] = label_sample #<-----Zheng's modify for 3 classes' classification
  return train_samples, label_samples

def get_batch_noCrop_fast(train_imgs,label_imgs,batch_size, ins, imageType, num_images):
  train_samples = np.zeros((batch_size,ins,ins,imageType)) #<--- for RGB image, their are three channels.
  label_samples = np.zeros((batch_size,ins,ins)) #<--- Zheng's modification
  for i in range(batch_size):
    train_sample, label_sample = get_image_noCrop_fast(train_imgs,label_imgs, num_images)
    train_samples[i,:,:,:] = train_sample
    label_samples[i,:,:,] = label_sample #<-----Zheng's modify for 3 classes' classification
  return train_samples, label_samples

def get_batch_noCrop_fast_gaussian(train_imgs,label_imgs,batch_size, ins, imageType, num_images):
  train_samples = np.zeros((batch_size,ins,ins,imageType)) #<--- for RGB image, their are three channels.
  label_samples = np.zeros((batch_size,ins,ins, 3)) #<--- Zheng's modification
  for i in range(batch_size):
    train_sample, label_sample = get_image_noCrop_fast_gaussian(train_imgs,label_imgs, num_images)
    train_samples[i,:,:,:] = train_sample
    label_samples[i,:,:,:] = label_sample #<-----Zheng's modify for 3 classes' classification
  return train_samples, label_samples

def distanceError(error):
    result = error.copy()
    result['distance'] = np.sqrt(result['left->right']**2 + result['head->tail']**2)
    return result
