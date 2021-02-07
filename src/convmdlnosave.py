import os
from pydicom import dcmread
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
import skimage
import tensorflow as tf
from tensorflow_addons.losses import pinball_loss
from tensorflow.keras.backend import greater, zeros_like, variable, constant, int_shape, mean, shape, ones_like
from tensorflow import boolean_mask, where
from pickle import load
import math

# Sample: conv3dModel(age,sex,week,fvc,percent,smokingStatus,imgpath,modelpath,scalerpath)

# Function to use
# Paths expect an / at the end
def conv3dModel(age,sex,week,fvc,percent,smokingStatus,imgpath,modelpath='',scalerpath=''):
    #print("conv3D Model:")
    try:
        jpegfolder = imageTrans(imgpath)
        xrImg=loadImages(jpegfolder)
    except:
        print('Saving JPEGS failed. RESNET model.')
        xImg = imageTrans2(imgpath)
        xrImg=[]
        for img in xImg:
            jImg=[]
            img=img/32768*255
            for i in img:
                kImg=np.dstack((i,i,i))
                jImg.append(kImg[0])
            xrImg.append(jImg)
        xrImg=np.array(xrImg)
    x=createTabInput(age,sex,week,fvc,percent,smokingStatus,scalerpath)
    output=conv3dPred(xrImg,x,modelpath)
    return output


#==================== Main sub-functions ====================


def imageTrans(imgpath):
    #renameImg(imgpath)
    foldersavedir=imgpath+'/jpeg/'
    images=sorted(os.listdir(imgpath))
    images=[img for img in images if img.endswith('.dcm')]

    if not os.path.exists(foldersavedir):
            os.mkdir(foldersavedir)
    if len(os.listdir(foldersavedir))==48:
        #print(".    Folder already created and images included.")
        return foldersavedir

    imagecrop=imgpath+"/"+images[round(len(images)/2)]
    ds1 = dcmread(imagecrop)
    imgCrop= cropImg(ds1.pixel_array)
    
    
    imglist=[]
    for j in range(len(images)):
        ds = dcmread(imgpath+"/"+images[j])
        imgC = ds.pixel_array[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3]]
        imglist.append(imgC)
        
    imgCR=resizeImg(imglist,48,48,48)
    
    for j in range(48):
        imagesavedir=foldersavedir+str(j)+'.jpeg'
        plt.figure(figsize=(1,1))
        plt.imshow(imgCR[j], cmap = 'Greys')
        plt.axis('off')
        plt.savefig(imagesavedir, bbox_inches='tight',pad_inches = 0,dpi=64)
        plt.close()
    renameImg(foldersavedir)
    #print(".    Folder and images have been created.")
    return foldersavedir

def loadImages(jpegpath):
    xImg=[]
    images=sorted(os.listdir(jpegpath))
    for img in images:
        rImg=plt.imread(jpegpath+img)
        rImg=skimage.img_as_ubyte(rImg)
        xImg.append(rImg)
    xImg=np.array(xImg)
    #print('.    3D images loaded.')
    return xImg

def imageTransNoSave(imgpath):
    images=sorted(os.listdir(imgpath), key=lambda v:int(v.split('.')[0]))
    if len(images)>96:
        images=images[::math.floor(len(images)/48)]    
    try:
        imagecrop=imgpath+"/"+images[round(len(images)/2)]
        ds1 = dcmread(imagecrop)
        imgCrop= cropImg(ds1.pixel_array)
        imgC = ds1.pixel_array[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3]]

    except:
        imgCrop=[0,48,0,48]
    
    imglist=[]
    for img in images:
        try:
            ds = dcmread(imgpath+"/"+img)
            imgC = ds.pixel_array[imgCrop[0]:imgCrop[1],imgCrop[2]:imgCrop[3]]
        except:
            imgC=np.zeros([np.abs(imgCrop[1]-imgCrop[0]),np.abs(imgCrop[3]-imgCrop[4])])
        imglist.append(imgC)
        
        imgCR=resizeImg(imglist,48,48,48)
        #xImg.append(imgCR)
    xImg=np.array(imgCR)        
    return xImg


def createTabInput(age,sex,week,fvc,percent,smokingStatus,scalerpath):
    scaler = load(open(scalerpath+'scaler.pkl', 'rb'))
    x=np.zeros(9)
    x[0]=week
    x[1]=fvc
    x[2]=percent
    x[3]=age
    
    if sex=="Male":
        x[5]=1
    elif sex=="Female":
        x[4]=1
        
    if smokingStatus=="Currently smokes":
        x[6]=1
    elif smokingStatus=="Ex-smoker":
        x[7]=1
    elif smokingStatus=="Never smoked":
        x[8]=1
    #print(".    Tabular input created.")
    x[0:4]=scaler.transform(x[0:4].reshape(1, -1))
    return x

def createLine(outPred):
    click=0
    first=0
    last=0
    for i in range(len(outPred)):
        if outPred[i]<=1001 and click==0:
            first=i+1
        elif outPred[i]<=1001 and click==1 and i>20:
            last=i+1
            break
        if not (outPred[i]<=1001):
            click=1

    y=np.array(outPred[first:last])
    x=np.array(list(range(first-12,last-12)))
    xmean=np.mean(x)
    ymean=np.mean(y)

    xycov = (x - xmean) * (y - ymean)
    xvar = (x - xmean)**2
    beta = xycov.sum() / xvar.sum()
    alpha = ymean - (beta * xmean)
    line=[i*beta+alpha for i in np.array(list(range(-12,134)))]
    return line

def conv3dPred(xImg,x,modelpath):
    model=tf.keras.models.load_model(modelpath+'ConvRegResBLK.hdf5',compile=False)
    outPred=model.predict([np.array([xImg]),np.array([x])])[0]
    line=createLine(outPred)
    outPred2=[outPred[i] if np.abs(outPred[i]-line[i])<300 else line[i] for i in range(len(outPred))]
    #print('.    Prediction successful.')
    return outPred2



# ==================== Utility Functions ====================
def cropImg(img):
    
    if (sum(img[0])==sum(img[round(len(img)/2)])):
        r_min=0
        r_max=len(img)
        c_min=0
        c_max=len(img[0])
        cropImg=[r_min,r_max,c_min,c_max]
        return cropImg
    r_min, r_max = None, None
    c_min, c_max = None, None
    
    for row in range(len(img)):
        if not (img[row,:]==img[0,0]).all() and r_min is None:
            r_min=row
        if (img[row,:]==img[0,0]).all() and r_min is not None and r_max is None:
            r_max=row
                
    flipImg=np.rot90(img)
    for col in range(len(flipImg)):
        if not (flipImg[col,:]==flipImg[0,0]).all() and c_min is None:
            c_min=col
        if (flipImg[col,:]==flipImg[0,0]).all() and c_min is not None and c_max is None:
            c_max=col
    cropImg=[r_min,r_max,c_min,c_max]
    return cropImg

def resizeImg(img,x3d,y3d,z3d):
    reX = x3d/len(img)
    reY = y3d/len(img[0])
    reZ = z3d/len(img[0][0])
    reImg=zoom(img,(reX,reY,reZ))
    return reImg

def renameImg(imgpath):
    images=os.listdir(imgpath)
    for image in images:
        if len(image.split('.')[0])==1:
            if os.path.exists(imgpath+'/00'+image):
                os.remove(imgpath+'/'+image)
            else:
                os.rename(imgpath+'/'+image, imgpath+'/00'+image)
        elif len(image.split('.')[0])==2:
            if os.path.exists(imgpath+'/0'+image):
                os.remove(imgpath+'/'+image)
            else:
                os.rename(imgpath+'/'+image, imgpath+'/0'+image)
                
def LLL_metric2(y_true, y_pred):
  zeros = zeros_like(y_true)
  bool_mask = greater(y_true, [0])
  y_pred = where(bool_mask, y_pred, y_true)

  diff = abs(y_pred - y_true)
  sigma = constant(value = 200, shape = [146])

  delta = minimum(diff, constant(value = 1000, shape = [146]))
  delta = diff
  sqrt2 = constant(value = 1.414, shape = [146])
  
  loss = - divide_no_nan(sqrt2 * delta, sigma) - log(where(bool_mask, sqrt2 * sigma, ones_like(y_true)))
  avg_loss = mean(boolean_mask(loss, bool_mask), axis = 0)
  
  return avg_loss

def pinball2(y_true, y_pred):
  zeros = zeros_like(y_true)
  bool_mask = greater(y_true, [0])
  y_pred = where(bool_mask, y_pred, y_true)
  #tf.print(y_pred, summarize = 20)

  return pinball_loss(y_true, y_pred)