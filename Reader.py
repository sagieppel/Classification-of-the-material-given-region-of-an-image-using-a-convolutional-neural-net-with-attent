
#
import numpy as np
import os
import random
import cv2
#------------------------Class for reading training and  validation data---------------------------------------------------------------------
class Reader:
################################Initiate folders were files are and list of train images############################################################################
    def __init__(self, ImageDir,AnnotationDir, MaxBatchSize=100,MinSize=160,MaxSize=1100,MaxPixels=800*800*5, AnnotationFileType="png", ImageFileType="jpg",BackgroundClass=0, NumClasses=-1):
        self.ImageDir=ImageDir # Image dir
        self.AnnotationDir=AnnotationDir # File containing image annotation
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight
        self.MaxSize=MaxSize #MAx image width and hight
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve out of memory issues)
        self.AnnotationFileType=AnnotationFileType # What is the the type (ending) of the annotation files
        self.ImageFileType=ImageFileType # What is the the type (ending) of the image files
        self.BackgroundClass=BackgroundClass # Value that belong to unknown/unmarked regions in the annotation map
        self.NumClass=NumClasses
#        self.CreateMaterialDict() # Dictionary between class and material for OpenSurfaces dataset
#-------------------Get All files in folder--------------------------------------------------------------------------------------
        self.FileList=[]
        for FileName in os.listdir(AnnotationDir):
            if AnnotationFileType in FileName:
                self.FileList.append(FileName)
        self.FileList=self.FileList[:10]
#-----------------------Get List o files for each class--------------------------------------------------------------------------
        print("Generating Class Map")
        self.ClassFiles={} # List of all files containing specific Class
        self.ClassNumImages=np.zeros(500,dtype=np.int64) # Number of files containing  Class


        for n,FileName in enumerate(self.FileList):
            if n%100==0: print(str(n/len(self.FileList)*100)+"%")
            Lb=cv2.imread(AnnotationDir+"/"+FileName,0)
            MaxClass=Lb.max()
            self.NumClass = np.max([self.NumClass, MaxClass])
            for Class in range(self.NumClass+1):
                      if (Class!=BackgroundClass) and  (Class in Lb):
                          if not Class in self.ClassFiles:
                              self.ClassFiles[Class]=[]
                          self.ClassFiles[Class].append(FileName)
                          self.ClassNumImages[Class]+=1


        self.ClassNumImages = self.ClassNumImages[:self.NumClass + 1] #number of images in each class
        if self.NumClass==-1:
            self.NumClass = int(self.NumClass)
        self.ImageN=0 #  Image counter For sequancel running over all images
##############################################################################################################################################
#Read next batch of images and labels with no augmentation but with croping and resizing pick random images labels and cropping (for training)
# EqualClassProbabilty: Do you want to pick examples with equal probability for each class
#  MinClassExample: Min number of training example for class in order for it to be use in training

    def ReadNextBatchRandom(self,EqualClassProbabilty=True,MinClassExample=1):
#=====================Initiate batch=============================================================================================
        Hb=np.random.randint(low=self.MinSize,high=self.MaxSize) # Batch hight
        Wb=np.random.randint(low=self.MinSize,high=self.MaxSize) # batch  width
        BatchSize=np.int(np.min((np.floor(self.MaxPixels/(Hb*Wb)),self.MaxBatchSize))) # Number of images in batch
        BImgs=np.zeros((BatchSize,Hb,Wb,3)) # Images
        BSegmentMask=np.zeros((BatchSize,Hb,Wb)) # Segment mask
        BLabels=np.zeros((BatchSize),dtype=np.int) # Class
        BLabelsOneHot=np.zeros((BatchSize,self.NumClass+1),dtype=np.float32) # classes in one hot encodig
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#======================================================================================================================================
        for i in range(BatchSize):
#=========================Load image and annotation========================================================================================
#----------------Select Class----------------------------------------------------------------------------------------
            if EqualClassProbabilty: # Choose random class with equal probabilty for each class
                    ClassNum = np.random.randint(self.NumClass)
                    while self.ClassNumImages[ClassNum]<MinClassExample: ClassNum = np.random.randint(self.NumClass)  # Choose random class with equal probabilty for each class
            else:
                    ClassNum=-1 #Choose random class with proportional to the number of images in which the class occur
                    Nm = np.random.randint(self.ClassNumImages.sum())+1  #
                    Sm=0
                    for cl in range(self.ClassNumImages.shape[0]):
                        Sm+=self.ClassNumImages[cl]
                        if (Sm>=Nm):
                            ClassNum=cl
                            break
#============================Select image==========================================================================================
            ImgNum = np.random.randint(self.ClassNumImages[ClassNum])  # Choose Random image
            Ann_name = self.ClassFiles[ClassNum][ImgNum]  # Get label image name
            Img_name = self.ClassFiles[ClassNum][ImgNum].replace(self.AnnotationFileType,self.ImageFileType)  # Get label image name
#==========================Read image===============================================================================================
            Img = cv2.imread(self.ImageDir + "/" + Img_name)  # Load Image
            Img = Img[..., :: -1]
            Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name,0)  # Load Annotation
            if (Img.ndim==2): #If grayscale turn to rgb
                  Img=np.expand_dims(Img,3)
                  Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3] # Get first 3 channels incase there are more
#========================Get Mask and bouding Box========================================================================================
            [NumCCmp, CCmpMask, CCompBB, CCmpCntr] = cv2.connectedComponentsWithStats((Ann == ClassNum).astype(np.uint8)) # apply connected component
            SegNum = np.random.randint(NumCCmp-1)+1 # Chose Random segment
            Mask=(CCmpMask==SegNum).astype(np.uint8)
            bbox=CCompBB[SegNum][:4]
#========================resize image if it two small to the batch size==================================================================================
            [h,w,d]= Img.shape
            Rs=np.max((Hb/h,Wb/w)) # Resize factor
            if Rs>1:# Resize image and mask
                h=int(np.max((h*Rs,Hb)))
                w=int(np.max((w*Rs,Wb)))
                Img=cv2.resize(Img,dsize=(w,h),interpolation = cv2.INTER_LINEAR)
                Mask=cv2.resize(Mask,dsize=(w,h),interpolation = cv2.INTER_NEAREST)
                bbox=(bbox.astype(np.float32)*Rs.astype(np.float32)).astype(np.int64)
#=======================Crop image to fit batch size===================================================================================
            x1 = int(np.floor(bbox[0]))   # Bounding box x position
            Wbox = int(np.floor(bbox[2])) # Bounding box width
            y1 = int(np.floor(bbox[1]))   # Bounding box y position
            Hbox = int(np.floor(bbox[3])) # Bounding box height
            if Wb>Wbox:
                Xmax=np.min((w-Wb,x1))
                Xmin=np.max((0,x1-(Wb-Wbox)))
            else:
                Xmin=x1
                Xmax=np.min((w-Wb, x1+(Wbox-Wb)))

            if Hb>Hbox:
                Ymax=np.min((h-Hb,y1))
                Ymin=np.max((0,y1-(Hb-Hbox)))
            else:
                Ymin=y1
                Ymax=np.min((h-Hb, y1+(Hbox-Hb)))
            # if Xmax<Xmin:
            #     print("waaa")
            # if Ymax < Ymin:
            #     print("dddd")


            if not (Xmin>=Xmax or Ymin>=Ymax or Xmin<0 or Ymin<0 or Xmax>Img.shape[1] or Ymax>Img.shape[0]):
                        x0=np.random.randint(low=Xmin,high=Xmax+1)
                        y0=np.random.randint(low=Ymin,high=Ymax+1)
                        Img=Img[y0:y0+Hb,x0:x0+Wb,:]
                        Mask=Mask[y0:y0+Hb,x0:x0+Wb]
            Img=cv2.resize(Img,(Wb,Hb),interpolation = cv2.INTER_LINEAR)
            Mask=cv2.resize(Mask,(Wb,Hb),interpolation = cv2.INTER_LINEAR)
           # misc.imshow(Img)

#======================Random mirror flip===========================================================================================
            if random.random() < 0.0:  # Agument the image by mirror image
                   Img = np.fliplr(Img)
                   Mask = np.fliplr(Mask)
#=====================Add to Batch================================================================================================
            BImgs[i] = Img
            BSegmentMask[i,:,:] = Mask
            BLabels[i] = int(ClassNum)
            BLabelsOneHot[i,ClassNum] = 1
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#======================================================================================================================================
        return BImgs,BSegmentMask,BLabels, BLabelsOneHot
##############################################################################################################################################
#Read next batch without augmentation (for evaluation).
# Given an image number and a class the batch conssit on all the instances of the input class in the input image######################################################################################################
    def ReadNextImageClean(self,MaxBatchSize=40,MaxPixels=1500000):
        # ==========================Read image and annotation===============================================================================================
        if self.ImageN>=len(self.FileList):
            print("No More files to read")
            return
        Img_name=self.FileList[self.ImageN].replace(self.AnnotationFileType,self.ImageFileType)
        Ann_name=self.FileList[self.ImageN] # Get label image name
        self.ImageN+=1
        Img = cv2.imread(self.ImageDir + "/" + Img_name)  # Load Image
        Img = Img[...,:: -1]
        Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name, 0)  # Load Annotation
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
        #===========================Resize if too big====================================================================================================
        Hb, Wb = Ann.shape
        Rt=MaxPixels/(Hb*Wb)
        if Rt<1:
            Hb*=Rt
            Wb*=Rt
            Img = cv2.resize(Img, (int(Wb),int(Hb)), interpolation=cv2.INTER_LINEAR)
            Ann = cv2.resize(Ann, (int(Wb),int(Hb)), interpolation=cv2.INTER_NEAREST)



        # =====================Initiate batch=============================================================================================
        Hb,Wb=Ann.shape
        BImgs = np.zeros((MaxBatchSize, Hb, Wb, 3))  # Images
        BSegmentMask = np.zeros((MaxBatchSize, Hb, Wb))  # Segment mask
        BLabels = np.zeros((MaxBatchSize), dtype=np.int)  # Class
        BLabelsOneHot = np.zeros((MaxBatchSize, self.NumClass + 1), dtype=np.float32)  # classes in one hot encodig
        # ========================Get ROI Mask========================================================================================
        i=0 # Batchcounter
        NumClass=np.max(Ann)
        for ClassNum in range(1,NumClass+1):
            if ClassNum in Ann:
              [NumCCmp, CCmpMask, CCompBB, CCmpCntr] = cv2.connectedComponentsWithStats((Ann == ClassNum).astype(np.uint8))  # apply connected component
              for SegNum in range(1,NumCCmp):
                 Mask = (CCmpMask == SegNum).astype(np.uint8)
        # =====================Add to Batch================================================================================================
                 BImgs[i] = Img
                 BSegmentMask[i, :, :] = Mask
                 BLabels[i] = int(ClassNum)
                 BLabelsOneHot[i, ClassNum] = 1
                 i+=1
        #==================================================================================================
        BImgs = BImgs[:i]
        BSegmentMask = BSegmentMask[:i]
        BLabels = BLabels[:i]
        BLabelsOneHot = BLabelsOneHot[:i]
        #======================================================================================================================================
        return BImgs,BSegmentMask,BLabels, BLabelsOneHot

