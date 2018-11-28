# Train fully convolutional neural net with valve filters and ROI map as input
# Instructions:
# a) Set folder of train images in Train_Image_Dir
# c) Set folder for ground truth labels in Label_DIR
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# e) Set number of classes number in NUM_CLASSES
# g) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder to Valid_Image_Dir (assume that the labels and ROI maps for the validation image are also in ROIMap_Dir and Label_Dir)
# h) Run scripty
##########################################################################################################################################################################
import numpy as np
import Resnet50AttentionOnlyBiasFirstLayer as Net
Trained_model_path="logs_Resnet50AttentionOnlyBiasFirstLayer/90000.torch" # If you want tos start from pretrained model
import OpenSurfaceReader as Reader

import os
import scipy.misc as misc
import torch
from torch.autograd import Variable
import numpy as np

#...........................................Input Parameters.................................................
UseCuda=True
TrainImageDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/OpenSurface/OpenSurfaceMaterialsSmall/Images/"
AnnotationDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/OpenSurface/OpenSurfaceMaterialsSmall/TestLabels/"

EvaluationFile=Trained_model_path.replace(".torch","Eval90.xls")
#---------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader = Reader.Reader(ImageDir=TrainImageDir, AnnotationDir=AnnotationDir)
NumClasses = Reader.NumClass

#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses+1,UseGPU=UseCuda)
Net.AddAttententionLayer()
Net.load_state_dict(torch.load(Trained_model_path))
if UseCuda: Net.cuda()
Net.eval()
#===========================================================================================================================
Sizes=[1000,2000,4000,8000,16000,32000,64000,128000,256000,500000,1000000] #sizes pixels
NumSizes=len(Sizes)
#--------------------Evaluate net accuracy---------------------------------------------------------------------------------
TP=np.zeros([Reader.NumClass+1],dtype=np.float64) # True positive per class
FP=np.zeros([Reader.NumClass+1],dtype=np.float64) # False positive per class
FN=np.zeros([Reader.NumClass+1],dtype=np.float64) # False Negative per class
SumPred=np.zeros([Reader.NumClass+1],dtype=np.float64)

SzTP=np.zeros([Reader.NumClass+1,NumSizes],dtype=np.float64) # True positive per class per size
SzFP=np.zeros([Reader.NumClass+1,NumSizes],dtype=np.float64) # False positive per class per size
SzFN=np.zeros([Reader.NumClass+1,NumSizes],dtype=np.float64) # False Negative per class per size
SzSumPred=np.zeros([Reader.NumClass+1,NumSizes],dtype=np.float64)
 # Counter of segment of specific class appearence
uu=0
while (Reader.ImageN<len(Reader.FileList)):
     # for i,sz in enumerate(Sizes):

      Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextImageClean()
      uu+=1
      print(uu)
      BatchSize = Images.shape[0]
      for i in range(BatchSize):
#..................................................................
            Prob, Lb = Net.forward(Images[i:i+1], ROI=SegmentMask[i:i+1],EvalMode=True)  # Run net inference and get prediction
            PredLb = np.array(Lb.data)
#.......................................................................................
            LbSize=SegmentMask[i].sum()
            SzInd=-1
            for f,sz in enumerate(Sizes):
                 if LbSize<sz:
                     SzInd=f
                     break

            if PredLb[0] == Labels[i]:
               #   print("Correct")
                  TP[Labels[i]] += 1
                  SzTP[Labels[i],SzInd] += 1
            else:
             #     print("Wrong")
                  FN[Labels[i]] += 1
                  FP[PredLb[0]] += 1
                  SzFN[Labels[i],SzInd] += 1
                  SzFP[PredLb[0],SzInd] += 1
            SumPred[Labels[i]] += 1
            SzSumPred[Labels[i],SzInd] += 1
            # Images[i,:,:,0]*=1-SegmentMask[i]
           # print("Real Label " + str(Reader.MaterialDict[Labels[i]]) + "   Predicted " + str(Reader.MaterialDict[PredLb[0]]))
           # misc.imshow(Images[i])
          #  print("I love beer mister hazir")

#=====================================================================================================
f = open(EvaluationFile, "w")

NrmF=len(SumPred)/(np.sum(SumPred>0)) # Normalization factor for classes with zero occurrences

txt="Mean Accuracy All Class Average =\t"+    str((TP/(SumPred+0.00000001)).mean()*NrmF*100)+"%"+"\r\n"
print(txt)
f.write(txt)

txt="Mean Accuracy Images =\t"+    str((TP.mean()/SumPred.mean())*100)+"%"+"\r\n"
print(txt)
f.write(txt)


print("\r\n=============================================================================\r\n")
print(txt)
f.write(txt)

txt="SizeMax\tMeanClasses\tMeanGlobal\tNum Instances\tNumValidClasses\r\n"
print(txt)
f.write(txt)
for i,sz in enumerate(Sizes):
    if SzSumPred[:,i].sum()==0: continue
    NumValidClass=np.sum(SzSumPred[:, i] > 0)
    NrmF = len(SzSumPred[:,i]) / NumValidClass  # Normalization factor for classes with zero occurrences
    txt=str(sz)+"\t"+str((SzTP[:,i]/(SzSumPred[:,i]+0.00001)).mean()*NrmF*100)+"%\t"+str(100*(SzTP[:,i]).mean()/(SzSumPred[:,i].mean()))+"%\t"+str(SzSumPred[:,i].sum())+"\t"+str(NumValidClass)+"\r\n"
    print(txt)
    f.write(txt)
f.close()


#
#
#
# f.write("All\n")
# txt="Number\tClass Name\tNum accurance\tAccuracy\t True Positive\tFalse Positive\tFalse Negative\tFP rate\tFN rate"+"\n"
# f.write(txt)
# print(txt)
# MeanAccuracy=0
# nn=0
# for i in range(1,Reader.NumClass+1):
#       if SumPred[i]>0:
#             nn+=1
#             MeanAccuracy+=TP[i]/SumPred[i]
#             txt=str(i)+")\t"+Reader.MaterialDict[i]+" \t"+str(SumPred[i])+"\t"+str(TP[i]/SumPred[i])+"\t"+str(TP[i])+"\t"+str(FP[i])+"\t"+str(FN[i])+"\t"+str(FP[i]/SumPred[i])+"\t"+str(FN[i]/SumPred[i])+"\n"
#             f.write(txt)
#             print(txt)
# txt="\nMean Accuracy="+str(MeanAccuracy/nn)+"\n"
# f.write(txt)
# print(txt)
# txt="===================================================================================================================================================\n"
# f.write(txt)
# print(txt)
# txt="More then 100 Examples\n\n"
# f.write(txt)
# print(txt)
# txt="Number\tClass Name\tNum accurance\tAccuracy\t True Positive\tFalse Positive\tFalse Negative\tFP rate\tFN rate"+"\n"
# f.write(txt)
# print(txt)
# nn=0
# MeanAccuracy=0
# for i in range(1,Reader.NumClass+1):
#       if SumPred[i]>=100:
#             nn=nn+1
#             MeanAccuracy+=TP[i]/SumPred[i]
#             txt = str(i) +"/"+str(nn)+")\t" + Reader.MaterialDict[i] + " \t" + str(SumPred[i]) + "\t" + str(TP[i] / SumPred[i]) + "\t" + str(TP[i]) + "\t" + str(FP[i]) + "\t" + str(FN[i]) + "\t" + str(FP[i] / SumPred[i]) + "\t" + str(FN[i] / SumPred[i]) + "\n"
#             f.write(txt)
#             print(txt)
# txt="\n\nMean Accuracy="+str(MeanAccuracy/nn)+"\t Num Casses="+str(nn)
# f.write(txt)
# print(txt)
#
# f.close()
#
#
#
#
