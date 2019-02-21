# Evaluate precision of image classification in a given image region
# Instructions:
# a) Set folder of images in Image_Dir
# c) Set folder for ground truth Annotation in AnnotationDir
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending. The value of each pixel correspond to it class
# d) Set number of classes number in NUM_CLASSES
# e) Set path to trained model weights in Trained_model_path
# e) Run script
##########################################################################################################################################################################



import Reader as Reader
import torch
import numpy as np
import AttentionNet as Net
#...........................................Input Parameters.................................................
UseCuda=True
ImageDir="ExampleData/TrainVal_Set/Images/"
AnnotationDir="ExampleData/TrainVal_Set/Annotations/"
Trained_model_path="logs/WeightRegionMaterialClassificationOpenSurface.torch" # If you want tos start from pretrained model
EvaluationFile=Trained_model_path.replace(".torch","Eval.xls")
NumClasses=44 # Number of classes if  -1 read num classes from the reader
BackgroundClass=0 # Marking for background/unknown class that will be ignored
#---------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader = Reader.Reader(ImageDir=ImageDir, AnnotationDir=AnnotationDir,NumClasses=NumClasses,BackgroundClass=BackgroundClass)
if NumClasses==-1: NumClasses = Reader.NumClass+1

#---------------------Load an initiate Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda)
Net.AddAttententionLayer()
Net.load_state_dict(torch.load(Trained_model_path))
if UseCuda: Net.cuda()
Net.eval()
#==============================Region size ranges in pixesl=============================================================================================
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
#.........................Use net to make predicition.........................................
            Prob, Lb = Net.forward(Images[i:i+1], ROI=SegmentMask[i:i+1],EvalMode=True)  # Run net inference and get prediction
            PredLb = Lb.data.cpu().numpy()
#.................................Evaluate accuracy per size range......................................................
            LbSize=SegmentMask[i].sum()
            SzInd=-1
            for f,sz in enumerate(Sizes): # Find size range of the ROI region
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

#==============================Write to file=======================================================================
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

