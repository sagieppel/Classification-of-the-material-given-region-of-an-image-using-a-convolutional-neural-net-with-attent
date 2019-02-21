# Trained net for region specific Classification
# 1) Set folder of train images in Image_Dir
# 2) Set folder for ground truth Annotation in AnnotationDir
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending. The value of each pixel correspond to it class
# 3) Set number of classes number in NUM_CLASSES
# 4) Run script
# 5) The trained net weight will be saved in the folder defined in: logs_dir
# 6) For other training parameters see Input section in train.py script
#------------------------------------------------------------------------------------------------------------------------
##########################################################################################################################################################################
import numpy as np
import AttentionNet as Net
import Reader as Reader

import os
import scipy.misc as misc
import torch

#...........................................Input Parameters.................................................
ImageDir="ExampleData/TrainVal_Set/Images/"
AnnotationDir="ExampleData/TrainVal_Set/Annotations/"
UseCuda=True
MinSize=160  # min width/hight of image
MaxSize=1200 # max width/hight of image
MaxBatchSize=20  # Maximoum number of images per batch h*w*bsize (reduce to prevent oom problems)
MaxPixels=800*800*3.#800*800*3. # Maximoum number of pixel per batch h*w*bsize (reduce to prevent oom problems)
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)


Trained_model_path="" # If you want  to  training start from pretrained model other wise set to =""
#-----------------------------Training Paramters------------------------------------------------------------------------
Learning_Rate=1e-5
learning_rate_decay=0.999999#


TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(300010) # Max  number of training iteration
NumClasses=-1 #Number of class if -1 Read num classes from reader
BackgroundClass=0 # Marking for background/unknown class that will be ignored
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Reader.Reader(ImageDir=ImageDir,AnnotationDir=AnnotationDir, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels, AnnotationFileType="png", ImageFileType="jpg",BackgroundClass=0, NumClasses=NumClasses)
    #Reader.Reader(TrainImageDir,TrainAnnotationFile, MaxBatchSize,MinSize,MaxSize,MaxPixels)
if NumClasses ==-1: NumClasses = Reader.NumClass+1
#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda) # Create main resnet image classification brach
Net.AddAttententionLayer()
if Trained_model_path!="":
    Net.load_state_dict(torch.load(Trained_model_path))
if UseCuda: Net.cuda()
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay)
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------
f = open(TrainLossTxtFile, "a")
f.write("Iteration\tloss\t Learning Rate="+str(Learning_Rate))
f.close()
AVGLoss=0
#..............Start Training loop: Main Training....................................................................
for itr in range(1,MAX_ITERATION):
    Images, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom(EqualClassProbabilty=False) # Read next batc
#**************************Run Trainin cycle***************************************************************************************
    Prob, Lb=Net.forward(Images,ROI=SegmentMask) # Run net inference and get prediction
    Net.zero_grad()
    OneHotLabels=torch.autograd.Variable(torch.from_numpy(LabelsOneHot).cuda(), requires_grad=False) # Convert labe to one hot encoding
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate cross entropy loss
    if AVGLoss==0:  AVGLoss=float(Loss.data.cpu().numpy()) #Caclculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy())
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient decend change weight
    torch.cuda.empty_cache()
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 30000 == 0 and itr>0:
        print("Saving Model to file in "+logs_dir)
        torch.save(Net.state_dict(), logs_dir+ "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss
        print("Step "+str(itr)+" Train Loss="+str(float(Loss.data.cpu().numpy()))+" Runnig Average Loss="+str(AVGLoss))
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write("\n"+str(itr)+"\t"+str(float(Loss.data.cpu().numpy()))+"\t"+str(AVGLoss))
            f.close()
##################################################################################################################################################

