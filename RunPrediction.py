# Run Predicition Using trained net based on OpenSurfaces material data set classes
# 1) set path for trained model in Trained_model_path
# 2) Set path for image in ImageFile
# 3) Set Path for ROI mask in ROIMaskFile
# 4) Run
##########################################################################################################################################################################

import numpy as np
import AttentionNet as Net
import matplotlib.pyplot as plt
import OpenSurfacesClasses
import cv2
import torch
import scipy.misc as misc

import numpy as np



#...........................................Input Parameters.................................................

Trained_model_path="logs/WeightRegionMaterialClassificationOpenSurface.torch" # If you want tos start from pretrained model weight  file
ImageFile='ExampleData/TestImages/Img.png' #
ROIMaskFile= 'ExampleData/TestImages/ROIMask3.png'
NumClasses=44
UseCuda=True

#---------------------Initiate neural net------------------------------------------------------------------------------------
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda)
Net.AddAttententionLayer()
Net.load_state_dict(torch.load(Trained_model_path))
if UseCuda: Net.cuda()
Net.eval()
#--------------------Read Image and segment mask---------------------------------------------------------------------------------
Images=cv2.imread(ImageFile)
ROIMask=cv2.imread(ROIMaskFile,0)

# cv2.imshow("Image",Images)
# cv2.imshow("Mask",ROIMask*255)
Images = Images[..., :: -1]
imgplot = plt.imshow(Images)
plt.show()
imgplot=plt.imshow(ROIMask*255) # Disply ROI mask
plt.show()

Images=np.expand_dims(Images,axis=0).astype(np.float32)
ROIMask=np.expand_dims(ROIMask,axis=0).astype(np.float32)
#-------------------Run Prediction----------------------------------------------------------------------------
Prob, PredLb = Net.forward(Images, ROI=ROIMask,EvalMode=True)  # Run net inference and get prediction
PredLb = PredLb.data.cpu().numpy()
Prob = Prob.data.cpu().numpy()
#---------------Print Prediction--------------------------------------------------------------------------
dic=OpenSurfacesClasses.CreateMaterialDict()
print("Predicted Label " + dic[PredLb[0]])
print("Predicted Label Prob="+str(Prob[0,PredLb[0]]*100)+"%")



