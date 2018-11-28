
import scipy.misc as misc
import numpy as np
import Reader
import cv2
import os
ImageDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/OpenSurface/OpenSurfaceMaterialsSmall/Images/"
AnnotationDir="/media/sagi/9be0bc81-09a7-43be-856a-45a5ab241d90/Data_zoo/OpenSurface/OpenSurfaceMaterialsSmall/TestLabels/"
Reader = Reader.Reader(ImageDir=ImageDir, AnnotationDir=AnnotationDir)
NumClasses = Reader.NumClass
for t in range(100):
    #Imgs, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextBatchRandom(EqualClassProbabilty=False)
    Imgs, SegmentMask, Labels, LabelsOneHot = Reader.ReadNextImageClean()
    for f in range(Imgs.shape[0]):
        print(f)
        if not os.path.exists(str(f)): os.mkdir(str(f))
       # print(Reader.cats[Labels[f]]['name'])
       # print(Reader.MaterialDict[Labels[f]])

        Imgs[f] = Imgs[f][..., :: -1]
        Img=misc.imresize(Imgs[f],[int(Imgs[f].shape[0]/2),int(Imgs[f].shape[1]/2)])
        ROI=misc.imresize(SegmentMask[f], [int(Imgs[f].shape[0] / 2), int(Imgs[f].shape[1] / 2)])
        cv2.imwrite(str(t)+"/Img"+str(f)+".png",Img)
        cv2.imwrite(str(t) + "/ROIMask" + str(f) + ".png", ROI)

        # misc.imsave("InputMask"+str(f+1)+".png",SegmentMask[f].astype(np.uint8))
        Imgs[f,:,:,1]  *=1-SegmentMask[f]
        Imgs[f, :, :, 2] *= 1 - SegmentMask[f]
        cv2.imwrite(str(t) + "/OverLay" + str(f) + ".png", Imgs[f])
        # misc.imshow(SegmentMask[f] * 100)
        # misc.imshow(Imgs[f][...,::-1])

        #misc.imshow(SegmentMask[f,:,:]*200)


