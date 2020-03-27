
import glob
import os

WD = 'D://PYIMAGESEARCH//pyimagesearch//SegNet-Tutorial-master//CamVid//train'
WD_Annot = 'D://PYIMAGESEARCH//pyimagesearch//SegNet-Tutorial-master//CamVid//trainannot'

w_dir = WD + "/*.png"
w_dir_annot = WD_Annot+ "/*.png"
print(w_dir_annot+" "+w_dir_annot)
zippedFile = zip(glob.glob(w_dir),glob.glob(w_dir_annot))
with open("train_new.txt","w")as fp:
    for path,pathAnnot in zippedFile:
        fp.write(path+"\t"+pathAnnot+"\n")


WD = 'D://PYIMAGESEARCH//pyimagesearch//SegNet-Tutorial-master//CamVid//val'
WD_Annot = 'D://PYIMAGESEARCH//pyimagesearch//SegNet-Tutorial-master//CamVid//valannot'

w_dir = WD + "/*.png"
w_dir_annot = WD_Annot+ "/*.png"
print(w_dir_annot+" "+w_dir_annot)
zippedFile = zip(glob.glob(w_dir),glob.glob(w_dir_annot))
with open("val_new.txt","w")as fp:
    for path,pathAnnot in zippedFile:
        fp.write(path+"\t"+pathAnnot+"\n")


WD = 'D://PYIMAGESEARCH//pyimagesearch//SegNet-Tutorial-master//CamVid//test'
WD_Annot = 'D://PYIMAGESEARCH//pyimagesearch//SegNet-Tutorial-master//CamVid//testannot'

w_dir = WD + "/*.png"
w_dir_annot = WD_Annot+ "/*.png"
print(w_dir_annot+" "+w_dir_annot)
zippedFile = zip(glob.glob(w_dir),glob.glob(w_dir_annot))
with open("test_new.txt","w")as fp:
    for path,pathAnnot in zippedFile:
        fp.write(path+"\t"+pathAnnot+"\n")
