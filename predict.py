import  numpy  as  np 
import  keras 
from  PIL  import  Image


import dataset
import cv2
import imutils
height = 360
width = 480
classes = 12
epochs = 100
batch_size = 1
log_filepath='./logs_100/'

data_shape = 360*480

def writeImage(image, filename):
    """ label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    
    im.save(filename)

def predict(test):
    model = keras.models.load_model('Enet.h5')
    print(test.shape)
    #np.reshape(test, (1,360,480, 3))
    test=test.reshape((1,) + test.shape)
    #test.reshape(360,480,3,1)
    
    #test=test.T.T
    #print(test.shape)
    probs = model.predict(test, batch_size=1)

    prob = probs[0].reshape((height, width, classes)).argmax(axis=2)
    #print(prob.shape)
    return prob

def main():
    #if not args.get("video",False):
    #    camera=cv2.VideoCapture(0)

    #else:
    camera=cv2.VideoCapture("CamVid.mp4")
    count=0
    while True:
        (grabbed,frame)=camera.read()
        print("1")
        #if args.get("video") and not grabbed:
        #    break
        dim = (480, 360)
        # resize image
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
 
        #frame= imutils.resize(frame,width=360,height=480)
        print(resized.shape)
        prob = predict(resized)
        filename="frame%d.png" % count
        print(filename)
        writeImage(prob, filename)
        count=count+1
if __name__ == '__main__':
    main()
