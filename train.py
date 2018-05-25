#download the dataset for training from  github.com/ardamavi/Sign-Language-Digits-Dataset
import numpy as np
import keras
from keras.models import Model,Input
from skimage.transform import rotate
from keras.layers import Dense
from  keras.layers import Convolution2D, MaxPooling2D,GlobalAveragePooling2D
from keras import initializers
import glob
import cv2
import random
import tqdm

#provide dataset path to read the images and model path to serielize the best model while training
path ="../dataset/"
model_path="../model/"

#find all the folder names
folder_names = glob.glob(path+"*")
num_class = len(folder_names)

train_set=[]
val_set =[]
val_split = 0.2 #20% of the dataset will be pushed to the valset

def add_to_dict(list_name,class_,list_):
    for kn in list_:
        list_name.append({"path":kn,"label":class_})
    
#add image path and label information to the train_set and val_set lists
for folder in folder_names:
    class_ = int(folder.split("\\")[-1])
    all_image_name  = glob.glob(folder+"\\*")
    random.shuffle(all_image_name)
    train_list = all_image_name[:int((1-val_split)*len(all_image_name))]
    val_list  = all_image_name[int((1-val_split)*len(all_image_name)):]
    add_to_dict(train_set,class_,train_list)
    add_to_dict(val_set,class_,val_list)
    


im_h = 100 #image height
im_w = 100 #image width
im_d = 3 #image_depth
batch_size  = 10
epoch = 50
#patience =10  #this is to early stop the network when there is no improvement in val set
init_accur =0 #initialize accuracy for model early stopping
   
def create_model(optimizer='adam'):
    inp = Input(shape=(im_h,im_w,im_d))
    conv1 = Convolution2D(64, kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(inp)
    conv2 = Convolution2D(64, kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(conv1)
    pool1  = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Convolution2D(128,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(pool1)
    conv4 = Convolution2D(128,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(conv3)
    pool2  = MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5 = Convolution2D(256,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(pool2)
    conv6 = Convolution2D(256,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = True)(conv5)
    conv7 = Convolution2D(256,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = True)(conv6)
    pool4  = GlobalAveragePooling2D()(conv7)
    #flatten = Flatten()(pool3)
    dense_2 = Dense(10, activation='softmax',kernel_initializer=initializers.he_normal(seed=1234),bias_initializer=initializers.Constant(value=0.1))(pool4)
    adam = keras.optimizers.Adam(lr=0.00005)
    #rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model = Model(inputs=inp, outputs=dense_2)
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
    return model

model = create_model()

#initialize layer weights from the vgg16 network
#download vgg weights from http://www.cs.toronto.edu/~frossard/post/vgg16/

vgg16 = np.load(model_path+"vgg16_weights.npz")
layer_names= []
for key in vgg16.keys():
    layer_names.append(key)
    
layer_no = 7
layer_names= np.sort(layer_names)
layer_names= layer_names[:2*layer_no]

model.layers[1].set_weights([vgg16[layer_names[0]],vgg16[layer_names[1]]])
model.layers[2].set_weights([vgg16[layer_names[2]],vgg16[layer_names[3]]])

model.layers[4].set_weights([vgg16[layer_names[4]],vgg16[layer_names[5]]])
model.layers[5].set_weights([vgg16[layer_names[6]],vgg16[layer_names[7]]])

model.layers[7].set_weights([vgg16[layer_names[8]],vgg16[layer_names[9]]])
model.layers[8].set_weights([vgg16[layer_names[10]],vgg16[layer_names[11]]])
model.layers[9].set_weights([vgg16[layer_names[12]],vgg16[layer_names[13]]])

#batch iterator
def batch_iterator(full_list, batch_size):
    len_ = len(full_list)
    for idx_ in range(0, len_, batch_size):
        yield full_list[idx_:min(idx_ + batch_size, len_)]
        
#one hot encode numeric class values      
def one_hot(val,num_class):
    zz= [0]*num_class
    zz[val]=1
    return zz        

#train the dataset in batches
for epo_ in range(0,epoch):
    #shuffle the train list for every epoch
    random.seed(epo_)
    random.shuffle(train_set)
    print(epo_)
    for chunks_ in tqdm.tqdm(batch_iterator(train_set, batch_size)):
        images_b = []
        labels_b = []
        for ll in chunks_:
            images_b.append(rotate(cv2.resize(cv2.imread(ll["path"]),(im_h,im_w))/255,random.randint(-15,15)))
            labels_b.append(one_hot(ll["label"],num_class))
        images_b =np.array(images_b)
        labels_b = np.array(labels_b)
        model.train_on_batch(x=images_b,y=labels_b)
    
    val_act=[]
    val_pred=[]   
    for chunks_ in tqdm.tqdm(batch_iterator(val_set, batch_size)):
        images_b = []
        labels_b = []
        for ll in chunks_:
            images_b.append(cv2.resize(cv2.imread(ll["path"]),(im_h,im_w))/255)
            labels_b.append(ll["label"])
        val_act+=labels_b
        images_b =np.array(images_b)
        predict = model.predict_on_batch(images_b)
        val_pred+=np.argmax(predict,axis=1).tolist()    
    val_act = np.array(val_act)
    val_pred = np.array(val_pred)
    accuarcy_ = np.sum(val_act==val_pred)/len(val_act)
    print("accuracy is:", accuarcy_)
    #save the best model
    if accuarcy_ > init_accur : 
       init_accur =  accuarcy_
       model.save_weights(model_path+"CAM_model.hdf5")