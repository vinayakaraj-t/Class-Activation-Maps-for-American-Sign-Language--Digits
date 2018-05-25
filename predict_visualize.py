import numpy as np
import keras
from keras.models import Model,Input
from keras.layers import Dense
from  keras.layers import Convolution2D,GlobalAveragePooling2D,MaxPooling2D
from keras import initializers
import cv2

#provide dataset path of test images for prediction and model path to load the best trained model
path ="../test_images/"
model_path="../model/"

#place image name here for prediction
img_name = "example_1.jpg"


im_h = 100 #image height
im_w = 100 #image width
im_d = 3 #image_depth

def create_model(optimizer='adam'):
    inp = Input(shape=(im_h,im_w,im_d))
    conv1 = Convolution2D(64, kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(inp)
    conv2 = Convolution2D(64, kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(conv1)
    pool1  = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Convolution2D(128,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(pool1)
    conv4 = Convolution2D(128,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same",trainable = False)(conv3)
    pool2  = MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5 = Convolution2D(256,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same")(pool2)
    conv6 = Convolution2D(256,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same")(conv5)
    conv7 = Convolution2D(256,kernel_size=(3, 3), activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros',padding="same")(conv6)
    pool4  = GlobalAveragePooling2D()(conv7)
    #flatten = Flatten()(pool3)
    dense_2 = Dense(10, activation='softmax',kernel_initializer=initializers.he_normal(seed=1234),bias_initializer=initializers.Constant(value=0.1))(pool4)
    adam = keras.optimizers.Adam(lr=0.0001)
    #rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model = Model(inputs=inp, outputs=[dense_2,conv7])
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
    return model

#load the saved best model
model = create_model()
model.load_weights(filepath=model_path+"CAM_model.hdf5")


#create CAM for the test image and save it in the test image path
read_img =  cv2.imread(path+img_name)   
read_img = cv2.resize(read_img,(im_h,im_w))/255
read_img = np.reshape(read_img,(1,im_h,im_w,im_d))
pred,conv_layer = model.predict(read_img)

#extract weights from the dense layer "dense2"
d_wt = model.layers[-1].get_weights()[0]
pred_class = np.argmax(pred)
d_wt = d_wt[:,pred_class]

#pick top 10 weights for creating the class activation maps
k = d_wt.argsort()[-10:][::-1]
d_wt_1 = d_wt[k]
c_l=conv_layer[:,:,:,k]
   
activation_map =[]
for len_ in range(len(d_wt_1)):
    activation_map.append(d_wt_1[len_]*np.squeeze(c_l[:,:,:,len_]))
    
activation_map = np.array(activation_map).sum(axis=0)
activation_map = (activation_map - np.min(activation_map))/np.max(activation_map)
activation_map = np.uint8(255 * activation_map)
#resize_map = resize(activation_map,(im_h,im_w))
#plt.imshow(np.squeeze(read_img)*(resize_map/255))
img = cv2.imread(path+img_name)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(activation_map,(width, height)), cv2.COLORMAP_JET)
result = heatmap*0.3 + img *0.5
cv2.imwrite(path+'CAM_'+img_name, result)