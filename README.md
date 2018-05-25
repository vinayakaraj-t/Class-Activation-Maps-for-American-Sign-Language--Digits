# Class-Activation-Maps-for-American-Sign-Language--Digits

Class Activation Maps(CAM) are an easy way to understand where the trained CNN is looking at while making predictions. It also helps to ensure whether or not the model is trained properly.Here, an attempt is made to build a CAM model using American Sign Language Dataset- Digits. The architecture of the CNN is  pretty simple with top seven convolution layers of VGG16 ,followed by a global average pooling and finally, ends with a softmax layer for classification. Only the weights of last two convolution layers are adjusted for the dataset and  the rest of the layers are kept as a fixed feature extraction layer (transfer learning). 

More on CAM can be found here https://arxiv.org/pdf/1512.04150.pdf.

Dataset for training the CAM model can be downloaded here  github.com/ardamavi/Sign-Language-Digits-Dataset. And the pretrained VGG16 model for weight initialization resides in http://www.cs.toronto.edu/~frossard/post/vgg16/.
