import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from keras.utils import to_categorical

def processImage(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

def processFolder(image_folder,num_images):
        X_images = np.zeros((num_images, 224, 224, 3))
        i = 0
        for root, dirs, filenames in os.walk(image_folder):
                for f in filenames:
                        fullpath = os.path.join(image_folder, f)
                        X_images[i] = processImage(fullpath)
                        i += 1                      
        return X_images

#Reference website: https://keras.io/applications/#resnet50      
def load_dataset(tiger_source_folder, num_tiger_pictures, leopard_source_folder, num_leopard_pictures, other_source_folder, num_other_pictures): 
        X_tiger = processFolder(tiger_source_folder,num_tiger_pictures)
        Y_tiger = np.zeros((num_tiger_pictures,1))
        X_leopard = processFolder(leopard_source_folder,num_leopard_pictures)
        Y_leopard = np.zeros((num_leopard_pictures,1)) + 1
        X_other = processFolder(other_source_folder,num_other_pictures)  
        Y_other = np.zeros((num_other_pictures,1)) + 2
        return np.concatenate((X_tiger, X_leopard,X_other)), np.concatenate((Y_tiger, Y_leopard, Y_other))

if __name__ == "__main__":

        # create the base pre-trained model        
        base_model = ResNet50(weights='imagenet',include_top=False)
        
        # add a global spatial average pooling layer
        # The purpose of global spatial average pooling is to reduce the size of representation to speed up computation,
        # and makes the features more robust.
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # add a fully-connected layer, 1024 neurons:
        x = Dense(1024, activation='relu')(x)

        # add a logistic layer to predict 3 classes: Amur tiger, Amur leopard and others
        predictions = Dense(3, activation='softmax')(x)
        
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)        
        
        # train only the top layers added (which were randomly initialized)
        # freeze all convolutional ResNet50 layers
        for layer in base_model.layers:
                layer.trainable = False
        
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # train the model on the new data for a few epochs
        tiger_source_folder = r"..\tiger_folder"
        num_tiger_pictures = 100
        leopard_source_folder = r"..\leopard_folder"
        num_leopard_pictures = 100
        other_source_folder = r"..\other_folder" 
        num_other_pictures = 100
        X_train,Y_train_orig = load_dataset(tiger_source_folder, num_tiger_pictures, leopard_source_folder, num_leopard_pictures, other_source_folder, num_other_pictures)

        # Convert training and test labels to one hot matrices
        Y_train = to_categorical(Y_train_orig, 3)
        
        #save the processed traning data 
        X_train_path = r'..\Data\TrainSamples\X_train.npy'
        Y_train_path = r'..\Data\TrainSamples\Y_train.npy'
        np.save(X_train_path,X_train)
        np.save(Y_train_path,Y_train)
        
        model.fit(X_train,Y_train,epochs=5,batch_size=32)
        
        preds = model.evaluate(X_train, Y_train)
        print ("Loss = ", str(preds[0]))
        print ("Test Accuracy = ", str(preds[1]))
        
        model.save(r'..\Model\leopard_tiger_model_epochs_30_batch_32.h5')
                