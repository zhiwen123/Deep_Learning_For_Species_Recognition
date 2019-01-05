import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from numpy import argmax
import numpy as np

def picRecognition(model,pic_path,pic_name,classes,out_text_file):
        img = image.load_img(pic_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)

        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        print(pic_name,': ', 'Predicted: ', classes[argmax(preds)],' Accuracy: ', preds[0]) 
        out_text_file.write(pic_name + ': ' + 'Predicted: ' + classes[argmax(preds)] + ' Accuracy: ' + str(preds[0]) + '\n') 

if __name__ == "__main__":
        fine_tuned_model_path = r'..\Model\leopard_tiger_model_epochs_30_batch_32.h5'
        fine_tuned_model = load_model(fine_tuned_model_path)
        source = r'..\Test_data'
        out_text_file_path = r'..\Amur_tiger_leopard_recognition.txt'
        out_text_file = open(out_text_file_path,"w+")
        classes = {0:'tiger',1:'leopard',2:'others'}
        for root, dirs, filenames in os.walk(source):
                for f in filenames:
                        fullpath = os.path.join(source, f)
                        picRecognition(fine_tuned_model,fullpath,f,classes,out_text_file)        
        
        out_text_file.close()    