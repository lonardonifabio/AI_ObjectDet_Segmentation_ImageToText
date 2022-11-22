import pixellib
from pixellib.instance import instance_segmentation
import cv2
import piexif
import os, sys
import pyexiv2
from exif import Image
import csv

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2

#load the trained model to classify sign
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import dump, load
from tensorflow.keras.preprocessing.image import load_img, img_to_array


CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
FILES_NAMES = []
destdir = '/Users/lonar/Pictures/test'
FILES_NAMES = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ]
ind = 0

pickle_in = open("wordtoix.pkl", "rb")
wordtoix = load(pickle_in)
pickle_in = open("ixtoword.pkl", "rb")
ixtoword = load(pickle_in)
max_length = 74

base_model = InceptionV3(weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)
model = load_model('new-model-1.h5')

segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5')

def preprocess_img(img_path):
    #inception v3 excepts img in 299*299
    img = load_img(img_path, target_size = (299, 299))
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess_img(image)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search(bpic, beam_index = 3):
    start = [wordtoix["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            e = bpic
            preds = model.predict([e, np.array(par_caps)])
            
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def classify(file_path):
    global label_packed
    enc = encode(file_path)
    image = enc.reshape(1, 2048)
    pred = greedy_search(image)
    pred2 = beam_search(image)
    pred3 = beam_search(image, 5)
    return pred, pred2, pred3

def transfer_metadata(f1,f2):
    print("f1: "+f1+"**********f2: "+f2)
    piexif.transplant(f1,f2)

def insert_sample_details(details, tag, prediction):
    #transfer_metadata(f1,f2)
    zeroth_ifd_det = {270: details}
    zeroth_ifd_tag = {40094: tag.encode("utf-16le")}
    zeroth_ifd_com = prediction
    #exif_bytes = piexif.dump({"0th":zeroth_ifd_tag, "0th":zeroth_ifd_det})
    exif_bytes_tag = piexif.dump({"0th":zeroth_ifd_det, "0th":zeroth_ifd_tag, "0th":zeroth_ifd_det})
    piexif.insert(exif_bytes_tag,f1)




header = ['File', 'TAG', 'Prediction LSTM', 'Prediction BEAM3', 'Prediction BEAM5']
with open('data.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    while ind < len(FILES_NAMES):
        details = ""
        tag = ""
        f1 = "c:" + destdir + "/" + FILES_NAMES[ind]
        f2 = "c:" + destdir + "/" + str(ind) + ".jpg"
        results, output = segmentation_model.segmentImage(f1, show_bboxes=True, output_image_name=f2)
        prediction, prediction2, prediction3 = classify(f1)
        ind = ind + 1 

        for i, category in enumerate(results['class_ids']):
            details = details + (" Indice: "+str(i)+" Oggetto: "+str(CLASS_NAMES[category])+" Score: "+str(results['scores'][i]))
            tag = tag + str(CLASS_NAMES[category])+ ";"

        data = [f1, details, prediction, prediction2, prediction3]
        writer.writerow(data)
        print("#####################################################################################")
        print("Photo: "+f1)
        print("Details: "+details)
        print("Prediction1: "+prediction)
        print("Prediction2: "+prediction2)
        print("Prediction3: "+prediction3)
        insert_sample_details(details,tag,prediction)
    

