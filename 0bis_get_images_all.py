import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as r101pi, decode_predictions as r101dp
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as v16pi, decode_predictions as v16dp
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as iv3pi, decode_predictions as iv3dp
import numpy as np
import random
from os import walk, path, makedirs
import config


# GETS THE IMAGE CORRECTLY CLASSIFIED BY ALL THE THREE NETS #
with open("ILSVRC2012_validation_ground_truth.txt", 'r') as in_file:
    validation_labels = in_file.read().split('\n')
labels = []
directories = [path.dirname(config.MASK_PATH), path.dirname(config.HOLED_PATH),
               path.dirname(config.INPAINTED_PATH), path.dirname(config.ADV_PATH)]
for directory in directories:
    if not path.exists(directory):
        makedirs(directory)

_, _, filenames = next(walk(config.IMG_PATH))
filenames.sort()
toshuffle = list(zip(filenames, validation_labels))
random.shuffle(toshuffle)
filenames, validation_labels = zip(*toshuffle)
filenames = filenames[:config.NUM_FILES * 3]  # Allow up to 0.33 accuracy on this set

# Check if Tensorflow is loaded correctly
print("Tensorflow version: " + tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

r101 = ResNet101(weights='imagenet')
v16 = VGG16(weights='imagenet')
iv3 = InceptionV3(weights='imagenet')

keptNames = []
keptImg = 0
for filename in filenames:
    ground_truth = int(validation_labels[filenames.index(filename)])
    img = image.load_img(path.join(config.IMG_PATH, filename), target_size=(224, 224))
    img_array = r101pi(np.expand_dims(image.img_to_array(img), axis=0))
    r101preds = r101.predict(img_array)
    r101dec_pred = r101dp(r101preds, top=1)[0]
    if list(config.CLASSES_DICT.keys()).index(r101dec_pred[0][0]) != ground_truth:
        continue
    else:
        img_array = v16pi(np.expand_dims(image.img_to_array(img), axis=0))
        v16preds = v16.predict(img_array)
        v16dec_pred = v16dp(v16preds, top=1)[0]
        if list(config.CLASSES_DICT.keys()).index(v16dec_pred[0][0]) != ground_truth:
            continue
        else:
            img = image.load_img(path.join(config.IMG_PATH, filename), target_size=(299,299))
            img_array = iv3pi(np.expand_dims(image.img_to_array(img), axis=0))
            iv3preds = iv3.predict(img_array)
            iv3dec_pred = iv3dp(iv3preds, top=1)[0]
            if list(config.CLASSES_DICT.keys()).index(iv3dec_pred[0][0]) != ground_truth:
                continue
            else:
                keptImg += 1
                if keptImg % 1000 == 0:
                    print(keptImg)
                else:
                    print(keptImg, end=", ")
                keptNames.append(filename)
                labels.append(list(config.CLASSES_DICT.keys()).index(iv3dec_pred[0][0]))
                if keptImg == config.NUM_FILES:
                    break

with open("results.txt", 'w') as out_file:
    out_file.write(keptNames)
    out_file.write("\n\n")
    out_file.write(labels)

print(keptNames)
print(labels)
print("Num of images: " + str(len(labels)))



