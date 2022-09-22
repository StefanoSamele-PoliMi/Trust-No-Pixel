import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
import numpy as np
import random
from os import walk, path, makedirs
import config


labels = []
directories = [path.dirname(config.MASK_PATH), path.dirname(config.HOLED_PATH),
               path.dirname(config.INPAINTED_PATH), path.dirname(config.ADV_PATH)]
for directory in directories:
    if not path.exists(directory):
        makedirs(directory)

_, _, filenames = next(walk(config.IMG_PATH))
random.shuffle(filenames)
filenames = filenames[:config.NUM_FILES * 2]  # Allow up to 0.50 accuracy on this set

# Check if Tensorflow is loaded correctly
print("Tensorflow version: " + tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = ResNet101(weights='imagenet')

keptNames = []
keptImg = 0
for filename in filenames:
    img = image.load_img(path.join(config.IMG_PATH, filename), target_size=config.NET_SHAPE)
    img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(img_array)
    dec_pred = decode_predictions(preds, top=1)[0]
    if dec_pred[0][0] not in filename:
        continue
    else:
        keptImg += 1
        keptNames.append(filename)
        labels.append(list(config.CLASSES_DICT.keys()).index(dec_pred[0][0]))
        if keptImg == config.NUM_FILES:
            break

print(keptNames)
print(labels)
print("Num of images: " + str(len(labels)))

