import numpy as np
from os import path
from tensorflow.keras.preprocessing import image
import sys
if len(sys.argv) > 1:
    if sys.argv[2] == "0":
        configname = "config.py"
        import config
    elif sys.argv[2] == "1":
        configname = "config1.py"
        import config1 as config
else:
    configname = "config.py"
    import config


model = config.MODEL(weights='imagenet')
progress = 0
correct = 0
topone = 0
tot = str(len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]))

for filename in config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]:
    if progress % 1000 == 0:
        print("Progress: " + str(progress) + " over " + tot + " images.")
    img = image.load_img(path.join(config.ADV_PATH, path.splitext(filename)[0] + config.FILE_EXT), target_size=config.NET_SHAPE)
    img_array = config.PREPROCESSING(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(img_array)

    dec_pred = config.DECODING(preds, top=5)[0]
    for q in range(5):
        if list(config.CLASSES_DICT.keys()).index(dec_pred[q][0]) == config.KEPT_IMAGE_LABELS[progress]:
            correct += 1
            topone = topone + 1 if q == 0 else topone
            break
    progress += 1

print("\n\nFrom " + str(len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET])) + " files, " + str(correct) + " was in the top-5.\n\n")
print("Top-1 acc: " + str(topone / len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET])))
print("Top-5 acc: " + str(correct / len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET])))

config_file = open(configname, "r")  # Accuracy can be lower because defeats include only saples in the bound of epsilon
list_of_lines = config_file.readlines()  # while the attack also generate adv samples outsite of the bounds
writelist = [item if "INITIAL_ACCURACY" not in item else "INITIAL_ACCURACY = "
                                                         + str(topone / len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]))
                                                         + "\n" for item in list_of_lines]
writelist1 = [item if "INITIAL_TOP5_ACCURACY" not in item else "INITIAL_TOP5_ACCURACY = "
                                                         + str(correct / len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]))
                                                         + "\n" for item in writelist]
config_file.close()
import os
os.remove(configname)
config_file = open(configname, "a")
config_file.writelines(writelist1)
config_file.close()
