import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from skimage.restoration import denoise_wavelet
from random import randint, uniform
import grad_cam_plus

#######################################################################
###   Implementing https://github.com/iamaaditya/pixel-deflection   ###
#######################################################################

# Check if Tensorflow is loaded correctly
print(tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Wavelet denoiser, set with parameters of the original paper
def denoiser(img):
    return denoise_wavelet(img / 255.0, sigma=0.04, mode='soft', multichannel=True, convert2ycbcr=True,
                           method='BayesShrink') * 255.0


# Pixel Deflection function definition
def pixel_deflection_with_map(img, rcam_prob, deflections, window):
    img = np.copy(img)
    height, width, channels = img.shape
    while deflections > 0:
        # for consistency, when we deflect the given pixel from all the three channels.
        for c in range(channels):
            x, y = randint(0, height - 1), randint(0, width - 1)
            # if a uniformly selected value is lower than the rcam probability skip that region
            if uniform(0, 1) < rcam_prob[x, y]:
                continue
            while True:  # this is to ensure that PD pixel lies inside the image
                a, b = randint(-1 * window, window), randint(-1 * window, window)
                if height > x + a > 0 and width > y + b > 0:
                    break
            img[x, y, c] = img[x + a, y + b, c]
        deflections -= 1
    return img


# Generate predictions of the clean image
IMG_PATH = 'reference_img.png'
model = ResNet50(weights='imagenet')
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
preds = model.predict(img_array)
print('Predicted clean image:', decode_predictions(preds, top=3)[0])

# Generate Class Activation heatMaps for the top-5 classes
sorted_args = np.argsort(preds, axis=1)
heatmaps = []
for i in range(1, 6):
    index = -1 * i
    heatmaps.append(grad_cam_plus.grad_cam_plus(model=model, img=np.squeeze(img_array), layer_name="conv5_block3_out",
                                                category_id=sorted_args[0][index]))

# Aggregate the top-5 CAM into the R-CAM by averaging
# heatmap = np.mean(heatmaps, axis=0) # Avoid using mathematical mean, original paper uses geom mean
heatmap = stats.mstats.gmean(heatmaps, axis=0)
plt.matshow(heatmaps[0])
plt.title("Original CAM")
plt.matshow(heatmap)
plt.title("R-CAM")
plt.show()

# Load image in its original size, process it and reshape the CAM accordingly and save the image
img_clean = keras.preprocessing.image.load_img(IMG_PATH)
img_clean = keras.preprocessing.image.img_to_array(img_clean)
full_cam = Image.fromarray((heatmap * 255).astype(np.uint8))\
    .resize((img_clean.shape[1], img_clean.shape[0]))
# full_cam.save("map.jpg") # Not necessary

rcam_prob = Normalize()(np.array(full_cam))  # We normalize the map, because rcam_prob is treated as probability
# values in function pixel_deflection_with_map

# Print the original image and the generated R-CAM
f, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img_clean.astype('uint8'))
ax[0].set_title('Original Image')
ax[1].imshow(rcam_prob)
ax[1].set_title('Robust CAM')
ax[2].imshow(img_clean.astype('uint8'), alpha=0.6)
ax[2].imshow(rcam_prob, cmap='jet', alpha=0.4)
ax[2].set_title('Overlayed')
for ax_ in ax:
    ax_.set_xticks([])
    ax_.set_yticks([])
plt.show()

# Deflect and then denoise the clean image
img_deflected_rcam = pixel_deflection_with_map(img_clean, rcam_prob, deflections=1500, window=25).astype(np.uint8)
img_deflected_rcam_denoised = denoiser(img_deflected_rcam).astype(np.uint8)

# Print those deflected and denoised images
plt.imshow(img_deflected_rcam)
plt.title("Only deflected")
plt.show()
plt.imshow(img_deflected_rcam_denoised)
plt.title("Deflected and denoised")
plt.show()

# Resize deflceted and denoised image for the prediction phase
img_array = np.array(Image.fromarray(img_deflected_rcam_denoised).resize((224, 224), Image.ANTIALIAS))

# Show results for what concerns the top-5 accuracy
preds = model.predict(np.expand_dims(img_array, axis=0))
print("Predicted:", decode_predictions(preds, top=5)[0])
