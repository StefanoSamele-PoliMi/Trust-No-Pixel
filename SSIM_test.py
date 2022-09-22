import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from os import walk, path, makedirs

_, _, filenames = next(walk("./images/adv"))
adv_orig = []
inp_orig = []
sep_orig = []

METHOD = "S&P"

if METHOD == "S&P":
	_, _, filenames = next(walk("./images/adv"))
	adv_orig = []
	inp_orig = []
	sep_orig = []

	for file in filenames:
		image_adv = cv2.imread("./images/adv/" + file)
		image_inp = cv2.imread("./images/inpainted/" + file)
		image_orig = cv2.imread("./ILSVRC2012_img_val/" + file[:-4] + ".JPEG").astype(int)
		image_adv = cv2.resize(image_adv, (256, 256))
		image_orig = cv2.resize(image_orig, (256, 256))
		row,col,ch = image_orig.shape
		s_vs_p = 0.5
		amount = 1
		value = 16
		sep_matrix = np.random.rand(2, 256, 256, 3)
		image_s_and_p = np.where((sep_matrix[0] < amount) & (sep_matrix[1] >= s_vs_p), image_orig - value, image_orig)  # pepper
		image_s_and_p = np.where((sep_matrix[0] < amount) & (sep_matrix[1] < s_vs_p), image_s_and_p + value, image_s_and_p)  # salt
		image_s_and_p = np.clip(image_s_and_p, 0, 255)
		adv_orig.append(ssim(image_adv, image_orig, multichannel=True))
		inp_orig.append(ssim(image_inp, image_orig, multichannel=True))
		sep_orig.append(ssim(image_s_and_p, image_orig, multichannel=True))

	print("Adv vs Orig: SSIM = " + str(np.mean(adv_orig)))
	print("Inp vs Orig: SSIM = " + str(np.mean(inp_orig)))
	print("Sep (0.5, 0.3, 16) vs Orig: SSIM = " + str(np.mean(sep_orig)))

elif METHOD == "Gauss":

	for file in filenames:
		image_adv = cv2.imread("./images/adv/" + file)
		image_inp = cv2.imread("./images/inpainted/" + file)
		image_orig = cv2.imread("./ILSVRC2012_img_val/" + file[:-4] + ".JPEG").astype(int)
		image_adv = cv2.resize(image_adv, (256, 256))
		image_orig = cv2.resize(image_orig, (256, 256))
		
		row,col,ch= image_orig.shape
		mean = 0
		sigma = 6 # 16/3 = 5.33
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = np.where(gauss < 0, np.floor(gauss), np.ceil(gauss))
		gauss = gauss.reshape(row,col,ch)
		image_gauss = np.clip(image_orig + gauss, 0, 255)
		
		adv_orig.append(ssim(image_adv, image_orig, multichannel=True))
		inp_orig.append(ssim(image_inp, image_orig, multichannel=True))
		sep_orig.append(ssim(image_gauss, image_orig, multichannel=True))

	print("Adv vs Orig: SSIM = " + str(np.mean(adv_orig)))
	print("Inp vs Orig: SSIM = " + str(np.mean(inp_orig)))
	print("Gauss (0, 6 approx 16/3) vs Orig: SSIM = " + str(np.mean(sep_orig)))
