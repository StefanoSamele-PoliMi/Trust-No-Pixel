import tensorflow as tf
import cv2
import numpy as np
import generative_inpainting.neuralgym2.neuralgym as ng
from generative_inpainting.inpaint_model import InpaintCAModel as InpaintCAModelv1
from generative_inpainting_v2.inpaint_model import InpaintCAModel as InpaintCAModelv2
from os import path
import random
import sys
if len(sys.argv) > 1:
    if sys.argv[2] == "0":
        import config
    elif sys.argv[2] == "1":
        import config1 as config
else:
    import config

if __name__ == "__main__":
    ng.get_gpus(1)
    filenames = config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]
    print("\nCONFIG --> Model: " + config.MODEL_NAME + ", Attack: " + config.ATTACK_NAME + "\n")
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, config.GAN_SHAPE[0], config.GAN_SHAPE[1] * 2, 3))
    if "v2" in config.GAN_TYPE:
        model = InpaintCAModelv2()
        FLAGS = ng.Config('generative_inpainting_v2/inpaint.yml')
        output = model.build_server_graph(FLAGS, input_image_ph)
    else:
        model = InpaintCAModelv1()
        output = model.build_server_graph(input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            config.CKPT_DIR, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded. Processing.')

    for filename in filenames:
        image = cv2.imread(path.join(config.ADV_PATH, path.splitext(filename)[0] + config.FILE_EXT))
        image = cv2.resize(image, dsize=config.GAN_SHAPE, interpolation=cv2.INTER_CUBIC)
        h, w, _ = image.shape

        if config.MASSIVE_IMPAINT:
            limits = [(0, 0), (w, h)]
        else:
            mask = cv2.imread(path.join(config.MASK_PATH, path.splitext(filename)[0] + config.FILE_EXT))
            mask = mask[:, :, 0]
            nz = np.nonzero(mask)
            # Where is the salient object? Or, use (0,0) and image.shape to fully impaint
            limits = [(min(nz[0] if len(nz[0]) > 0 else [0]), min(nz[1] if len(nz[1]) > 0 else [0])),
                      (max(nz[0] if len(nz[0]) > 0 else [mask.shape[0]]), max(nz[1] if len(nz[1]) > 0 else [mask.shape[1]]))]

        runs = 0
        image_result = np.zeros_like(image)
        for tile_size in config.RIAD_NxM_TILE_SIZES:
            runs = runs + 1

            t_arr = [np.zeros_like(image).astype(bool) for i in range(config.RIAD_N_IMAGES)]
            for x in range(limits[0][0], limits[1][0], tile_size[0]):
                for y in range(limits[0][1], limits[1][1], tile_size[1]):
                    if x + tile_size[0] >= image.shape[0]:
                        x_to = image.shape[0] #- 1
                    else:
                        x_to = x + tile_size[0]
                    if y + tile_size[1] >= image.shape[1]:
                        y_to = image.shape[1] #- 1
                    else:
                        y_to = y + tile_size[1]
                    t_arr[random.randint(0, config.RIAD_N_IMAGES - 1)][x:x_to, y:y_to] = True

            for riad_mask in t_arr:
                riad_img = image * np.logical_not(riad_mask) + riad_mask.astype(np.uint8) * 255.

                assert riad_img.shape == riad_mask.shape

                grid = 8
                riad_img = riad_img[:h // grid * grid, :w // grid * grid, :]
                riad_mask = riad_mask[:h // grid * grid, :w // grid * grid, :]
                riad_img = np.expand_dims(riad_img, 0)
                input_riad_mask3d = np.expand_dims(riad_mask.astype(np.uint8) * 255., 0)
                input_image = np.concatenate([riad_img, input_riad_mask3d], axis=2)
                result = sess.run(output, feed_dict={input_image_ph: input_image})
                inpainted = result[0][:, :, ::-1]
                image_result = image_result * np.logical_not(riad_mask) \
                                   + (image_result * riad_mask * (runs-1) + inpainted * riad_mask) / runs


        # print("OK")
        cv2.imwrite(path.join(config.OUT_PATH, path.splitext(filename)[0] + config.FILE_EXT), np.squeeze(image_result).astype(np.uint8))
