import tensorflow as tf
import cv2
import numpy as np
import generative_inpainting.neuralgym2.neuralgym as ng
from generative_inpainting.inpaint_model import InpaintCAModel as InpaintCAModelv1
from generative_inpainting_v2.inpaint_model import InpaintCAModel as InpaintCAModelv2
from os import path
import sys
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
        mask = cv2.imread(path.join(config.MASK_PATH, path.splitext(filename)[0] + config.FILE_EXT))

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        result = sess.run(output, feed_dict={input_image_ph: input_image})
        #print('Processed: {}'.format(path.splitext(filename)[0] + config.FILE_EXT))
        cv2.imwrite(path.join(config.OUT_PATH, path.splitext(filename)[0] + config.FILE_EXT), result[0][:, :, ::-1])
