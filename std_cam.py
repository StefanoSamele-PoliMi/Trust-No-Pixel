from tensorflow.keras import backend as K
import numpy as np
import scipy


def get_class_activation_map(model, img, layer_name="conv5_block3_out", category_id=None):
    # expand dimension to fit the image to a network accepted input size
    img = np.expand_dims(img, axis=0)

    # predict to get the winning class
    if not category_id:
        predictions = model.predict(img)
        category_id = np.argmax(predictions)

    # Get the 2048 input weights to the softmax of the winning class.
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, category_id]

    # get the final conv layer
    final_conv_layer = model.get_layer(layer_name)

    # create a function to fetch the final conv layer output maps (should be shape (1, 7, 7, 2048))
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])

    # squeeze conv map to shape image to size (7, 7, 2048)
    conv_outputs = np.squeeze(conv_outputs)

    # bilinear upsampling to resize each filtered image to size of original image
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (32, 32, 1), order=1)  # dim: 224 x 224 x 2048

    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224 * 224, 2048)), class_weights_winner).reshape(224,
                                                                                                 224)  # dim: 224 x 224

    # return class activation map
    return final_output  # , label_index
