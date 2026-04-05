import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("dr_model.keras", compile=False)

IMG_SIZE = 224


def preprocess(img_path):

    original_img = cv2.imread(img_path)

    resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    img_input = resized / 255.0

    img_input = np.expand_dims(img_input, axis=0)

    return original_img, img_input


def predict_image(img_path):

    original_img, img_input = preprocess(img_path)

    prediction = model.predict(img_input)

    return prediction, original_img, img_input


def generate_gradcam(img_input, original_img):

    base_model = model.layers[0]

    last_conv_layer = base_model.get_layer("Conv_1")

    conv_model = tf.keras.Model(
        base_model.input,
        last_conv_layer.output
    )

    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    for layer in model.layers[1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_output = conv_model(img_input)

        tape.watch(conv_output)

        predictions = classifier_model(conv_output)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return heatmap, overlay