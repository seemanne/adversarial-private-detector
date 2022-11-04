import tensorflow as tf
from private_detector.image_dataset import ImageDataset
from private_detector.private_detector import PrivateDetector

loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing= 0.1, from_logits=True)

model = PrivateDetector(inital_learning_rate = 1e-4, class_labels= [0,1])
restore_path = tf.train.latest_checkpoint('saved_checkpoint')




def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

