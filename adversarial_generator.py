import tensorflow as tf
import matplotlib.pyplot as plt
from private_detector.image_dataset import ImageDataset
from private_detector.private_detector import PrivateDetector


checkpoint_dir = 'saved_checkpoint'

loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing= 0.1, from_logits=True)

dataset = ImageDataset(
    classes_files='imagefiles.json',
    batch_size =1,
    batch_seed = 1,
    batch_sequential= True,
    steps_per_epoch=1,
    rotation_augmentation=0,
    use_augmentation=None,
    scale_crop_augmentation=1,
    image_dtype=tf.dtypes.float32,
)


model = PrivateDetector(
    initial_learning_rate = 1e-4,
    class_labels = dataset.classes,
    checkpoint_dir=checkpoint_dir,
    batch_size=1,
    use_fp16=False,
    reg_loss_weight=0,
    tensorboard_log_dir='tf_logs',
    eval_threshold=0.5
)

restore_path = tf.train.latest_checkpoint(checkpoint_dir)
model.restore(restore_path)

clown =0

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model.model(input_image)
        loss = loss_object(naive_label_parser(input_label), prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def naive_label_parser(label):
    if label.numpy()[0] == 0: return tf.convert_to_tensor([[1,0]])
    return tf.convert_to_tensor([[1, 0]])

img, input_image, input_label = next(iter(dataset))
perturbation = create_adversarial_pattern(input_image, input_label)

perturbed_image = input_image - 0.01 * perturbation


clown = 0