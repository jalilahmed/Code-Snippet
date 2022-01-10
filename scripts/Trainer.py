""" Training script

This script is used to train the neural network for project

created by Jalil Ahmed.
"""
import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC

from src.GlobalVariables import DATA_FOLDER, CLASS_NAMES, MODEL_NAME, HYPER_PARAMETER_DICT, EXPERIMENT_FOLDER
from src.Utils import import_images, save_model_summary
from src.WrapperFunctions import stack_vertical, full, shuffle_data, split_data
from src.Model import create_model
from src.DataGenerator import ImageGenerator


def main():
    """ Training pipeline of the experiment

    :return: None
    """
    print("Running Trainer")

    images = []
    for class_name in CLASS_NAMES:
        class_images, _ = import_images(os.path.join(DATA_FOLDER, class_name))
        images.append(class_images)

    labels = stack_vertical((full((len(images[0]), 1), 0), full((len(images[1]), 1), 1)))
    images = stack_vertical((images[0], images[1]))

    images, labels = shuffle_data(images, labels)

    model = create_model(input_shape=(64, 64, 3), num_classes=2, model_name=MODEL_NAME)

    model.summary(print_fn=save_model_summary)

    image_train, image_valid, label_train, label_valid = split_data(images,
                                                                    labels,
                                                                    test_size=0.25)

    generator_train = ImageGenerator(image_train,
                                     label_train,
                                     HYPER_PARAMETER_DICT['batch_size'],
                                     HYPER_PARAMETER_DICT['num_patches_from_image'],
                                     patch_size=HYPER_PARAMETER_DICT['patch_size'],
                                     augmentation=HYPER_PARAMETER_DICT['augmentation_toggle'])

    generator_valid = ImageGenerator(image_valid,
                                     label_valid,
                                     HYPER_PARAMETER_DICT['batch_size'],
                                     HYPER_PARAMETER_DICT['num_patches_from_image'],
                                     patch_size=HYPER_PARAMETER_DICT['patch_size'],
                                     augmentation=HYPER_PARAMETER_DICT['augmentation_toggle'])

    tensorboard_logs_path = os.path.join(EXPERIMENT_FOLDER, 'tensorboard_logs')
    tensorboard_callback = TensorBoard(log_dir=tensorboard_logs_path,
                                       histogram_freq=0,
                                       write_graph=False,
                                       write_images=False,
                                       update_freq='epoch')

    model_checkpoint_path = os.path.join(EXPERIMENT_FOLDER, 'tmp/checkpoint')
    model_checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_path,
                                                save_weights_only=True,
                                                monitor='val_categorical_accuracy',
                                                mode='max',
                                                save_best_only=True)

    early_stopper_callback = EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=0,
                                           verbose=0,
                                           mode='auto',
                                           baseline=None,
                                           restore_best_weights=False)

    optimizer = Adam(learning_rate=HYPER_PARAMETER_DICT['learning_rate'],
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=1e-07,
                     amsgrad=False,
                     name='AdamOptimizer')

    model.compile(optimizer=optimizer,
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy(), AUC()])

    model.fit(generator_train,
              epochs=HYPER_PARAMETER_DICT['epochs'],
              validation_data=generator_valid,
              verbose=1,
              callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopper_callback])


if __name__ == '__main__':
    main()
