import argparse
import pickle
from datetime import datetime
from typing import Optional, List, Union

from numpy import  random

import pandas as pd
import tensorflow as tf
K = tf.keras
layers, utils, mixed_precision, optimizers, backend = K.layers, K.utils, K.mixed_precision, K.optimizers, K.backend
load_model = K.models.load_model
convnext = K.applications.convnext
xception = K.applications.xception
efficientnet_v2 = K.applications.efficientnet_v2
efficientnet = K.applications.efficientnet
import tensorflow_hub as hub
import os
IMG_LENGTH = 380
IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)
IMG_SHAPE = IMG_SIZE + (3,)
EPOCHS = 10
TOTAL_EXAMPLES_FOR_TRAIN = 1153005
TOTAL_EXAMPLES_FOR_VALIDATION = 128161
TRAINING_BATCH_SIZE = 16
VALIDATE_BATCH_SIZE = 32
CLASSIFIER_OPTIMIZER = optimizers.SGD(.0002, momentum=.5, nesterov=True)
BASE_DIR = os.environ.get('ImageNetDir')
SEED_VAL = random.randint(0, high=2**30)
IMAGENET_DIR = BASE_DIR + f'/ImageNetImages{IMG_LENGTH}Size'

def make_dataset(labels: Union[str, List[int]], batch_size: int, directory: str, validation_split: int = .1,
                 subset: Optional[str] = None, label_mode: str = 'binary', class_names: Optional[List[str]] = None) -> tf.data.Dataset:
    return utils.image_dataset_from_directory(
        directory=directory,
        labels=labels,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=IMG_SIZE,
        seed=SEED_VAL,
        validation_split=validation_split,
        interpolation='bicubic',
        class_names=class_names,
        subset=subset
    )


def setup_ml_env(
        use_mixed_precision=False,
        training_batch_size=TRAINING_BATCH_SIZE,
        validate_batch_size=VALIDATE_BATCH_SIZE
)-> None:
    if use_mixed_precision:
        print('Using Mixed Precision')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    else:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_global_policy(policy)

    epsilon = 1e-7
    dtype = 'float32'
    backend.set_epsilon(epsilon)
    backend.set_floatx(dtype)
    print('Parameters', backend.floatx(), backend.epsilon(), training_batch_size, validate_batch_size)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)


def train(
        model: K.Model,
        training_batch_size: int,
        validate_batch_size: int,
        model_name: str,
        directory: str,
        epochs: int,
)-> None:

    classes = ['not_bird', 'bird']
    train_gen = make_dataset('inferred', training_batch_size, directory, subset='training', class_names=classes)
    validation_gen = make_dataset('inferred', validate_batch_size, directory, subset='validation',
                                  class_names=classes)
    print(train_gen)
    print(validation_gen)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_gen.shuffle(100).prefetch(buffer_size=AUTOTUNE)

    # print(model.predict(train_ds.take(10)))
    val_ds = validation_gen.prefetch(buffer_size=AUTOTUNE)
    curr_date = datetime.now().date().isoformat()
    save_dir = f'{model_name}_{curr_date}'
    kwargs = {}
    if model_name == 'EfficientNet':
        kwargs['save_weights_only'] = True
        save_dir = f'{model_name}_weights_{curr_date}'
    save_name = 'birdwatcher_{epoch:03d}-{val_loss:.4f}.hdf5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/{save_name}'
    print(save_path)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            K.callbacks.ModelCheckpoint(
                save_path,
                **kwargs
            )
        ]
    )
    print(history)
    loss, acc = model.evaluate(val_ds)
    model.save(f'birdwatcher{model_name}_{int(loss * 1000)}_{int(acc * 100)}.h5')
    model.save_weights(f'birdwatcher{model_name}_{int(loss * 1000)}_{int(acc * 100)}_weights.h5')

def main(use_mixed_precision: bool = False, training_batch_size: int = TRAINING_BATCH_SIZE,
         validate_batch_size: int = VALIDATE_BATCH_SIZE,
         classifier_optimizer: optimizers.Optimizer = CLASSIFIER_OPTIMIZER, model_name: str = 'Xception',
         directory: str=IMAGENET_DIR, epochs: int=EPOCHS) -> None:
    num_examples_per_epoch_train = TOTAL_EXAMPLES_FOR_TRAIN - (TOTAL_EXAMPLES_FOR_TRAIN % training_batch_size)
    num_examples_per_epoch_validation = TOTAL_EXAMPLES_FOR_VALIDATION - (TOTAL_EXAMPLES_FOR_VALIDATION % validate_batch_size)
    setup_ml_env(use_mixed_precision=use_mixed_precision, training_batch_size=training_batch_size, validate_batch_size=validate_batch_size)
    kwargs = {
        "include_top": False,
        "input_shape": IMG_SHAPE,
        "weights": "imagenet",
        "pooling": "max",
        # "classes": 2,
    }
    i = layers.Input(IMG_SHAPE, dtype=tf.uint8)
    if model_name == 'EfficientNetV2':
        i = tf.cast(i, tf.float32)
        core = efficientnet_v2.EfficientNetV2S(**kwargs)(i)

    elif model_name == 'ConvNetX':
        core = convnext.ConvNeXtTiny(**kwargs)(i)

    elif model_name == 'EfficientNet':
        i = tf.cast(i, tf.float32)
        core = efficientnet.EfficientNetB1(**kwargs)(i)

    else:
        i = tf.cast(i, tf.float32)
        i = xception.preprocess_input(i)
        core = xception.Xception(**kwargs)(i)
    output = core
    # output = layers.Flatten()(core)
    # output = layers.Dropout(.1)(output)
    # output = layers.Dense(1, kernel_regularizer='l2')(output)
    output = layers.Dense(units = 120, activation='relu', kernel_regularizer='l2')(output)
    output = layers.Dropout(.2)(output)
    output = layers.Dense(units = 120, activation='relu', kernel_regularizer='l2')(output)
    output = layers.Dropout(.1)(output)
    output = layers.Dense(units = 1, activation='sigmoid')(output)
    bird_watcher = tf.keras.Model(inputs=[i], outputs=[output], name='BirdWatcher')

    bird_watcher.compile(
        loss="binary_crossentropy",
        optimizer=classifier_optimizer,
        metrics=['accuracy',],
    )
    bird_watcher.build((None,) + IMG_SHAPE)

    bird_watcher.summary()

    train(
        model=bird_watcher,
        training_batch_size=training_batch_size,
        validate_batch_size=validate_batch_size,
        model_name=model_name,
        directory=directory,
        epochs=epochs,
    )
    final_gen = make_dataset([1] * 76541, validate_batch_size, IMAGENET_DIR + "\\bird",
                             validation_split=0, label_mode='int')
    vals = bird_watcher.evaluate(final_gen.take(10))
    print(vals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an Image Classifier that sees birds.')
    parser.add_argument(
        '--use-mixed-precision',
        help='Used mixed precision',
        action='store_true'
    )
    parser.add_argument(
        '-o',
        '--optimizer',
        help='optimizier to user',
        default='sgd',
        choices=['sgd', 'nadam', 'adam', 'adamw']
    )
    parser.add_argument(
        '-b',
        '--training_batch_size',
        help='Size of batches to use',
        default=TRAINING_BATCH_SIZE,
        type=int
    )
    parser.add_argument(
        '-vb',
        '--validate-batch-size',
        help='Size of batches to use',
        default=VALIDATE_BATCH_SIZE,
        type=int
    )
    parser.add_argument(
        '-m',
        '--model',
        help='Model to use',
        default='Xception',
        choices=['Xception', 'EfficientNet', 'ConvNetX', 'EfficientNetV2']
    )
    parser.add_argument(
        '-d',
        '--directory',
        help='Directory to get images from',
        default=IMAGENET_DIR,
    )
    parser.add_argument(
        '-e',
        '--epochs',
        help='Number of epochs to do',
        default=EPOCHS,
    )
    args = parser.parse_args()
    if args.optimizer == 'adam':
        print('Using Adam optimizer')
        CLASSIFIER_OPTIMIZER = tf.keras.optimizers.Adam(1e-4, beta_1=0)
    elif args.optimizer == 'nadam':
        print('Using Nadam optimizer')
        CLASSIFIER_OPTIMIZER = tf.keras.optimizers.Nadam(1e-4, beta_1=0)
    elif args.optimizer == 'adamw':
        print('Using AdamW optimizer')
        CLASSIFIER_OPTIMIZER = tf.keras.optimizers.experimental.AdamW(1e-4, beta_1=0)
    else:
        print('Using SGD optimizer')
    main(use_mixed_precision=args.use_mixed_precision, training_batch_size=args.training_batch_size,
         validate_batch_size=args.validate_batch_size, classifier_optimizer=CLASSIFIER_OPTIMIZER, model_name=args.model)