import argparse

import pandas as pd
import tensorflow as tf
import keras as K
from keras import layers, utils, mixed_precision, optimizers, backend
import tensorflow_hub as hub
import os
IMG_LENGTH = 224
IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)
IMG_SHAPE =  IMG_SIZE + (3,)
EPOCHS = 1000
TOTAL_EXAMPLES_FOR_TRAIN = 1153005
TOTAL_EXAMPLES_FOR_VALIDATION = 128161
TRAINING_BATCH_SIZE = 8
VALIDATE_BATCH_SIZE = 16
CLASSIFIER_OPTIMIZER = optimizers.SGD(.0002, momentum=.5, nesterov=True)

def make_dataset(df: pd.DataFrame, batch_size:int, total_num_examples: int) -> tf.data.Dataset:
    pass
    # train_csv_rows, _ = ImageNetSifter.decodeDir()

    # train_datagen =  (
    #     rescale=1. / 255
    # )
    # # start = time.time()
    # # train_df = train_df.sample(frac=1).reset_index(drop=True)
    # # print('Took ', time.time() - start, ' seconds')
    # generator = lambda : train_datagen.flow_from_dataframe(
    #     dataframe=df,
    #     x_col='file',
    #     y_col='class_num',
    #     target_size=IMG_SHAPE,
    #     validate_filenames=False,
    #     batch_size=batch_size,
    #     class_mode='sparse'
    # )
    # imgs, classes = next(validation_generator)
    # print(imgs.shape)
    # print(classes.shape)
    # return utils.image_dataset_from_directory(
    #     directory=
    # )


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
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        model: K.Model,
        classifier_optimizer: optimizers.Optimizer,
        checkpoint: tf.train.Checkpoint,
        checkpoint_prefix: str,
        num_examples_per_epoch_train: int,
        num_examples_per_epoch_validation: int,
        training_batch_size: int,
        validate_batch_size: int,
)-> None:
    train_gen = make_dataset(train_df, training_batch_size, TOTAL_EXAMPLES_FOR_TRAIN)
    validation_gen = make_dataset(validation_df, validate_batch_size, TOTAL_EXAMPLES_FOR_VALIDATION)
    print(train_gen)
    print(validation_gen)
    pass


def main(
    use_mixed_precision: bool=False,
    training_batch_size: int=TRAINING_BATCH_SIZE,
    validate_batch_size: int=VALIDATE_BATCH_SIZE,
    classifier_optimizer: int=CLASSIFIER_OPTIMIZER
)-> None:
    num_examples_per_epoch_train = TOTAL_EXAMPLES_FOR_TRAIN - (TOTAL_EXAMPLES_FOR_TRAIN % training_batch_size)
    num_examples_per_epoch_validation = TOTAL_EXAMPLES_FOR_VALIDATION - (TOTAL_EXAMPLES_FOR_VALIDATION % validate_batch_size)
    train_df = pd.read_csv('./classes_train.csv')
    validation_df = pd.read_csv('./classes_validate.csv')
    setup_ml_env(use_mixed_precision=use_mixed_precision, training_batch_size=training_batch_size, validate_batch_size=validate_batch_size)

    mobile_net = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5",
                               trainable=True, arguments=dict(batch_norm_momentum=0.997))
    bird_watcher = K.Sequential()
    bird_watcher.add(mobile_net)
    bird_watcher.add(layers.Dense(1, activation='relu'))
    bird_watcher.compile(loss="binary_crossentropy", optimizer=classifier_optimizer)
    # bird_watcher.summary()

    checkpoint_dir = './bird_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        bird_watcher=bird_watcher,
    )
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    train(
        train_df=train_df,
        validation_df=validation_df,
        model=bird_watcher,
        classifier_optimizer=classifier_optimizer,
        checkpoint=checkpoint,
        checkpoint_prefix=checkpoint_prefix,
        num_examples_per_epoch_train=num_examples_per_epoch_train,
        num_examples_per_epoch_validation=num_examples_per_epoch_validation,
        training_batch_size=training_batch_size,
        validate_batch_size=validate_batch_size,
    )


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
        default='SGD'
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
    args = parser.parse_args()
    if args.optimizer == 'adam':
        print('Using Adam optimizer')
        CLASSIFIER_OPTIMIZER = tf.keras.optimizers.Adam(2e-4, beta_1=0)
    elif args.optimizer == 'nadam':
        print('Using Nadam optimizer')
        CLASSIFIER_OPTIMIZER = tf.keras.optimizers.Nadam(2e-4, beta_1=0)
    else:
        print('Using SGD optimizer')
    main(
        use_mixed_precision=args.use_mixed_precision,
        classifier_optimizer=CLASSIFIER_OPTIMIZER,
        training_batch_size=args.training_batch_size,
        validate_batch_size=args.validate_batch_size,
    )