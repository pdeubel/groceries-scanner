import os
from typing import List, Tuple

import click
import numpy as np
import yaml
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ParameterGrid
import tensorflow as tf
import wandb

CLASS_ID_PATH = "../freiburg_groceries_dataset/classid.txt"
DATASET_DIR = "../freiburg_groceries_dataset/images"

# CLASS_ID_PATH = "classid.txt"
# DATASET_DIR = "images/"


def load_dataset():
    classes_to_id = {}
    with open(CLASS_ID_PATH, "r") as f:
        for line in f:
            class_name, class_id = line.split()
            classes_to_id[class_name] = int(class_id)

    id_to_classes = {v: k for k, v in classes_to_id.items()}
    num_classes = len(classes_to_id.values())

    file_paths = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(DATASET_DIR):
        if len(dirnames) == 0:
            file_paths.extend([os.path.join(dirpath, x) for x in filenames])
            labels.extend([classes_to_id[os.path.basename(dirpath)] for _ in filenames])

    X = np.array(file_paths)
    y = np.array(labels)

    return X, y, classes_to_id, id_to_classes, num_classes


def create_tf_dataset(X, y, img_height, img_width, num_classes, batch_size):
    def decode_img(file_path):
        img = tf.io.read_file(file_path)
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_png(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_dataset(file_path, label):
        img = decode_img(file_path)
        label = tf.one_hot(label, depth=num_classes)

        return img, label

    def configure_for_performance(ds):
        # Caching exceeds memory usage
        # ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds


    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    ds = configure_for_performance(ds)

    return ds


def create_tf_datasets_from_splits(X_train_fold_list: List[np.ndarray], X_test_fold_list: List[np.ndarray],
                                   y_train_fold_list: List[np.ndarray], y_test_fold_list: List[np.ndarray],
                                   img_height, img_width, num_classes, batch_size):
    def decode_img(file_path):
        img = tf.io.read_file(file_path)
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_png(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_dataset(file_path, label):
        img = decode_img(file_path)
        label = tf.one_hot(label, depth=num_classes)

        return img, label

    def configure_for_performance(ds):
        # Caching exceeds memory usage
        # ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    datasets = []

    for X_train_fold, y_train_fold, X_test_fold, y_test_fold in zip(
            X_train_fold_list, y_train_fold_list, X_test_fold_list, y_test_fold_list):

        train_ds = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_fold, y_test_fold))

        train_ds = train_ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = configure_for_performance(train_ds)
        test_ds = configure_for_performance(test_ds)

        datasets.append((train_ds, test_ds))

    return datasets


def train_cross_validation(X_train_fold_list, X_test_fold_list, y_train_fold_list, y_test_fold_list,
                           num_classes, config):
    (run, img_height, img_width,
     num_classes, batch_size, model_of_fold, chosen_optimizer, lr,
     epochs) = setup_model_and_start_logging(num_classes, config)

    datasets = create_tf_datasets_from_splits(
        X_train_fold_list,
        X_test_fold_list,
        y_train_fold_list,
        y_test_fold_list,
        img_height, img_width, num_classes, batch_size)

    histories = []

    accuracies_per_fold = []
    val_accuracies_per_fold = []
    losses_per_fold = []
    val_losses_per_fold = []

    for i, (train_ds_fold, val_ds_fold) in enumerate(datasets):
        model = tf.keras.models.clone_model(model_of_fold)
        model.set_weights(model_of_fold.get_weights())

        optimizer = get_optimizer(chosen_optimizer, lr)

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=["accuracy"])

        history = model.fit(
            train_ds_fold,
            epochs=epochs,
            validation_data=val_ds_fold)

        histories.append(history)

        _accuracies = history.history["accuracy"]
        _val_accuracies = history.history["val_accuracy"]
        _losses = history.history["loss"]
        _val_losses = history.history["val_loss"]

        accuracies_per_fold.append(_accuracies)
        val_accuracies_per_fold.append(_val_accuracies)
        losses_per_fold.append(_losses)
        val_losses_per_fold.append(_val_losses)

        for _acc, _val_acc, _loss, _val_loss in zip(_accuracies, _val_accuracies, _losses, _val_losses):
            wandb.log({
                f"accuracy_{i}": _acc,
                f"val_accuracy_{i}": _val_acc,
                f"loss_{i}": _loss,
                f"val_loss_{i}": _val_loss,
            })

    mean_accuracy_per_fold = np.mean(accuracies_per_fold, axis=0)
    mean_val_accuracy_per_fold = np.mean(val_accuracies_per_fold, axis=0)
    mean_loss_per_fold = np.mean(losses_per_fold, axis=0)
    mean_val_loss_per_fold = np.mean(val_losses_per_fold, axis=0)

    for _acc, _val_acc, _loss, _val_loss in zip(
            mean_accuracy_per_fold, mean_val_accuracy_per_fold, mean_loss_per_fold, mean_val_loss_per_fold):
        wandb.log({
            "fold_accuracy": _acc,
            "fold_val_accuracy": _val_acc,
            "fold_loss": _loss,
            "fold_val_loss": _val_loss,
        })


def get_optimizer(chosen_optimizer, lr):
    if chosen_optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise RuntimeError(f"Optimizer '{chosen_optimizer}' not supported")
    return optimizer


def train_complete_dataset(X, y, num_classes, config_path=None):
    (run, img_height, img_width,
     num_classes, batch_size, model, chosen_optimizer, lr,
     epochs) = setup_model_and_start_logging(num_classes, config_path)

    ds = create_tf_dataset(X, y, img_height, img_width, num_classes, batch_size)
    optimizer = get_optimizer(chosen_optimizer, lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    history = model.fit(
        ds,
        epochs=epochs
    )

    return history


def setup_model_and_start_logging(num_classes, config=None):
    """
    1. Select hyperparameter / create hyperparameter grid
    2. For each hyperparameter set, use cross validation to train the chosen hyperparameter set + model
    3. Store the results

    2.:
        - Create datasets
        - Iterate over folds
        - Use current fold to train, test_fold to evaluate
        - Store fold result in a list
        - Map hyperparameter set and fold result list and return it


    :return:
    """

    if config is not None:
        run = wandb.init(dir="wandb_metadata", config=config)
    else:
        run = wandb.init(dir="wandb_metadata")

    img_width = wandb.config.img_size[0]
    img_height = wandb.config.img_size[1]

    chosen_model = wandb.config.model
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    chosen_optimizer = wandb.config.optimizer

    use_dropout = wandb.config.dropout
    dropout_prob = None
    if use_dropout:
        dropout_prob = wandb.config.dropout_prob

    epochs = wandb.config.epochs

    if chosen_model == "mobilenet_v2":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        base_model = tf.keras.applications.MobileNetV2(input_shape=(img_width, img_height, 3),
                                                       include_top=False,
                                                       weights="imagenet")
    elif chosen_model == "mobilenet_v3":
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

        base_model = tf.keras.applications.MobileNetV3Small(input_shape=(img_width, img_height, 3),
                                                            include_top=False,
                                                            weights="imagenet")
    else:
        raise RuntimeError(f"Model '{chosen_model}' not supported")

    base_model.trainable = False

    prediction_layer = tf.keras.layers.Dense(num_classes)

    inputs = tf.keras.Input(shape=(img_width, img_height, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if use_dropout:
        x = tf.keras.layers.Dropout(dropout_prob)(x)

    x = prediction_layer(x)
    outputs = tf.keras.layers.Softmax()(x)

    model_of_fold = tf.keras.Model(inputs, outputs)

    return run, img_height, img_width, num_classes, batch_size, model_of_fold, chosen_optimizer, lr, epochs



@click.command()
@click.option("-c", "--config", "config_path", type=str, help="Path to a YAML training config")
def main(config_path: str = None):
    config = None
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    X, y, classes_to_id, id_to_classes, num_classes = load_dataset()

    if config is not None and config["cross_val"] is False:
        training_results = train_complete_dataset(X, y, num_classes, config)
    else:
        number_cv_splits = 5
        shuffle_cv = True
        random_seed = 42

        # Create a train and a test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
        X_train, X_test, y_train, y_test = None, None, None, None

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # Create the split for the cross validation
        skf = StratifiedKFold(n_splits=number_cv_splits, shuffle=shuffle_cv, random_state=random_seed)

        X_train_fold_list = []
        X_test_fold_list = []
        y_train_fold_list = []
        y_test_fold_list = []

        for train_index, test_index in skf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            X_train_fold_list.append(X_train_fold)
            X_test_fold_list.append(X_test_fold)
            y_train_fold_list.append(y_train_fold)
            y_test_fold_list.append(y_test_fold)

        training_results = train_cross_validation(
            X_train_fold_list, X_test_fold_list, y_train_fold_list, y_test_fold_list,
            num_classes,
            config
        )

    return training_results


if __name__ == "__main__":
    final_train_results = main()
