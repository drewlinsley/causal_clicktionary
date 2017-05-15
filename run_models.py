#!/usr/bin/env python
from argparse import ArgumentParser
from tqdm import tqdm
import os


def main(config_name=None):
    """
    Runs a query on the database and returns an image list that can be
    packaged into a portable file format.
    :::
    Inputs: config_name, the file used to generate a labeling dataset.
    Each file will become a different category.
    """
    if config_name is None:
        raise RuntimeError(
            'You need to pass the name of your label config file!')
    in_config = getattr(
        __import__('settings', fromlist=[config_name]), config_name)
    dbc = in_config.config()
    optim_method = getattr(
        __import__('ops', fromlist=[dbc.optim_method]), dbc.optim_method)
    test_method = getattr(
        __import__('ops', fromlist=[dbc.test_method]), dbc.test_method)

    # Always train on "train" tfrecords and test on validation
    train_pointer = os.path.join(
        dbc.packaged_data_path, '%s_%s.%s' % (
            'train',
            dbc.packaged_data_file,
            dbc.output_format))
    validation_pointer = os.path.join(
        dbc.packaged_data_path, '%s_%s.%s' % (
            'validation',
            dbc.packaged_data_file,
            dbc.output_format))

    # 1. Loop through model types
    for m, p in dbc.model_types.iteritems():
        print 'Training model %s' % m
        # 2. Loop through layers to test
        for layer in tqdm(p[1]):
            # Train model
            if p[2] is None:
                ckpt_file, ckpt_dir = optim_method.train_classifier_on_model(
                    model_type=m,
                    train_pointer=train_pointer,
                    selected_layer=layer,
                    model_weights=p[0],
                    config=dbc)
            else:
                ckpt_file, ckpt_dir = p[2][0], p[2][1]
                # Test model
                test_method.test_classifier(
                    model_type=m,
                    model_ckpt=ckpt_file,
                    model_dir=ckpt_dir,
                    selected_layer=layer,
                    validation_pointer=validation_pointer,
                    model_weights=p[0],
                    config=dbc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, dest="config_name",
        default='config', help="Name of labeling configuration file.")
    args = parser.parse_args()
    main(**vars(args))
