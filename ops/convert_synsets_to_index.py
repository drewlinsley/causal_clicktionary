import os
from argparse import ArgumentParser
from ops import synsets, utilities


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

    # 1. Find all images in each ip corresponding to the give lf.
    for f in dbc.label_files:
        cats = utilities.get_label_list(
            os.path.join(
                dbc.image_list_path,
                f))
        indices = [synsets.get_idx_from_file(ct) for ct in cats]
        names = [synsets.get_name_from_file(ct) for ct in cats]
        indices = [x[0] for x in indices if len(x)]
        names = [x[0] for x in names if len(x)]
        output = os.path.join(
                dbc.image_list_path, 'idx_%s' % f)
        with open(output, 'a') as f1:
            for idx in indices:
                f1.write('%s\n' % idx)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, dest="config_name",
        default='config', help="Name of labeling configuration file.")
    args = parser.parse_args()
    main(**vars(args))
