#!/usr/bin/env python
from argparse import ArgumentParser
from ops import utilities, data_packager
from tqdm import tqdm
import os
import re


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
    for pi, ip, lf, ls, sp, ff, sm in zip(
            dbc.package_indices,
            dbc.image_paths,
            dbc.label_files,
            dbc.label_split,
            dbc.image_sampling,
            dbc.image_file_filter,
            dbc.search_mode):
        print 'Working on: %s' % ip
        # 1. Find all images in each ip corresponding to the give lf.
        cats = [utilities.get_label_list(
            os.path.join(
                dbc.image_list_path,
                it_list)) for it_list in lf]
        file_list = utilities.get_files(
            os.path.join(ip, ff))
        comb_list, comb_categories = [], []
        for idx, cg in enumerate(cats):
            it_list = []
            for e in tqdm(
                    file_list,
                    desc='Searching file list %s/%s' % (idx, len(cats))):
                if sm == 'exhaustive':
                    it_list += [e for c in cg if re.search(
                        c, e.split('/')[-1]) is not None]
                else:
                    it_list += [e for c in cg if c in e]
            comb_categories += [[idx] * len(it_list)]
            comb_list += [it_list]
        if sp is not None:
            comb_list = utilities.resample_image_lists(comb_list, sp)
        image_lists = utilities.flatten_list(comb_list)
        image_categories = utilities.flatten_list(comb_categories)
        # 2. Package these in a tf_records with pi appended to the name
        data_dict = [{
            'filename/string': f,
            'label/int64': c,
        } for f, c in zip(image_lists, image_categories)]

        if dbc.output_format == 'tfrecords':
            # 2. Package data into a portable format
            output_pointer = os.path.join(
                dbc.packaged_data_path, '%s_%s.%s' % (
                    pi,
                    dbc.packaged_data_file,
                    dbc.output_format))
            data_packager.tfrecords(
                data_dict=data_dict,
                file_path=ip,
                output_pointer=output_pointer)
        else:
            raise RuntimeError(
                'A wrapper for your output format spec isn\'t implemented.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, dest="config_name",
        default='config', help="Name of labeling configuration file.")
    args = parser.parse_args()
    main(**vars(args))
