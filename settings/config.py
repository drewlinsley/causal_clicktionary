import os


class config():
    def __init__(self):

        """
        User configuration for creating and updating
        the serrelab image database
        """
        self.db_schema_file = os.path.join('settings', 'db_schema.txt')
        self.relative_path_pointer = '/media/data_cifs/'

        # Path to search for images
        self.image_list_path = 'image_lists'
        self.package_indices = [
            'validation',
            'train'
        ]
        self.image_paths = [
            '/media/data_cifs/clicktionary/webapp_data/clicktionary_probabilistic_region_growth_centered',
            '/media/data_cifs/clicktionary/webapp_data/lmdb_trains'
        ]
        self.label_files = [
            [
                'clicktionary_animal.txt',
                'clicktionary_vehicle.txt',
            ],
            [
                'idx_ILSVRC12_animal.txt',
                'idx_ILSVRC12_vehicle.txt'
            ]
        ]
        self.keep_proportion = [
        ]
        self.label_split = [
            '\d+',
            '_'
        ]
        self.image_file_filter = ['*.png', '*.JPEG']
        self.search_mode = ['clicktionary', 'ILSVRC12']
        self.image_sampling = [
            None,
            [50000, 50000]
        ]
        self.preshuffle = [False, True]
        self.skip_list_output = 'skip_lists/quick_package_animal_vehicle_skipped'
        self.output_format = 'tfrecords'
        self.packaged_data_path = '/media/data_cifs/clicktionary/causal_experiment_modeling/tf_records'  # '/home/drew/Desktop/clicktionary_files/'
        self.packaged_data_file = 'animal_vehicle_sampled_fixed'

        # Model output
        self.checkpoint_directory = '/media/data_cifs/clicktionary/causal_experiment_modeling/checkpoints'

        # Model parameters
        self.model_types = {
            'vgg16': [os.path.join(
                '/media/data_cifs/clicktionary/', 'pretrained_weights', 'vgg16.npy'),
                ['fc7'],
                ['/media/data_cifs/clicktionary/causal_experiment_modeling/checkpoints/vgg16_fc7_01_2017_05_15_08_27_05/model_1041.pkl', '/media/data_cifs/clicktionary/causal_experiment_modeling/checkpoints/vgg16_fc7_01_2017_05_15_08_27_05']],
            # 'clickme_vgg16': [os.path.join(
            #     '/media/data_cifs/clicktionary/clickme_experiment/checkpoints', 'gradient_0001_112369_2017_05_10_18_39_40', 'model_96000.ckpt-96000'),
            #     ['fc7'],
            #     None],  # ['/media/data_cifs/clicktionary/causal_experiment_modeling/checkpoints/clickme_vgg16_fc7_01_2017_05_14_16_44_12/model_1041.pkl',
                # '/media/data_cifs/clicktionary/causal_experiment_modeling/checkpoints/clickme_vgg16_fc7_01_2017_05_14_16_44_12/']], 
            # 'alexnet': [os.path.join(
            #     '/media/data_cifs/clicktionary/', 'pretrained_weights', 'alexnet.npy'),
            #     'conv1', 'conv2', 'conv3', 'conv4', 'fc5', 'fc6'],
            # 'inception_resnet_v2': [os.path.join(
            #    '/media/data_cifs/clicktionary/', 'pretrained_weights', 'inception_resnet_v2_2016_08_30.ckpt'),
            #      ['PreLogitsFlatten'],
            #      None]
        }

        self.optim_method = 'modeling_sklearn'  # _sklearn'  # or 'modeling'
        self.test_method = 'testing_sklearn'  # _sklearn'  # testing'
        self.train_image_size = [256, 256, 3]  # image size
        self.validation_image_size = [300, 300, 3]  # image size
        self.train_augmentations = None
        self.validation_augmentations = 'resize'
        self.model_image_size = [224, 224, 3]  # input to CNN size
        self.train_batch = 96
        self.validation_batch = 20
        self.lr = 1e-2
        self.epochs = 1
        self.classifier = 'svm'  # softmax or svm
        self.c = 1e-4  # 10**-4
        self.number_of_features = 4096

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)
