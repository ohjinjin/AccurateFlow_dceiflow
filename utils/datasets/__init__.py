from .FlyingChairs2 import FlyingChairs2
from .MVSEC import MVSEC
from .GoPro import GoPro
from .REBlur import REBlur

def fetch_dataset(args):
    """ Create the data loader for the corresponding training set """

    train_dataset = None
    val_datasets = None
    val_setnames = None

    if args.stage == 'chairs2':
        aug_params = {'crop_size': args.crop_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        train_dataset = FlyingChairs2(args, '/data/FlyingChairs2', data_kind='train', aug_params=aug_params)
        dataset1_val = FlyingChairs2(args, '/data/FlyingChairs2', data_kind='trainval')
        val_datasets = [dataset1_val]
        val_setnames = ['chairs2trainval']
    
    elif args.stage == 'gopro':
        aug_params = {'crop_size': args.crop_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        train_dataset = GoPro(args, '/data/gopro_dceiflow_s2s3', data_kind='train', aug_params=aug_params)
        dataset1_val = GoPro(args, '/data/gopro_dceiflow_s2s3', data_kind='traintest')
        val_datasets = [dataset1_val]
        val_setnames = ['goprotraintest']

    assert train_dataset is not None

    return train_dataset, val_datasets, val_setnames


def fetch_test_dataset(args):
    """ Create the torch Dataset for the corresponding testing set / name """

    test_datasets = None
    names = None

    if args.stage == 'chairs2' or args.stage == 'chairs2val':
        dataset = FlyingChairs2(args, '/data/FlyingChairs2', data_kind='val')
        test_datasets = [dataset]
        names = ['chairs2val']

    elif args.stage == 'chairs2train':
        dataset = FlyingChairs2(args, '/data/FlyingChairs2', data_kind='train')
        test_datasets = [dataset]
        names = ['chairs2train']

    elif args.stage == 'mvsec' or args.stage == 'mvsecfull':
        dataset1 = MVSEC(args, '/data/MVSEC_HDF5', data_split='indoor_flying1')
#         dataset2 = MVSEC(args, '/data/MVSEC_HDF5', data_split='indoor_flying2')
#         dataset3 = MVSEC(args, '/data/MVSEC_HDF5', data_split='indoor_flying3')
#         dataset4 = MVSEC(args, '/data/MVSEC_HDF5', data_split='outdoor_day1')
#         dataset5 = MVSEC(args, '/data/MVSEC_HDF5', data_split='outdoor_day2')
        test_datasets = [dataset1]#, dataset2, dataset3, dataset4, dataset5]
        names = ['mvsecval/indoor_flying1']#, 'mvsecval/indoor_flying2', 'mvsecval/indoor_flying3', \
            #'mvsecval/outdoor_day1', 'mvsecval/outdoor_day2']
    
#     elif args.stage == 'gopro' or args.stage == 'goprofull':
# #         dataset = GoPro(args, '/data/gopro_simulated')
#         dataset = GoPro(args, '/data/gopro_dceiflow')
#         test_datasets = [dataset]
#         names = ['GOPR0384_11_00']
        
    elif args.stage == 'gopro' or args.stage == 'goprotest':
        dataset = GoPro(args, '/data/gopro_dceiflow_2', data_kind='train')
        test_datasets = [dataset]
        names = ['goprotest']

#    elif args.stage == 'gopro':
#        dataset = FlyingChairs2(args, '/data/gopro_dceiflow', data_kind='train')
#        test_datasets = [dataset]
#        names = ['goprotrain']
    
    elif args.stage == 'reblur' or args.stage == 'reblurtest':
        dataset = REBlur(args, '/data/REBlur_dceiflow', data_kind='test')
        test_datasets = [dataset]
        names = ['reblurtest']

    assert test_datasets is not None

    return test_datasets, names
