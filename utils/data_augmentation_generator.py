import Augmentor
import os

core_path = 'C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\datos\data_v3\set_80x80'


def aug_train_valide(path_train, path_validate,num_train,num_validate,output):
    path_train = path_train
    path_validate = path_validate

    output_train = os.path.join(core_path,'data_augmentation_2300_325\\train\\' + output)
    output_validate = os.path.join(core_path,'data_augmentation_2300_325\\validate\\' + output)


    p_train = Augmentor.Pipeline(path_train, output_train, save_format="JPG")

    p_train.rotate90(0.5)
    p_train.rotate180(0.25)
    p_train.rotate270(0.25)
    #p_train.random_distortion(0.4, grid_width=4, grid_height=4, magnitude=8)

    p_validate = Augmentor.Pipeline(path_validate, output_validate, save_format="JPG")

    p_validate.rotate90(0.5)
    p_validate.rotate180(0.25)
    p_validate.rotate270(0.25)
    #p_validate.random_distortion(0.4, grid_width=4, grid_height=4, magnitude=8)


    p_train.sample(num_train)
    p_validate.sample(num_validate)

aug_train_valide(os.path.join(core_path, 'train\parking'),
                 os.path.join(core_path, 'validate\parking'),
                 800,100,
                 'parking')
aug_train_valide(os.path.join(core_path, 'train\piscina'),
                 os.path.join(core_path, 'validate\piscina'),
                 1000,150,
                 'piscina')
aug_train_valide(os.path.join(core_path, 'train\\rotonda'),
                 os.path.join(core_path, 'validate\\rotonda'),
                 500, 75,
                 'rotonda')