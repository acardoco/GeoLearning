import Augmentor

path_train = "data_v2\\train"
path_validate = "data_v2\\validate"

output_train = "..\data_augmentation_output_v2_3000_800\\train"
output_validate = "..\data_augmentation_output_v2_3000_800\\validate"


p_train = Augmentor.Pipeline(path_train, output_train, save_format="PNG")

p_train.rotate90(0.5)
p_train.rotate180(0.25)
p_train.rotate270(0.25)
p_train.random_distortion(0.4, grid_width=4, grid_height=4, magnitude=8)

p_validate = Augmentor.Pipeline(path_validate, output_validate, save_format="PNG")

p_validate.rotate90(0.5)
p_validate.rotate180(0.25)
p_validate.rotate270(0.25)
p_validate.random_distortion(0.4, grid_width=4, grid_height=4, magnitude=8)


p_train.sample(3000)
p_validate.sample(800)