import Augmentor

path_train = "data\\train"
path_validate = "data\\validate"

output_train = "..\data_augmentation_output\\train"
output_validate = "..\data_augmentation_output\\validate"


p_train = Augmentor.Pipeline(path_train, output_train, save_format="PNG")

p_train.rotate(probability=1.0, max_left_rotation=5, max_right_rotation=10)

p_validate = Augmentor.Pipeline(path_validate, output_validate, save_format="PNG")

p_validate.rotate(probability=1.0, max_left_rotation=5, max_right_rotation=10)


p_train.sample(2000)
p_validate.sample(600)