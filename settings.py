base_architecture = 'resnet34'
img_size = 224
num_classes = 5
prototype_shape = (num_classes*10, 128, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'two-small-poc/'

data_home = 'Data/chestxray/multi/'
data_path = data_home + 'small/'


# Train
results_path = 'results/' + \
    data_path.split('/')[-3]+'/'+data_path.split('/')[-2]+'/'
train_dir = data_path + 'Train/'
test_dir = data_path + 'Valid/'
train_push_dir = data_path + 'Train/'
train_batch_size = 64
test_batch_size = 16
train_push_batch_size = 64

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.8,
    'l1': 1e-4,
}

epoch_start = 0
num_train_epochs = 11
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(epoch_start, num_train_epochs) if i % 10 == 0]


# Test
load_model_dir = "saved_models"
load_model_name = "multi.pth"

test_image_dir = data_home + 'small/Test/Fibrosis/'
test_image_name = '00000092_003.png'

prototype_number = 18

write_path = "Test images/PRP/"
