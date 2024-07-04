base_architecture = 'resnet34'
img_size = 224
prototype_shape = (20, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'Combined-p-0-ChestXray14/'

data_home = 'Data/chestxray/Pneumonia/'
data_path = data_home + 'Combined-p-0-ChestXray14/'


## Train
results_path = 'results/'+data_path.split('/')[-3]+'/'+data_path.split('/')[-2]+'/'
train_dir = data_path + 'Train/'
test_dir = data_path + 'Valid/'
train_push_dir = data_path + 'Train/'
train_batch_size = 64
test_batch_size = 64
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
num_train_epochs = 31
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(epoch_start,num_train_epochs) if i % 10 == 0]



 #### Test
load_model_dir = "saved_models"
load_model_name = "0H1-100H2-50_push1.0000.pth"

test_image_dir =  data_home + 'Test-100-p-d2/Pneumonia/'
test_image_name = 'patient04665-study2-view1_frontal.jpg'

prototype_number = 18

write_path = "Test images/PRP/"