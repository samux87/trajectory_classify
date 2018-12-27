datasetDir = r'/home/ubuntu/Documents/traj_mining/data/categories14_allfea_30'

binaryDatasetDir = r'/home/ubuntu/Documents/traj_mining/data/binary_30'

#RNN Model Prarmeters
batch_size = 32
pl_d=64
hidden_neurons=64
learning_rate=0.001
number_layers = 1
training_epoch = 10

isAttention = False
biDirectional = True
min_grid_len = 5
GPU = "1"

#View
print_batch = 50