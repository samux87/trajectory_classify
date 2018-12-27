import torch
import torch.autograd as autograd
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import os,config
from rnn_cells import LSTMCell

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def keras_setter_(tensor, another_tensor):
    with torch.no_grad():
        return tensor.set_(another_tensor)

class ClassificationRNN(Module):
    def __init__(self, place_dim, pl_d=config.pl_d, classes_num = 14,
                 hidden_neurons=config.hidden_neurons,
                 isAttention=False):
        super(ClassificationRNN, self).__init__()
        self.place_embedding = nn.Embedding(place_dim+1, pl_d, padding_idx=0).cuda()
        self.isAttention = isAttention

        self.lstm = LSTMCell(pl_d, hidden_neurons).cuda()

        self.linear1 = nn.Linear(hidden_neurons * 2, hidden_neurons).cuda()
        self.linear3 = nn.Linear(hidden_neurons * 300, 256).cuda()
        self.linear2 = nn.Linear(hidden_neurons, classes_num).cuda()
        self.linear4 = nn.Linear(256, classes_num).cuda()
        self.relu = nn.ReLU().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        for name,param in self.lstm.named_parameters():
            print(name)
            print((param.size()))
            if 'bias_hh' in name:
                init_t = torch.Tensor([0 for _ in range(hidden_neurons)]+
                                     [1 for _ in range(hidden_neurons)]+
                                     [0 for _ in range(hidden_neurons)]+
                                     [0 for _ in range(hidden_neurons)]).cuda()
                keras_setter_(param, init_t)
            elif 'bias_ih' in name:
                torch.nn.init.zeros_(param)
            elif 'weight_ih' in name:
                torch.nn.init.kaiming_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)


    def forward(self, inputs_arrays, traj_lens):
        # pl_input = autograd.Variable(torch.LongTensor(inputs_arrays[0]),requires_grad =False).cuda()
        # time_input = autograd.Variable(torch.LongTensor(inputs_arrays[1]),requires_grad =False).cuda()
        # user_input = autograd.Variable(torch.LongTensor(inputs_arrays[2]),requires_grad =False).cuda()
        pl_input = torch.LongTensor(inputs_arrays).cuda()
        attrs_latent = self.place_embedding(pl_input)
        batch_size = inputs_arrays.shape[0]
        out, state = torch.zeros(batch_size, config.hidden_neurons).cuda(), \
                     torch.zeros(batch_size, config.hidden_neurons).cuda()
        time_stemp = inputs_arrays.shape[1]
        outputs = []
        for t in range(time_stemp):
            cell_input = attrs_latent[:, t, :]
            out, state = self.lstm(cell_input, (out, state))
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim = 1)
        # outputs = self.lstm(attrs_latent, (out, state))[0]
        if self.isAttention:
            # Attention Model
            atten_out = []
            length_out = [torch.stack([outputs[i][k][:] for k in range(j)]) for i, j in enumerate(traj_lens)]
            for h_stats in length_out:
                query = h_stats[-1,:].view(1, -1)
                hidden_size = query.size()[1]
                context = h_stats
                score = torch.matmul(query, context.view(hidden_size,-1))
                prob = F.softmax(score.view(1, -1), dim=1)
                out = torch.matmul(prob, context.view(-1, hidden_size)).squeeze(0)
                out_tensor = F.tanh(self.linear1(torch.cat([out, query.squeeze(0)], dim = 0)))
                atten_out.append(out_tensor)
            atten_out = torch.stack(atten_out)
            out_res = self.softmax(self.linear2(atten_out))
            return out_res
        else:
            last_out = outputs.contiguous().view(batch_size, -1)
            dens_out = self.relu(self.linear3(last_out))
            out_res = self.softmax(self.linear4(dens_out))
            return out_res

    def get_last_state(self, all_out, traj_lens):
        return [all_out[i, j, :] for i,j in enumerate(traj_lens)]




class ClassificationRNN2(Module):
    def __init__(self, place_dim, pl_d=config.pl_d, classes_num = 14,
                 hidden_neurons=config.hidden_neurons, number_layers = 1,
                 isAttention=False, biDirection = False):
        super(ClassificationRNN2, self).__init__()
        self.place_embedding = nn.Embedding(place_dim+1, pl_d, padding_idx=0).cuda()
        self.bi_direct = biDirection
        self.isAttention = isAttention
        self.number_layers = number_layers
        if self.bi_direct:
            self.lstm = nn.LSTM(input_size=pl_d, hidden_size=hidden_neurons,
                                num_layers=number_layers, dropout=0,
                                batch_first=True, bidirectional= self.bi_direct).cuda()
            self.linear1 = nn.Linear(hidden_neurons * 4, hidden_neurons).cuda()
            self.linear3 = nn.Linear(hidden_neurons * 300 * 2, 256).cuda()
        else:
            self.lstm = nn.LSTM(input_size=pl_d, hidden_size=hidden_neurons,
                                num_layers=number_layers, dropout=0,
                                batch_first=True,bidirectional= self.bi_direct).cuda()
            self.linear1 = nn.Linear(hidden_neurons * 2, hidden_neurons).cuda()
            self.linear3 = nn.Linear(hidden_neurons * 300, 256).cuda()
        self.linear2 = nn.Linear(hidden_neurons, classes_num).cuda()
        self.linear4 = nn.Linear(256, classes_num).cuda()
        self.relu = nn.ReLU().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        for name,param in self.lstm.named_parameters():
            print(name)
            print((param.size()))
            if 'bias_hh' in name:
                init_t = torch.Tensor([0 for _ in range(hidden_neurons)]+
                                     [1 for _ in range(hidden_neurons)]+
                                     [0 for _ in range(hidden_neurons)]+
                                     [0 for _ in range(hidden_neurons)]).cuda()
                keras_setter_(param, init_t)
            elif 'bias_ih' in name:
                torch.nn.init.zeros_(param)
            elif 'weight_ih' in name:
                torch.nn.init.kaiming_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)


    def forward(self, inputs_arrays, traj_lens):
        # pl_input = autograd.Variable(torch.LongTensor(inputs_arrays[0]),requires_grad =False).cuda()
        # time_input = autograd.Variable(torch.LongTensor(inputs_arrays[1]),requires_grad =False).cuda()
        # user_input = autograd.Variable(torch.LongTensor(inputs_arrays[2]),requires_grad =False).cuda()
        pl_input = torch.LongTensor(inputs_arrays).cuda()
        attrs_latent = self.place_embedding(pl_input)
        batch_size = inputs_arrays.shape[0]
        if self.bi_direct:
            out, state = torch.zeros(self.number_layers * 2, batch_size, config.hidden_neurons).cuda(), \
                         torch.zeros(self.number_layers * 2, batch_size, config.hidden_neurons).cuda()
        else:
            out, state = torch.zeros(self.number_layers, batch_size, config.hidden_neurons).cuda(), \
                         torch.zeros(self.number_layers, batch_size, config.hidden_neurons).cuda()
        outputs = self.lstm(attrs_latent, (out, state))[0]
        if self.isAttention:
            # Attention Model
            atten_out = []
            length_out = [torch.stack([outputs[i][k][:] for k in range(j)]) for i, j in enumerate(traj_lens)]
            for h_stats in length_out:
                query = h_stats[-1,:].view(1, -1)
                hidden_size = query.size()[1]
                context = h_stats
                score = torch.matmul(query, context.view(hidden_size,-1))
                prob = F.softmax(score.view(1, -1), dim=1)
                out = torch.matmul(prob, context.view(-1, hidden_size)).squeeze(0)
                out_tensor = F.tanh(self.linear1(torch.cat([out, query.squeeze(0)], dim = 0)))
                atten_out.append(out_tensor)
            atten_out = torch.stack(atten_out)
            out_res = self.softmax(self.linear2(atten_out))
            return out_res
        else:
            last_out = outputs.contiguous().view(batch_size, -1)
            dens_out = self.relu(self.linear3(last_out))
            out_res = self.softmax(self.linear4(dens_out))
            return out_res

    def get_last_state(self, all_out, traj_lens):
        return [all_out[i, j, :] for i,j in enumerate(traj_lens)]
