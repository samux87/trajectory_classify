from torch.nn import Module, Parameter
import config
import torch
import torch.nn.functional as F
import math


def _hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    return x.clamp(0., 1.)

class RNNCellBase(Module):

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** of shape `(batch, hidden_size)`: tensor containing the initial cell state
          for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: h_1, c_1
        - **h_1** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch
        - **c_1** of shape `(batch, hidden_size)`: tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self.custom_lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


    def custom_lstm_cell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hx, cx = hidden
        gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = _hard_sigmoid(ingate)
        forgetgate = _hard_sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = _hard_sigmoid(outgate)
        # spatialgate = F.sigmoid(spatialgate)
        cy_h = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy_h)

        return hy, cy_h

class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)
        return self.coustom_gru_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

    def coustom_gru_cell(self, input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        resetgate = F.sigmoid(i_r + h_r)
        updategate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hyy = (1- updategate)*newgate + updategate*hidden
        return hyy