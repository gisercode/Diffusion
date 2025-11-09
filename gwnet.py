import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
from tednet.tnn.tensor_ring import TRConv2D

def factorize(n):
    for i in range(int(n**0.5), 1, -1):
        if n % i == 0:
            return i, n // i
    return n, 1


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, tr_ranks=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        in_shape = factorize(c_in)
        out_shape = factorize(c_out)
        self.final_conv = TRConv2D(in_shape=in_shape, out_shape=out_shape, ranks=[tr_ranks]*(len(in_shape)+len(out_shape)+1), kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2,
                 apt_size=10, tr_ranks=2):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.addaptadj = addaptadj


        in_shape = factorize(in_dim)
        out_shape = factorize(residual_channels)
        self.start_conv = TRConv2D(in_shape=in_shape, out_shape=out_shape, ranks=[tr_ranks]*(len(in_shape)+len(out_shape)+1), kernel_size=(1, 1))

        self.fixed_supports = supports or []

        self.supports_len = len(self.fixed_supports)#2
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        depth = list(range(blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in depth]) #dilation_channels = 64,esidual_channels = 6
        self.skip_convs = ModuleList([Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList([GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len, tr_ranks=tr_ranks)
                                              for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            D = 1
            for i in range(layers):
                in_shape = factorize(residual_channels)
                out_shape = factorize(dilation_channels)
                ranks = [tr_ranks]*(len(in_shape)+len(out_shape)+1)
                self.filter_convs.append(TRConv2D(in_shape=in_shape, out_shape=out_shape, ranks=ranks, kernel_size=(1, kernel_size)))
                self.gate_convs.append(TRConv2D(in_shape=in_shape, out_shape=out_shape, ranks=ranks, kernel_size=(1, kernel_size)))
                D *= 2

        in_shape = factorize(skip_channels)
        out_shape = factorize(out_dim)
        self.end_conv = TRConv2D(in_shape=in_shape, out_shape=out_shape, ranks=[tr_ranks]*(len(in_shape)+len(out_shape)+1), kernel_size=(1, 1), bias=True)

    @staticmethod
    def svd_init(apt_size, aptinit):
        # Convert numpy array to torch tensor before SVD
        if not isinstance(aptinit, torch.Tensor):
            aptinit = torch.from_numpy(aptinit).float()
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, x):
        x = self.start_conv(x)
        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adjacency_matrices = self.fixed_supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            
            # Causal padding to preserve time dimension
            pad = (self.filter_convs[i].kernel_size[-1] - 1, 0)
            residual_padded = F.pad(residual, pad)

            filter = torch.tanh(self.filter_convs[i](residual_padded))
            gate = torch.sigmoid(self.gate_convs[i](residual_padded))
            x = filter * gate
            
            # parametrized skip connection
            s = self.skip_convs[i](x)
            if isinstance(skip, int):
                skip = s
            else:
                skip = skip + s

            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out
            else:
                x = self.residual_convs[i](x)
            
            x = x + residual
            x = self.bn[i](x)

        return self.end_conv(skip)