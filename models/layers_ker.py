import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd.function import Function
import pdb


class SparseBackward(Function):

    @staticmethod
    def forward(ctx, support, edge, spec_domain, sig, mu):

        ctx.save_for_backward(support, edge, spec_domain, sig, mu)

        diff = spec_domain[edge[0, :], :] - spec_domain[edge[1, :], :]
        # aa = torch.sqrt(torch.sum(diff.pow(2), 1)).mean()
        # bb = torch.sqrt(torch.sum(diff.pow(2), 1)).std()
        # pdb.set_trace()
        diff_mu = diff - mu.expand_as(diff)
        qq = -0.5 * torch.sum(diff_mu.pow(2), dim=1)
        value = torch.exp(sig * qq)
        shape = torch.Size((support.shape[0], support.shape[0]))
        phi = torch.sparse.FloatTensor(edge, value, shape)

        output = torch.sparse.mm(phi, support)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        support, edge, spec_domain, sig, mu = ctx.saved_tensors

        grad_support = grad_edge = grad_spec_domain = grad_sig = grad_mu = None

        diff = spec_domain[edge[0, :], :] - spec_domain[edge[1, :], :]
        qq = -0.5 * torch.sum((diff - mu.expand_as(diff)).pow(2), dim=1)
        values = torch.exp(sig * qq)
        shape = torch.Size((support.shape[0], support.shape[0]))
        phi = torch.sparse.FloatTensor(edge, values, shape)

        if ctx.needs_input_grad[0]:
            grad_support = torch.mm(phi.t(), grad_output)

        if ctx.needs_input_grad[3]:
            dir_values = values * qq
            dir_phi = torch.sparse.FloatTensor(edge, dir_values, shape)
            dir_psy = torch.mm(dir_phi, support)
            grad_sig = (grad_output * dir_psy).sum().unsqueeze(0)

        if ctx.needs_input_grad[4]:
            dmu = torch.FloatTensor(mu.shape[0]).cuda()
            for d in range(mu.shape[0]):
                dir_values = sig * ((diff[:, d] * values) - (mu[d] * values))
                dir_phi = torch.sparse.FloatTensor(edge, dir_values, shape)
                dir_psy = torch.mm(dir_phi, support)
                dmu[d] = (grad_output * dir_psy).sum()
            grad_mu = dmu

        return grad_support, grad_edge, grad_spec_domain, grad_sig, grad_mu


class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, emb_size, ker_size, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.emb_size = emb_size
        self.ker_size = ker_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features, ker_size))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.mu = Parameter(torch.FloatTensor(emb_size, ker_size))
        self.sig = Parameter(torch.FloatTensor(1, ker_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.mu.data.normal_(0, 0.005)
        # self.sig.data.normal_(0.0047, 0.00001).pow(-2)
        self.sig.data = self.sig.data.zero_() + 60140
        self.weight.data.normal_(0, 0.1)
        self.bias.data.normal_(0, 0.1)

    def forward(self, data, spec_domain):
        output = torch.autograd.Variable(torch.tensor((), dtype=torch.float32)).cuda()
        output = output.new_zeros((data.x.shape[0], self.out_features, self.ker_size))

        sp_fn = SparseBackward.apply

        for k in range(self.ker_size):
            support = torch.mm(data.x, self.weight[:, :, k])
            output[:, :, k] = sp_fn(support, data.edge_idx, spec_domain, self.sig[:, k], self.mu[:, k])

        return output.sum(2) + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
