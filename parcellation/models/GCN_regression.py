import torch
import torch.nn as nn
from parcellation.models.layers_ker import GraphConvolution


class GCN(nn.Module):
    def __init__(self, feat, hid1, hid2, hid3, par, emb_size, ker_size):
        super(GCN, self).__init__()

        self.gc1_1 = GraphConvolution(feat, hid1, emb_size, ker_size)
        self.gc1_2 = nn.LeakyReLU(0.01)

        self.gc2_1 = GraphConvolution(feat + hid1, hid2, emb_size, ker_size)
        self.gc2_2 = nn.LeakyReLU(0.01)

        self.gc3_1 = GraphConvolution(feat + hid1 + hid2, hid3, emb_size, ker_size)
        self.gc3_2 = nn.LeakyReLU(0.01)

        self.gc4_1 = GraphConvolution(feat + hid1 + hid2 + hid3, par, emb_size, ker_size)
        self.gc4_2 = nn.LeakyReLU(0.01)

        self.gcn_poo = nn.AdaptiveAvgPool2d((1, par))

        self.lin1 = torch.nn.Linear(par, 16)
        self.lin1act = nn.LeakyReLU(0.01)

        self.lin2 = torch.nn.Linear(16, 8)
        self.lin2act = nn.LeakyReLU(0.01)

        self.lin3 = torch.nn.Linear(8, 1)
        self.lin3act = nn.LeakyReLU()

    def forward(self, data):
        spec_domain = data.x[:, :3]

        x1 = self.gc1_1(data, spec_domain)
        x1 = self.gc1_2(x1)
        data.x = torch.cat((x1, data.x), 1)

        x2 = self.gc2_1(data, spec_domain)
        x2 = self.gc2_2(x2)
        data.x = torch.cat((x2, data.x), 1)

        x3 = self.gc3_1(data, spec_domain)
        x3 = self.gc3_2(x3)
        data.x = torch.cat((x3, data.x), 1)

        x4 = self.gc4_1(data, spec_domain)
        x4 = self.gc4_2(x4)

        x = self.gcn_poo(x4.unsqueeze(0))

        x = self.lin1(x.squeeze())
        x = self.lin1act(x)
        x = self.lin2(x)
        x = self.lin2act(x)
        x = self.lin3(x)

        return x