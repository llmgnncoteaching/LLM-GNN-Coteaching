from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F

class GCN2(nn.Module):
	def __init__(self, input_dim, hidden_dim, out_dim):
		super(GCN2, self).__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, out_dim)
		self.softmax = nn.Softmax()

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		logits = self.softmax(x)
		preds = torch.argmax(logits, dim=1)
		return logits, preds, x
        # return logits, preds, r_final


class GCN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
				 dropout):
		super(GCN, self).__init__()
		self.softmax = nn.Softmax()

		self.convs = torch.nn.ModuleList()
		self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
		self.bns = torch.nn.ModuleList()
		self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
		for _ in range(num_layers - 2):
			self.convs.append(
				GCNConv(hidden_channels, hidden_channels, cached=True))
			self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
		self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

		self.dropout = dropout


	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		for bn in self.bns:
			bn.reset_parameters()


	def forward(self, x, adj_t):
		for i, conv in enumerate(self.convs[:-1]):
			x = conv(x, adj_t)
			x = self.bns[i](x)
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.convs[-1](x, adj_t)
		logits = self.softmax(x)
		preds = torch.argmax(logits, dim=1)
		return logits, preds, x
