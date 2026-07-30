import torch
from torch import nn, optim
# import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import mul
from torch_sparse import sum as sparsesum

def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)
    scaled_inverted_row_sum = row_sum.pow_(-1)
    scaled_inverted_row_sum.masked_fill_(scaled_inverted_row_sum == float("inf"), 0.0)
    matrix = mul(adj, row_sum.view(-1, 1))

    return matrix

def col_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=0)
    scaled_inverted_col_sum = row_sum.pow_(-0.20)
    scaled_inverted_col_sum.masked_fill_(scaled_inverted_col_sum == float("inf"), 0.0)
    matrix = mul(adj, row_sum.view(1, -1))

    return matrix


def directed_norm(adj, exponent):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(exponent)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(exponent)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))

    return adj


def directed_norm_ones(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.add_self_loops
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 1.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 1.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))

    return adj


def norm_laplacian(adj):
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5) * (-1)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("-inf"), -1.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5) * (-1)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("-inf"), -1.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))

    row, col = torch.arange(adj.size(0)), torch.arange(adj.size(0))
    identity_matrix = SparseTensor(row=row, col=col, sparse_sizes=(adj.size(0), adj.size(0))).to(device=adj.device())

    return adj.add(identity_matrix)


def directed_opposite_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 1.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 1.0)

    adj = mul(adj, out_deg_inv_sqrt.view(1, -1))
    adj = mul(adj, in_deg_inv_sqrt.view(-1, 1))

    return adj


def get_norm_adj(adj, norm , exponent = -0.25):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "dir_ones":
        return directed_norm_ones(adj)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "col":
        return col_norm(adj)
    elif norm == "dir":
        return directed_norm(adj, exponent)
    elif norm == "norm_laplacian":
        return norm_laplacian(adj)
    elif norm == "opposite":
        return directed_opposite_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")



class ComplexFaberConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, K_plus=3, exponent=-0.25, weight_penalty='exp', zero_order=False):
        super(ComplexFaberConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K_plus = K_plus
        self.exponent = exponent
        self.weight_penalty = weight_penalty
        self.zero_order = zero_order

        if zero_order:
            # Zero Order Lins
            # Source to destination
            self.lin_real_src_to_dst_zero = Linear(input_dim, output_dim)
            self.lin_imag_src_to_dst_zero = Linear(input_dim, output_dim)

            # Destination to source
            self.lin_real_dst_to_src_zero = Linear(input_dim, output_dim)
            self.lin_imag_dst_to_src_zero = Linear(input_dim, output_dim)

        # Lins for positive powers:
        # Source to destination
        self.lins_real_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])  # real part

        self.lins_imag_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])  # imaginary part

        # Destination to source
        self.lins_real_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])  # real part
        self.lins_imag_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(K_plus)
        ])  # imaginary part
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x_real, x_imag, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x_real.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir", exponent=self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir", exponent=self.exponent)

        y_real = self.adj_norm @ x_real
        y_imag = self.adj_norm @ x_imag

        y_real_t = self.adj_t_norm @ x_real
        y_imag_t = self.adj_t_norm @ x_imag

        sum_real_src_to_dst = self.lins_real_src_to_dst[0](y_real) - self.lins_imag_src_to_dst[0](y_imag)
        sum_imag_src_to_dst = self.lins_imag_src_to_dst[0](y_real) + self.lins_real_src_to_dst[0](y_imag)
        sum_real_dst_to_src = self.lins_real_src_to_dst[0](y_real_t) - self.lins_imag_src_to_dst[0](y_imag_t)
        sum_imag_dst_to_src = self.lins_imag_src_to_dst[0](y_real) + self.lins_real_src_to_dst[0](y_imag_t)
        if self.zero_order:
            sum_real_src_to_dst = sum_real_src_to_dst + self.lin_real_src_to_dst_zero(
                x_real) - self.lin_imag_src_to_dst_zero(x_imag)
            sum_imag_src_to_dst = sum_imag_src_to_dst + self.lin_imag_src_to_dst_zero(
                x_real) + self.lin_real_src_to_dst_zero(x_imag)

            sum_real_dst_to_src = sum_real_dst_to_src + self.lin_real_dst_to_src_zero(
                x_real) - self.lin_imag_dst_to_src_zero(x_imag)
            sum_imag_dst_to_src = sum_imag_dst_to_src + self.lin_imag_dst_to_src_zero(
                x_real) + self.lin_real_dst_to_src_zero(x_imag)

        if self.K_plus > 1:
            if self.weight_penalty == 'exp':
                for i in range(1, self.K_plus):
                    y_real = self.adj_norm @ x_real
                    y_imag = self.adj_norm @ x_imag

                    y_real_t = self.adj_t_norm @ x_real
                    y_imag_t = self.adj_t_norm @ x_imag

                    sum_real_src_to_dst = sum_real_src_to_dst + (
                                self.lins_real_src_to_dst[i](y_real) - self.lins_imag_src_to_dst[i](y_imag)) / (2 ** i)
                    sum_imag_src_to_dst = sum_imag_src_to_dst + (
                                self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag)) / (2 ** i)

                    sum_real_dst_to_src = sum_real_dst_to_src + (
                                self.lins_real_src_to_dst[i](y_real_t) - self.lins_imag_src_to_dst[i](y_imag_t)) / (
                                                      2 ** i)
                    sum_imag_dst_to_src = sum_imag_dst_to_src + (
                                self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag_t)) / (
                                                      2 ** i)
            elif self.weight_penalty == 'lin':
                for i in range(1, self.K_plus):
                    y_real = self.adj_norm @ x_real
                    y_imag = self.adj_norm @ x_imag

                    y_real_t = self.adj_t_norm @ x_real
                    y_imag_t = self.adj_t_norm @ x_imag

                    sum_real_src_to_dst = sum_real_src_to_dst + (
                                self.lins_real_src_to_dst[i](y_real) - self.lins_imag_src_to_dst[i](y_imag)) / i
                    sum_imag_src_to_dst = sum_imag_src_to_dst + (
                                self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag)) / i

                    sum_real_dst_to_src = sum_real_dst_to_src + (
                                self.lins_real_src_to_dst[i](y_real_t) - self.lins_imag_src_to_dst[i](y_imag_t)) / i
                    sum_imag_dst_to_src = sum_imag_dst_to_src + (
                                self.lins_imag_src_to_dst[i](y_real) + self.lins_real_src_to_dst[i](y_imag_t)) / i
            else:
                raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")

        total_real = self.alpha * sum_real_src_to_dst + (1 - self.alpha) * sum_real_dst_to_src
        total_imag = self.alpha * sum_imag_src_to_dst + (1 - self.alpha) * sum_imag_dst_to_src

        return total_real, total_imag


class HoloGNN(torch.nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            hidden_dim,
            num_layers=2,
            dropout=0,
            conv_type="complex-fabernet",
            jumping_knowledge=False,
            normalize=False,
            alpha=1 / 2,
            learn_alpha=False,
            K_plus=3,
            exponent=-0.25,
            weight_penalty='exp',
            lrelu_slope=-1.0,
            zero_order=False,
    ):
        super(HoloGNN, self).__init__()
        self.conv_type = conv_type
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        self.lrelu_slope = lrelu_slope

        output_dim = hidden_dim if jumping_knowledge else num_classes

        self.convs = ModuleList([ComplexFaberConv(num_features, hidden_dim, alpha=self.alpha, K_plus=K_plus,
                                          zero_order=zero_order, exponent=exponent, weight_penalty=weight_penalty)])
        for _ in range(num_layers - 2):
            self.convs.append(
                ComplexFaberConv(hidden_dim, hidden_dim, alpha=self.alpha, K_plus=K_plus, zero_order=zero_order,
                         exponent=exponent, weight_penalty=weight_penalty))
        self.convs.append(
            ComplexFaberConv(hidden_dim, output_dim, alpha=self.alpha, K_plus=K_plus, zero_order=zero_order,
                     exponent=exponent, weight_penalty=weight_penalty))

        if jumping_knowledge is not None:
            if self.conv_type == "complex-fabernet":
                input_dim = 2 * hidden_dim * num_layers if jumping_knowledge == "cat" else 2 * hidden_dim
            else:
                input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        if self.conv_type == "complex-fabernet":
            x_real = x

            x_imag = torch.zeros_like(x)

            xs = []
            for i, conv in enumerate(self.convs):
                x_real, x_imag = conv(x_real, x_imag, edge_index)
                if i != len(self.convs) - 1 or self.jumping_knowledge:
                    x_real = F.leaky_relu(x_real, negative_slope=self.lrelu_slope)
                    x_imag = F.leaky_relu(x_imag, negative_slope=self.lrelu_slope)

                    x_real = F.dropout(x_real, p=self.dropout, training=self.training)
                    x_imag = F.dropout(x_imag, p=self.dropout, training=self.training)
                    if self.normalize:
                        x_real = F.normalize(x_real, p=2, dim=1)
                        x_imag = F.normalize(x_imag, p=2, dim=1)

                xs += [torch.cat((x_real, x_imag), 1)]
            x = torch.cat((x_real, x_imag), 1)

            if self.jumping_knowledge is not None:
                x = self.jump(xs)
                x = self.lin(x)

            logits = torch.nn.functional.log_softmax(x, dim=1)

            prediction = torch.argmax(logits, dim=1)
            return logits, prediction
        else:
            xs = []
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i != len(self.convs) - 1 or self.jumping_knowledge:
                    x = F.leaky_relu(x, negative_slope=self.lrelu_slope)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    if self.normalize:
                        x = F.normalize(x, p=2, dim=1)
                xs += [x]

            if self.jumping_knowledge is not None:
                x = self.jump(xs)
                x = self.lin(x)

            logits = torch.nn.functional.log_softmax(x, dim=1)

            prediction = torch.argmax(logits, dim=1)
            return logits, prediction













