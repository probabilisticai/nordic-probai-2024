from torch import nn
import torch
from .utils import center_zero


class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

    def edge_model(self, source, target, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp(out)

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
    ):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], edge_attr)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
    ):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr=None,
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr)
        return coord


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            ),
        )
        self.to(self.device)

    def forward(self, h, x, edge_index, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        dist_sq, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        edge_attr = torch.cat([dist_sq, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr,
            )
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr)
        return h, x


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=3,
        norm_diff=True,
        out_node_nf=None,
        tanh=True,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=1,
        normalization_factor=1,
        aggregation_method="sum",
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    device=device,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.to(self.device)

    def forward(self, h, x, edge_index):
        # Edit Emiel: Remove velocity as input
        distances_sq, _ = coord2diff(x, edge_index)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                edge_attr=distances_sq,
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        return h, x


class GNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        aggregation_method="sum",
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        normalization_factor=1,
        out_node_nf=None,
    ):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                ),
            )
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        h = self.embedding_out(h)

        return h


class EGNNScore(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        n_layers=3,
        out_node_nf=None,
        condition: bool = False,
    ):
        super(EGNNScore, self).__init__()
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            n_layers=n_layers,
        )
        self.condition = condition

    def forward(self, z_t, t, edge_index, batch, context=None):
        x_in = z_t[:, 0:3]
        x_in = center_zero(x_in, batch)
        h_t = z_t[:, 3:]
        h = torch.concatenate([h_t, t.unsqueeze(1)], dim=-1)

        if self.condition and context is not None:
            context = context[batch]
            h = torch.concatenate([h, context.unsqueeze(1)], dim=-1)
        elif self.condition and context is None:
            raise Exception("Expecting context to condition but it is set to None")

        h, x = self.egnn(h, x_in, edge_index)
        x = center_zero(x, batch)

        x_out = x - x_in
        output = torch.concatenate([x_out, h], dim=-1)
        return output


class GNNScore(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        n_layers=3,
        out_node_nf=None,
    ):
        super(GNNScore, self).__init__()
        self.gnn = GNN(
            in_node_nf=in_node_nf,
            in_edge_nf=0,
            hidden_nf=hidden_nf,
            out_node_nf=out_node_nf,
            n_layers=n_layers,
        )

    def forward(self, z_t, t, edge_index, batch, context=None):
        zt_t = torch.concatenate([z_t, t.unsqueeze(1)], dim=-1)

        output = self.gnn(zt_t, edge_index)

        return output


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
