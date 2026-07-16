"""Hypergraph message passing for congestion prediction."""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class Hypergraph:

    def __init__(self):
        self.net_pins: Dict[str, list] = {}
        self.net_bbox: Dict[str, Tuple] = {}
        self.gcell_nets: Dict[Tuple, Set[str]] = defaultdict(set)
        self.net_slack: Dict[str, float] = {}
        self.grid_shape: Tuple[int, int, int] = (0, 0, 0)

    @classmethod
    def from_netlist(cls, data_net: dict, data_cap: dict) -> 'Hypergraph':
        hg = cls()
        hg.grid_shape = (data_cap['nLayers'], data_cap['ySize'], data_cap['xSize'])

        for net_name, net_data in data_net.items():
            pins, min_slack = [], float('inf')
            min_x = min_y = 10**9
            max_x = max_y = -10**9

            for pin_info in net_data:
                if len(pin_info) == 3:
                    _, slack, coords = pin_info
                    slack = float(slack)
                    min_slack = min(min_slack, slack)
                    if coords:
                        z, x, y = coords[0]
                        pins.append((z, x, y, slack))
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                        hg.gcell_nets[(z, y, x)].add(net_name)
                else:
                    z, x, y = pin_info[0]
                    pins.append((z, x, y, 0.0))
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    hg.gcell_nets[(z, y, x)].add(net_name)

            if pins:
                hg.net_pins[net_name] = pins
                hg.net_bbox[net_name] = (min_x, min_y, max_x, max_y)
                hg.net_slack[net_name] = min_slack if min_slack < float('inf') else 0.0

        return hg

    @property
    def n_nets(self): return len(self.net_pins)

    @property
    def n_pins(self): return sum(len(p) for p in self.net_pins.values())


class HypergraphMessagePassing:

    def __init__(self, hypergraph: 'Hypergraph', n_hops: int = 2, hidden_dim: int = 4):
        self.hg = hypergraph
        self.n_hops = n_hops
        self.hidden_dim = hidden_dim


        np.random.seed(42)
        self.W_n2e = np.random.randn(3, hidden_dim).astype(np.float32) / np.sqrt(3)
        self.a_n2e = np.random.randn(hidden_dim * 2, 1).astype(np.float32) / np.sqrt(hidden_dim * 2)

        self.W_e2n = np.random.randn(hidden_dim, 3).astype(np.float32) / np.sqrt(hidden_dim)
        self.a_e2n = np.random.randn(hidden_dim + 3, 1).astype(np.float32) / np.sqrt(hidden_dim + 3)

        self._preindex()

    def _preindex(self):
        net_names = list(self.hg.net_pins.keys())
        self._net_name_list = net_names
        self._net_idx_map = {n: i for i, n in enumerate(net_names)}
        self._n_nets = len(net_names)
        nL, H, W = self.hg.grid_shape

        pin_net, pin_z, pin_y, pin_x, pin_slack = [], [], [], [], []
        for net_name in net_names:
            ni = self._net_idx_map[net_name]
            slack = self.hg.net_slack.get(net_name, 0.0)

            for z, x, y, _ in self.hg.net_pins[net_name]:
                if 0 <= z < nL and 0 <= y < H and 0 <= x < W:
                    pin_net.append(ni)
                    pin_z.append(z)
                    pin_y.append(y)
                    pin_x.append(x)
                    pin_slack.append(slack)

        self._pin_net = np.array(pin_net, dtype=np.int32)
        self._pin_z = np.array(pin_z, dtype=np.int32)
        self._pin_y = np.array(pin_y, dtype=np.int32)
        self._pin_x = np.array(pin_x, dtype=np.int32)
        self._pin_slack = np.array(pin_slack, dtype=np.float32)
        self._pin_linear = self._pin_z * H * W + self._pin_y * W + self._pin_x

    def _leaky_relu(self, x, alpha=0.2):
        return np.maximum(alpha * x, x)

    def _softmax_scatter(self, src, indices, num_elements):
        max_val = np.zeros(num_elements, dtype=src.dtype)
        np.maximum.at(max_val, indices, src)
        shifted = src - max_val[indices]
        exp_val = np.exp(shifted)
        sum_exp = np.zeros(num_elements, dtype=src.dtype)
        np.add.at(sum_exp, indices, exp_val)
        return exp_val / (sum_exp[indices] + 1e-9)

    def forward(self, cap_orig: np.ndarray, cap_cur: np.ndarray, layer_dirs: list) -> np.ndarray:
        nL, H, W = self.hg.grid_shape


        util = np.clip(1.0 - cap_cur / (cap_orig + 1e-6), 0, 2)
        overflow = np.maximum(-cap_cur, 0)
        of_max = overflow.max()
        of_norm = overflow / (of_max + 1e-6) if of_max > 0 else overflow
        cap_remain = np.clip(cap_cur / (cap_orig.max() + 1e-6), -1, 1)

        X_node = np.stack([util, of_norm, cap_remain], axis=-1)
        flat_size = nL * H * W

        for _ in range(self.n_hops):

            X_pin = X_node[self._pin_z, self._pin_y, self._pin_x]
            H_pin = X_pin @ self.W_n2e


            net_context_slack = np.expand_dims(self._pin_slack, axis=-1)
            attention_input_n2e = np.concatenate([H_pin, np.tile(net_context_slack, (1, self.hidden_dim))], axis=1)
            e_n2e = self._leaky_relu(attention_input_n2e @ self.a_n2e).squeeze()
            alpha_n2e = self._softmax_scatter(e_n2e, self._pin_net, self._n_nets)


            Weighted_H_pin = H_pin * alpha_n2e[:, None]
            E_net = np.zeros((self._n_nets, self.hidden_dim), dtype=np.float32)
            np.add.at(E_net, self._pin_net, Weighted_H_pin)


            E_pin_spread = E_net[self._pin_net]
            X_update = E_pin_spread @ self.W_e2n


            attention_input_e2n = np.concatenate([X_update, E_pin_spread], axis=1)
            e_e2n = self._leaky_relu(attention_input_e2n @ self.a_e2n).squeeze()
            alpha_e2n = self._softmax_scatter(e_e2n, self._pin_linear, flat_size)


            Weighted_X_update = X_update * alpha_e2n[:, None]
            X_new_flat = np.zeros((flat_size, 3), dtype=np.float32)
            np.add.at(X_new_flat, self._pin_linear, Weighted_X_update)

            X_new = X_new_flat.reshape(nL, H, W, 3)


            X_node = 0.8 * X_node + 0.2 * np.tanh(X_new)


        output_score = np.clip(0.6 * X_node[:, :, :, 0] + 0.4 * X_node[:, :, :, 1], 0, 1)
        return output_score

    def get_net_congestion(self, cap_cur: np.ndarray) -> dict:
        overflow = np.maximum(-cap_cur, 0)
        pin_of = overflow[self._pin_z, self._pin_y, self._pin_x]
        net_scores = np.zeros(self._n_nets, dtype=np.float32)
        np.add.at(net_scores, self._pin_net, pin_of)
        return {name: float(net_scores[i]) for i, name in enumerate(self._net_name_list) if net_scores[i] > 0}

class Partition:
    def __init__(self, pid, x_lo, y_lo, x_hi, y_hi):
        self.id = pid
        self.x_lo, self.y_lo = x_lo, y_lo
        self.x_hi, self.y_hi = x_hi, y_hi
        self.nets: List[str] = []
        self.cut_nets: Set[str] = set()

    def contains(self, x, y):
        return self.x_lo <= x <= self.x_hi and self.y_lo <= y <= self.y_hi

    def __repr__(self):
        return (f'P{self.id}[{self.x_lo},{self.y_lo}]-[{self.x_hi},{self.y_hi}] '
                f'nets={len(self.nets)} cuts={len(self.cut_nets)}')


class SpatialPartitioner:
    @staticmethod
    def partition(hg: Hypergraph, n_parts: int = 4):
        _, H, W = hg.grid_shape
        regions = [(0, 0, W - 1, H - 1)]
        split_x = True
        while len(regions) < n_parts:
            new = []
            for x0, y0, x1, y1 in regions:
                if split_x:
                    mid = (x0 + x1) // 2
                    new.append((x0, y0, mid, y1))
                    new.append((mid + 1, y0, x1, y1))
                else:
                    mid = (y0 + y1) // 2
                    new.append((x0, y0, x1, mid))
                    new.append((x0, mid + 1, x1, y1))
            regions = new
            split_x = not split_x

        partitions = [Partition(i, *r) for i, r in enumerate(regions)]
        all_cut = set()
        for net_name, bbox in hg.net_bbox.items():
            nx0, ny0, nx1, ny1 = bbox
            overlap = [p for p in partitions
                       if nx0 <= p.x_hi and nx1 >= p.x_lo
                       and ny0 <= p.y_hi and ny1 >= p.y_lo]
            if len(overlap) == 1:
                overlap[0].nets.append(net_name)
            elif overlap:
                all_cut.add(net_name)
                pins = hg.net_pins[net_name]
                best = max(overlap, key=lambda p: sum(
                    1 for z, x, y, s in pins if p.contains(x, y)))
                best.nets.append(net_name)
                for p in overlap:
                    p.cut_nets.add(net_name)
        return partitions, all_cut
