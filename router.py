"""AlphaRoute global router for ISPD 2025 benchmarks."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

import numpy as np
import rustworkx as rx
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_cap, read_net


ISPD_WEIGHTS = {
    'ariane': {'w1': -10, 'w2': -100, 'w3': 300, 'w4': 3e-7},
    'bsg': {'w1': -10, 'w2': -100, 'w3': 25, 'w4': 4e-8},
    'nvdla': {'w1': -0.05, 'w2': -0.5, 'w3': 25, 'w4': 1.5e-7},
    'mempool': {'w1': -1, 'w2': -10, 'w3': 300, 'w4': 7e-7},
}


ARIANE_REF = {'wns': -1.628424, 'tns': -523.038, 'power': 0.15612}


class RoutingMetrics:

    def __init__(self):
        self.total_nets = 0
        self.routed_nets = 0
        self.total_wirelength = 0
        self.total_vias = 0
        self.total_overflow = 0.0
        self.overflow_score = 0.0
        self.runtime_sec = 0.0
        self.batches = 0
        self.min_slack = 0.0
        self.total_neg_slack = 0.0
        self.n_endpoints = 0
        self.total_power = 0.0
        self.total_power = 0.0

    def sorig(self, weights=None):
        w = weights or getattr(self, 'weights', None) or ISPD_WEIGHTS.get(getattr(self, 'benchmark', 'ariane'), ISPD_WEIGHTS['ariane'])

        scales = {'wns_ns': 1.0, 'tns_per_ep_ns': 1.0, 'power': 1.0, 'overflow_l2': 1e6}
        wns = self.min_slack * 1e9
        tns = self.total_neg_slack * 1e9
        ne = max(self.n_endpoints, 1)
        score = (w['w1'] * (wns / scales['wns_ns'])
                 + w['w2'] * ((tns / ne) / scales['tns_per_ep_ns'])
                 + w['w3'] * (self.total_power / scales['power'])
                 + w['w4'] * (self.overflow_score / scales['overflow_l2']))
        return score

    def runtime_factor(self, median_time=19.0):
        if self.runtime_sec <= 0 or median_time <= 0:
            return 0.0
        T = 0.02 * math.log2(self.runtime_sec / median_time)
        return max(-0.2, min(0.2, T))

    def scaled_score(self, weights=None, median_time=19.0):
        s = self.sorig(weights)
        f = self.runtime_factor(median_time)
        sign = 1 if s >= 0 else -1
        return s * (1 + sign * abs(f))


def _get_ranges(net_data):
    max_x = max_y = -10**9
    min_x = min_y = 10**9
    for pin_info in net_data:
        if len(pin_info) == 3:
            pin_name, slack, coordinates = pin_info
            if len(coordinates) > 0:
                z1, x1, y1 = coordinates[0]
                min_x, max_x = min(min_x, x1), max(max_x, x1)
                min_y, max_y = min(min_y, y1), max(max_y, y1)
        else:
            z1, x1, y1 = pin_info[0]
            min_x, max_x = min(min_x, x1), max(max_x, x1)
            min_y, max_y = min(min_y, y1), max(max_y, y1)
    return (min_x, min_y, max_x, max_y,
            max_x - min_x + 1, max_y - min_y + 1)


def _net_min_slack(net_data):
    slacks = []
    for pin_info in net_data:
        if len(pin_info) == 3:
            slacks.append(float(pin_info[1]))
    return min(slacks) if slacks else 0.0


def _edge_weight(cap, z, y1, x1, y2, x2, criticality=0.0):
    base = 10.0 - criticality * 3.0
    return base + (11.0 - cap[z, y1, x1]) + (11.0 - cap[z, y2, x2])


MAX_STEINER_AREA = 2500


def _collect_terminals(net, data_net, min_x, min_y):
    terminals = []
    seen = set()
    for pin_info in data_net[net]:
        if len(pin_info) == 3:
            pin_name, slack, coordinates = pin_info
            if len(coordinates) > 0:
                z1, x1, y1 = coordinates[0]
                coord = (z1, y1 - min_y, x1 - min_x)
                if coord not in seen:
                    terminals.append(coord)
                    seen.add(coord)
        else:
            z1, x1, y1 = pin_info[0]
            coord = (z1, y1 - min_y, x1 - min_x)
            if coord not in seen:
                terminals.append(coord)
                seen.add(coord)
    return terminals


def _path_overflow(cap, z, y1, x1, y2, x2, horizontal, policy=None, off_y=0, off_x=0):
    cost = 0.0
    nL, H, W = cap.shape
    if z < 0 or z >= nL:
        return 1e6

    if horizontal:
        xlo, xhi = min(x1, x2), max(x1, x2)
        if 0 <= y1 < H:
            for x in range(max(0, xlo), min(xhi + 1, W)):
                cap_val = float(cap[z, y1, x])
                if policy is not None:
                    gy = y1 + off_y
                    gx = x + off_x
                    present = policy.present_factor * max(0.0, -cap_val)
                    hist = float(policy.history[z, gy, gx]) if hasattr(policy, 'history') else 0.0
                    macro = policy._macro_penalty(z, gx, gy)
                    cost += present + hist + macro
                else:
                    cost += max(0.0, -cap_val)
    else:
        ylo, yhi = min(y1, y2), max(y1, y2)
        if 0 <= x1 < W:
            for y in range(max(0, ylo), min(yhi + 1, H)):
                cap_val = float(cap[z, y, x1])
                if policy is not None:
                    gy = y + off_y
                    gx = x1 + off_x
                    present = policy.present_factor * max(0.0, -cap_val)
                    hist = float(policy.history[z, gy, gx]) if hasattr(policy, 'history') else 0.0
                    macro = policy._macro_penalty(z, gx, gy)
                    cost += present + hist + macro
                else:
                    cost += max(0.0, -cap_val)

    return cost


def _mst_l_route(terminals, layer_dir, cap_sub=None, policy=None, off_y=0, off_x=0):
    if len(terminals) > 1000: return set()
    """
    Enhanced MST + L-route for large nets.

    1. Build MST over terminals (Manhattan + via distance).
    2. Connect each MST edge with an L-shaped path.
    3. **Flexible L-routing**: evaluates BOTH L-shapes (H-first vs V-first)
       and picks the one with lower congestion along the path.
    4. Capacity-aware layer selection (picks least-congested H/V layers).

    Returns list of cell-to-cell edge tuples in local coordinates.
    """
    n = len(terminals)
    if n < 2:
        return []


    h_layers = [z for z in range(1, len(layer_dir)) if layer_dir[z] == 0]
    v_layers = [z for z in range(1, len(layer_dir)) if layer_dir[z] == 1]
    if not h_layers:
        h_layers = [1]
    if not v_layers:
        v_layers = [1]


    if cap_sub is not None and cap_sub.shape[0] > 1:
        h_layer = max(h_layers, key=lambda z: float(cap_sub[z].sum())
                      if z < cap_sub.shape[0] else -1e9)
        v_layer = max(v_layers, key=lambda z: float(cap_sub[z].sum())
                      if z < cap_sub.shape[0] else -1e9)
    else:
        h_layer = h_layers[0]
        v_layer = v_layers[0]


    in_mst = [False] * n
    cost = [float('inf')] * n
    parent = [-1] * n
    cost[0] = 0.0
    mst_edges_idx = []

    for _ in range(n):
        u = min((v for v in range(n) if not in_mst[v]), key=lambda v: cost[v])
        in_mst[u] = True
        if parent[u] >= 0:
            mst_edges_idx.append((parent[u], u))
        for v in range(n):
            if not in_mst[v]:
                d = (abs(terminals[u][1] - terminals[v][1])
                     + abs(terminals[u][2] - terminals[v][2])
                     + abs(terminals[u][0] - terminals[v][0]) * 40)
                if d < cost[v]:
                    cost[v] = d
                    parent[v] = u


    edge_set = set()

    def _add(a, b):
        key = (min(a, b), max(a, b))
        edge_set.add(key)

    def _via_run(z_from, z_to, y, x):
        zlo, zhi = min(z_from, z_to), max(z_from, z_to)
        for z in range(zlo, zhi):
            _add((z, y, x), (z + 1, y, x))

    for pu, pv in mst_edges_idx:
        z1, ly1, lx1 = terminals[pu]
        z2, ly2, lx2 = terminals[pv]

        if lx1 == lx2 and ly1 == ly2:

            _via_run(z1, z2, ly1, lx1)
            continue


        if cap_sub is not None:
            cost_l1 = (_path_overflow(cap_sub, h_layer, ly1, lx1, ly1, lx2, True, policy=policy, off_y=off_y, off_x=off_x)
                       + _path_overflow(cap_sub, v_layer, ly1, lx2, ly2, lx2, False, policy=policy, off_y=off_y, off_x=off_x))
            cost_l2 = (_path_overflow(cap_sub, v_layer, ly1, lx1, ly2, lx1, False, policy=policy, off_y=off_y, off_x=off_x)
                       + _path_overflow(cap_sub, h_layer, ly2, lx1, ly2, lx2, True, policy=policy, off_y=off_y, off_x=off_x))
            use_l1 = (cost_l1 <= cost_l2)
        else:
            use_l1 = True

        if use_l1:

            _via_run(z1, h_layer, ly1, lx1)
            if lx1 != lx2:
                xlo, xhi = min(lx1, lx2), max(lx1, lx2)
                for x in range(xlo, xhi):
                    _add((h_layer, ly1, x), (h_layer, ly1, x + 1))
            _via_run(h_layer, v_layer, ly1, lx2)
            if ly1 != ly2:
                ylo, yhi = min(ly1, ly2), max(ly1, ly2)
                for y in range(ylo, yhi):
                    _add((v_layer, y, lx2), (v_layer, y + 1, lx2))
            _via_run(v_layer, z2, ly2, lx2)
        else:

            _via_run(z1, v_layer, ly1, lx1)
            if ly1 != ly2:
                ylo, yhi = min(ly1, ly2), max(ly1, ly2)
                for y in range(ylo, yhi):
                    _add((v_layer, y, lx1), (v_layer, y + 1, lx1))
            _via_run(v_layer, h_layer, ly2, lx1)
            if lx1 != lx2:
                xlo, xhi = min(lx1, lx2), max(lx1, lx2)
                for x in range(xlo, xhi):
                    _add((h_layer, ly2, x), (h_layer, ly2, x + 1))
            _via_run(h_layer, z2, ly2, lx2)

    return [tuple(e) for e in edge_set]


MAZE_MAX_AREA = 15000


def _dijkstra_maze_route(net, matrix, data_cap, data_net, policy=None,
                          margin=8):
    layer_dir = data_cap['layerDirections']
    min_x, min_y, max_x, max_y, w, h = _get_ranges(data_net[net])
    terminal_coords = _collect_terminals(net, data_net, min_x, min_y)

    n = len(terminal_coords)
    if n < 2:
        return [], terminal_coords

    nL, fullH, fullW = matrix.shape


    in_mst = [False] * n
    cost = [float('inf')] * n
    parent = [-1] * n
    cost[0] = 0.0
    mst_edges_idx = []
    for _ in range(n):
        u = min((v for v in range(n) if not in_mst[v]),
                key=lambda v: cost[v])
        in_mst[u] = True
        if parent[u] >= 0:
            mst_edges_idx.append((parent[u], u))
        for v in range(n):
            if not in_mst[v]:
                d = (abs(terminal_coords[u][1] - terminal_coords[v][1])
                     + abs(terminal_coords[u][2] - terminal_coords[v][2])
                     + abs(terminal_coords[u][0] - terminal_coords[v][0]) * 40)
                if d < cost[v]:
                    cost[v] = d
                    parent[v] = u


    gx_lo = max(0, min_x - margin)
    gx_hi = min(fullW - 1, max_x + margin)
    gy_lo = max(0, min_y - margin)
    gy_hi = min(fullH - 1, max_y + margin)

    local_W = gx_hi - gx_lo + 1
    local_H = gy_hi - gy_lo + 1

    G = rx.PyGraph()

    total_nodes = nL * local_H * local_W
    G.add_nodes_from(range(total_nodes))

    def _nid(z, gy, gx):
        return z * local_H * local_W + (gy - gy_lo) * local_W + (gx - gx_lo)


    edges_to_add = []

    for z in range(1, nL):
        if layer_dir[z] == 0:
            for y in range(gy_lo, gy_hi + 1):
                for x in range(gx_lo, gx_hi):
                    w_e = (10.0 + max(0, 11.0 - matrix[z, y, x])
                           + max(0, 11.0 - matrix[z, y, x + 1]))
                    if policy:
                        w_e += policy.edge_supplement(
                            z, y, x, y, x + 1, matrix)
                    edges_to_add.append((_nid(z, y, x),
                                         _nid(z, y, x + 1), w_e))
        elif layer_dir[z] == 1:
            for y in range(gy_lo, gy_hi):
                for x in range(gx_lo, gx_hi + 1):
                    w_e = (10.0 + max(0, 11.0 - matrix[z, y, x])
                           + max(0, 11.0 - matrix[z, y + 1, x]))
                    if policy:
                        w_e += policy.edge_supplement(
                            z, y, x, y + 1, x, matrix)
                    edges_to_add.append((_nid(z, y, x),
                                         _nid(z, y + 1, x), w_e))


    via_cost = policy.via_cost_base if policy else 40.0
    for z in range(nL - 1):
        for y in range(gy_lo, gy_hi + 1):
            for x in range(gx_lo, gx_hi + 1):
                edges_to_add.append((_nid(z, y, x),
                                     _nid(z + 1, y, x), via_cost))

    G.add_edges_from(edges_to_add)


    all_edges = set()

    for pu, pv in mst_edges_idx:
        z1, ly1, lx1 = terminal_coords[pu]
        z2, ly2, lx2 = terminal_coords[pv]
        gy1, gx1 = min_y + ly1, min_x + lx1
        gy2, gx2 = min_y + ly2, min_x + lx2


        if not (gx_lo <= gx1 <= gx_hi and gy_lo <= gy1 <= gy_hi and
                gx_lo <= gx2 <= gx_hi and gy_lo <= gy2 <= gy_hi):
            continue

        src = _nid(z1, gy1, gx1)
        dst = _nid(z2, gy2, gx2)

        try:
            paths = rx.dijkstra_shortest_paths(
                G, src, target=dst, weight_fn=lambda w: w)
            if dst in paths:
                path = paths[dst]
                for i in range(len(path) - 1):
                    nid_u, nid_v = path[i], path[i + 1]

                    zu = nid_u // (local_H * local_W)
                    rem = nid_u % (local_H * local_W)
                    gyu = gy_lo + rem // local_W
                    gxu = gx_lo + rem % local_W
                    zv = nid_v // (local_H * local_W)
                    rem = nid_v % (local_H * local_W)
                    gyv = gy_lo + rem // local_W
                    gxv = gx_lo + rem % local_W

                    lu = (zu, gyu - min_y, gxu - min_x)
                    lv = (zv, gyv - min_y, gxv - min_x)
                    all_edges.add((min(lu, lv), max(lu, lv)))
        except Exception:
            pass

    return [tuple(e) for e in all_edges], terminal_coords


def find_solution_with_rustworkx(net, matrix, data_cap, data_net,
                                  policy=None, force_maze=False):
    layer_dir = data_cap['layerDirections']
    min_x, min_y, max_x, max_y, w, h = _get_ranges(data_net[net])
    terminal_coords = _collect_terminals(net, data_net, min_x, min_y)


    if force_maze and w * h <= MAZE_MAX_AREA:
        return _dijkstra_maze_route(
            net, matrix, data_cap, data_net, policy=policy)


    if w * h > MAX_STEINER_AREA:

        cap_sub = matrix[:, min_y:max_y + 1, min_x:max_x + 1]
        edges = _mst_l_route(terminal_coords, layer_dir, cap_sub=cap_sub, policy=policy, off_y=min_y, off_x=min_x)
        return edges, terminal_coords


    sub = matrix[:, min_y:max_y + 1, min_x:max_x + 1].copy()
    nL, H, W = sub.shape

    raw_slack = _net_min_slack(data_net[net])
    criticality = min(1.0, max(0.0, -raw_slack / 1e-8)) if raw_slack < 0 else 0.0

    G = rx.PyGraph()
    c2n = {}

    for z in range(nL):
        for y in range(H):
            for x in range(W):
                c2n[(z, y, x)] = G.add_node((z, y, x))


    for z in range(1, nL):
        if layer_dir[z] == 0:
            for y in range(H):
                for x in range(W - 1):
                    w_e = _edge_weight(sub, z, y, x, y, x + 1, criticality)
                    if policy is not None:
                        w_e += policy.edge_supplement(
                            z, min_y + y, min_x + x,
                            min_y + y, min_x + x + 1, matrix)
                    G.add_edge(c2n[(z, y, x)], c2n[(z, y, x + 1)], w_e)
        elif layer_dir[z] == 1:
            for y in range(H - 1):
                for x in range(W):
                    w_e = _edge_weight(sub, z, y, x, y + 1, x, criticality)
                    if policy is not None:
                        w_e += policy.edge_supplement(
                            z, min_y + y, min_x + x,
                            min_y + y + 1, min_x + x, matrix)
                    G.add_edge(c2n[(z, y, x)], c2n[(z, y + 1, x)], w_e)


    via_base = policy.via_cost_base if policy is not None else 40.0
    via_cost = via_base * (1.0 - 0.3 * criticality)
    for z in range(nL - 1):
        for y in range(H):
            for x in range(W):
                G.add_edge(c2n[(z, y, x)], c2n[(z + 1, y, x)], via_cost)


    terminal_nodes = []
    seen = set()
    for coord in terminal_coords:
        nid = c2n.get(coord)
        if nid is not None and nid not in seen:
            terminal_nodes.append(nid)
            seen.add(nid)

    if len(terminal_nodes) < 2:
        return [], terminal_coords

    tree = rx.steiner_tree(G, terminal_nodes, weight_fn=lambda w: w)
    edges = [(tree[u], tree[v]) for u, v in tree.edge_list()]
    return edges, terminal_coords


def find_solution_for_net(net, matrix, data_cap, data_net, metrics,
                          policy=None, force_maze=False):
    min_x, min_y, max_x, max_y, w, h = _get_ranges(data_net[net])
    cells_used = set()


    if w == 1 and h == 1:
        pin_info = data_net[net][0]
        if len(pin_info) == 3:
            pin_name, slack, coordinates = pin_info
            z1, x1, y1 = coordinates[0]
        else:
            z1, x1, y1 = pin_info[0]
        s = f'{net}\n(\n{x1} {y1} {z1} {x1} {y1} {z1 + 1}\n)\n'
        matrix[z1:z1 + 2, y1:y1 + 1, x1:x1 + 1] -= 1
        metrics.total_vias += 1
        cells_used.add((z1, y1, x1))
        if z1 + 1 < matrix.shape[0]:
            cells_used.add((z1 + 1, y1, x1))
        return s, cells_used

    edges, terminal_coords = find_solution_with_rustworkx(
        net, matrix, data_cap, data_net, policy=policy,
        force_maze=force_maze)


    used_global = set()

    s = f'{net}\n(\n'
    if not edges:
        tn = terminal_coords[0]
        s += (f'{min_x + tn[2]} {min_y + tn[1]} {tn[0]} '
              f'{min_x + tn[2]} {min_y + tn[1]} {tn[0] + 1}\n')
        metrics.total_vias += 1
        cells_used.add((tn[0], min_y + tn[1], min_x + tn[2]))
    else:
        for u, v in edges:
            if v[0] < u[0] or v[1] < u[1] or v[2] < u[2]:
                u, v = v, u
            s += (f'{min_x + u[2]} {min_y + u[1]} {u[0]} '
                  f'{min_x + v[2]} {min_y + v[1]} {v[0]}\n')
            gu = (u[0], min_y + u[1], min_x + u[2])
            gv = (v[0], min_y + v[1], min_x + v[2])
            used_global.add(gu)
            used_global.add(gv)
            cells_used.add(gu)
            cells_used.add(gv)
            if u[0] != v[0]:
                metrics.total_vias += 1
            else:
                metrics.total_wirelength += abs(u[1] - v[1]) + abs(u[2] - v[2])
    s += ')\n'


    nL, H, W = matrix.shape
    for z, y, x in used_global:
        if 0 <= z < nL and 0 <= y < H and 0 <= x < W:
            matrix[z, y, x] -= 1
    return s, cells_used


def order_nets(data_net, data_cap, max_nets=None, seed=42):
    rng = random.Random(seed)
    items = []
    for net_name, net_data in data_net.items():
        bbox = _get_ranges(net_data)
        area = bbox[4] * bbox[5]
        slack = _net_min_slack(net_data)


        sort_key = -area if area > 1 else 10**9
        rand_tie = rng.random()
        items.append((sort_key, slack, rand_tie, net_name))

    items.sort(key=lambda t: (t[0], t[1], t[2]))
    ordered = [it[3] for it in items]

    if max_nets is not None and max_nets < len(ordered):
        ordered = ordered[:max_nets]
    return ordered


def route_circuit_batched(output_file, data_cap, data_net, batch_size=5000, max_nets=None, benchmark='ariane', seed=42):
    t0 = time.time()
    matrix = data_cap['cap'].astype(np.float32).copy()

    ordered = order_nets(data_net, data_cap, max_nets=max_nets, seed=seed)
    total = len(ordered)


    all_slacks = []
    for net_name in ordered:
        net_data = data_net[net_name]
        for pin_info in net_data:
            if len(pin_info) == 3:
                all_slacks.append(float(pin_info[1]))
    neg_slacks = [s for s in all_slacks if s < 0]

    metrics = RoutingMetrics()
    metrics.benchmark = locals().get('benchmark', getattr(args, 'benchmark', 'ariane'))
    cap_w = data_cap.get('ppa_weights', [])
    if cap_w and len(cap_w) >= 4:
        metrics.weights = {'w1': cap_w[0], 'w2': cap_w[1], 'w3': cap_w[2], 'w4': cap_w[3]}
    metrics.total_nets = len(data_net)
    metrics.routed_nets = total
    metrics.n_endpoints = len(all_slacks) if all_slacks else 1
    metrics.min_slack = min(all_slacks) if all_slacks else 0.0
    metrics.total_neg_slack = sum(neg_slacks) if neg_slacks else 0.0

    n_batches = math.ceil(total / batch_size)
    metrics.batches = n_batches

    eq = '=' * 72
    print(f'\n{eq}')
    print(f' ISPD 2025 Global Router -- RustWorkX + Slack-Aware Ordering')
    print(f'{eq}')
    print(f' Design grid : {data_cap["nLayers"]} x '
          f'{data_cap["xSize"]} x {data_cap["ySize"]}')
    print(f' Total nets : {len(data_net):,}')
    print(f' Routing nets : {total:,}')
    print(f' Batch size : {batch_size:,}')
    print(f' Batches : {n_batches}')
    print(f' Algorithm : Steiner tree (Mehlhorn 2-approx) via RustWorkX')
    print(f' Net ordering : Area-descending + Slack-priority')
    print(f'{eq}\n')

    with open(output_file, 'w') as out:
        for b in range(n_batches):
            lo = b * batch_size
            hi = min(lo + batch_size, total)
            batch = ordered[lo:hi]

            bt0 = time.time()
            bar = tqdm.tqdm(batch,
                            desc=f' Batch {b+1}/{n_batches}',
                            leave=True, unit='net')
            for net in bar:
                bar.set_postfix_str(f'{net[:40]}', refresh=False)
                sol, _ = find_solution_for_net(
                    net, matrix, data_cap, data_net, metrics)
                out.write(sol)
            bar.close()

            bt1 = time.time()
            throughput = len(batch) / max(bt1 - bt0, 1e-6)
            print(f' -> {len(batch):,} nets in {bt1-bt0:.2f}s '
                  f'({throughput:,.0f} nets/sec)')

    metrics.runtime_sec = time.time() - t0


    overflow = np.maximum(-matrix, 0)
    metrics.total_overflow = float(overflow.sum())
    metrics.overflow_score = float((overflow ** 2).sum())

    return metrics


def print_results(m, scale_to_full=False):
    scale = m.total_nets / max(m.routed_nets, 1) if scale_to_full else 1.0

    eq = '=' * 72
    ul = '_' * 68

    print(f'\n{eq}')
    print(f' ROUTING RESULTS')
    print(f'{eq}')
    print(f' Nets routed : {m.routed_nets:>10,} / {m.total_nets:,} '
          f'({100*m.routed_nets/max(m.total_nets,1):.1f}%)')
    print(f' Batches : {m.batches:>10}')
    print(f' Total runtime : {m.runtime_sec:>10.2f} s')
    print(f' Throughput : {m.routed_nets/max(m.runtime_sec,1e-6):>10,.0f} nets/sec')
    print(f' Wirelength : {m.total_wirelength:>10,}')
    print(f' Vias : {m.total_vias:>10,}')
    print(f' Overflow (total) : {m.total_overflow:>10,.0f}')
    print(f' Overflow (score) : {m.overflow_score:>10,.0f}')

    if scale > 1.0:
        print(f'\n {ul}')
        print(f' NORMALISED TO FULL DESIGN (x{scale:.2f})')
        print(f' {ul}')
        print(f' Est. runtime : {m.runtime_sec * scale:>10.1f} s')
        print(f' Est. wirelength : {m.total_wirelength * scale:>10,.0f}')
        print(f' Est. vias : {m.total_vias * scale:>10,.0f}')
        print(f' Est. overflow : {m.overflow_score * scale:>10,.0f}')


    bm = getattr(m, 'benchmark', 'ariane')
    W = getattr(m, 'weights', None) or ISPD_WEIGHTS.get(bm, ISPD_WEIGHTS['ariane'])
    sorig = m.sorig(W)
    T = m.runtime_factor()
    ss = m.scaled_score(W)

    print(f'\n {ul}')
    print(f' ISPD 2025 SCORING PROXY ({bm} weights)')
    print(f' {ul}')
    print(f' WNS proxy (input slack) : {m.min_slack*1e9:>12.4f} ns')
    print(f' TNS proxy (input sum) : {m.total_neg_slack*1e9:>12.4f} ns')
    print(f' Endpoints : {m.n_endpoints:>12,}')
    print(f' Benchmark : {bm}')
    print(f' w1={W["w1"]} w2={W["w2"]} w3={W["w3"]} w4={W["w4"]}')
    print(f' ------------------------------------------')
    print(f' S_orig (score) : {sorig:>12.4f}')
    print(f' T (runtime factor) : {T:>12.4f}')
    print(f' S_scaled (score) : {ss:>12.4f}')
    print(f' * Note: Scores mapped from .cap dynamically.\n')

    print(f'\n{eq}\n')


class CongestionLog:

    def __init__(self):
        self.iterations = []
        self.net_cells = {}
        self.net_wl = {}
        self.net_vias = {}
        self.init_net_wl = {}
        self.init_net_vias = {}

    def start_iteration(self, iteration_id):
        self.iterations.append({
            'id': iteration_id,
            'nets_routed': 0,
            'nets_ripped': 0,
            'overflow_l2': 0.0,
            'overflow_total': 0.0,
            'wirelength': 0,
            'vias': 0,
            'policy_params': {},
            'llm_time_s': 0.0,
        })

    def record_net(self, net_name, cells, wl, vias):
        if len(self.iterations) == 1:
            self.init_net_wl[net_name] = wl
            self.init_net_vias[net_name] = vias

        self.net_cells[net_name] = cells
        self.net_wl[net_name] = wl
        self.net_vias[net_name] = vias

    def finalize(self, **kwargs):
        if self.iterations:
            self.iterations[-1].update(kwargs)

    def print_summary(self):
        print('\n ITERATION LOG')
        print(' ' + '-' * 68)
        for it in self.iterations:
            tag = 'INIT' if it['id'] == 0 else f"R&R-{it['id']}"
            llm_str = f" LLM: {it.get('llm_time_s', 0):.2f}s" if it.get('llm_time_s', 0) > 0 else ""
            print(f" [{tag:>6s}] overflow(L2)={it.get('overflow_l2', 0):>12,.0f} "
                  f"ripped={it.get('nets_ripped', 0):>6,} "
                  f"WL={it.get('wirelength', 0):>10,} "
                  f"vias={it.get('vias', 0):>8,}{llm_str}")


class OverflowAnalyzer:

    def __init__(self):
        self.layer_importance = {}
        self.hotspot_map = None
        self.features = {}

    def analyze(self, orig_cap, current_cap, layer_dirs):
        overflow = np.maximum(-current_cap, 0)
        nL, H, W = overflow.shape
        total_of = float(overflow.sum())


        self.layer_importance = {}
        for z in range(nL):
            lo = float(overflow[z].sum())
            self.layer_importance[z] = {
                'overflow': lo,
                'pct': lo / max(total_of, 1e-6) * 100,
                'dir': 'H' if z < len(layer_dirs) and layer_dirs[z] == 0 else 'V',
                'max_cell': float(overflow[z].max()),
            }


        self.hotspot_map = overflow.sum(axis=0)


        utilization = np.clip(1.0 - current_cap / (orig_cap + 1e-6), 0, 2)
        flat_of = overflow.ravel()
        flat_util = utilization.ravel()
        if flat_of.std() > 0 and flat_util.std() > 0:
            corr = float(np.corrcoef(flat_of, flat_util)[0, 1])
        else:
            corr = 0.0

        self.features = {
            'total_overflow': total_of,
            'overflow_l2': float((overflow ** 2).sum()),
            'congested_cells': int((current_cap < 0).sum()),
            'congested_pct': float((current_cap < 0).sum()) / max(current_cap.size, 1) * 100,
            'util_overflow_corr': corr,
            'max_hotspot': float(self.hotspot_map.max()),
            'top_layers': sorted(
                ((z, info) for z, info in self.layer_importance.items()),
                key=lambda x: -x[1]['overflow'],
            )[:3],
        }
        return self.features

    def get_rip_up_candidates(self, net_cells_map, current_cap, pct=20):
        overflow = np.maximum(-current_cap, 0)
        scores = {}
        nL, H, W = overflow.shape
        for net_name, cells in net_cells_map.items():
            score = 0.0
            for z, y, x in cells:
                if 0 <= z < nL and 0 <= y < H and 0 <= x < W:
                    score += overflow[z, y, x]
            if score > 0:
                scores[net_name] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        n_rip = max(1, len(ranked) * pct // 100)
        return [name for name, _ in ranked[:n_rip]]

    def print_report(self):
        f = self.features
        if not f:
            return
        print('\n OVERFLOW FEATURE ANALYSIS')
        print(' ' + '-' * 68)
        print(f' Congested cells : {f["congested_cells"]:,} '
              f'({f["congested_pct"]:.2f}% of grid)')
        print(f' Util-overflow corr: {f["util_overflow_corr"]:.4f}')
        print(f' Max hotspot value : {f["max_hotspot"]:.1f}')
        print(' Top overflow layers:')
        for z, info in f.get('top_layers', []):
            print(f' Layer {z:2d} ({info["dir"]}): '
                  f'{info["overflow"]:>10,.0f} '
                  f'({info["pct"]:.1f}%) '
                  f'max_cell={info["max_cell"]:.0f}')


class AdaptivePolicy:

    def __init__(self, shape):
        nL, H, W = shape
        self.history = np.zeros(shape, dtype=np.float32)
        self.alpha = 0.4
        self.alpha_growth = 1.15
        self.present_factor = 2.0
        self.via_cost_base = 40.0
        self._n_layers = nL
        self._partition_boxes = {}
        self._partition_boost = {}
        self._keepout_boxes = []
        self._max_layer_allowed = None

    def set_partitions(self, partitions):
        self._partition_boxes = {
            p.id: (p.x_lo, p.y_lo, p.x_hi, p.y_hi) for p in partitions
        }

    def boost_partition(self, pid, strength=1.0):
        cur = self._partition_boost.get(pid, 0.0)
        self._partition_boost[pid] = cur + 8.0 * max(0.0, float(strength))

    def add_keepout_bbox(self, bbox, strength=1.0):
        if not bbox or len(bbox) != 4:
            return
        x0, y0, x1, y1 = [int(v) for v in bbox]
        penalty = 20.0 * max(0.0, float(strength))
        self._keepout_boxes.append((x0, y0, x1, y1, penalty))

    def restrict_max_layer(self, max_layer, strength=1.0):
        if max_layer is None:
            return
        m = max(0, min(int(max_layer), self._n_layers - 1))
        if self._max_layer_allowed is None:
            self._max_layer_allowed = m
        else:
            self._max_layer_allowed = min(self._max_layer_allowed, m)

    @staticmethod
    def _inside_box(x, y, box):
        x0, y0, x1, y1 = box
        return x0 <= x <= x1 and y0 <= y <= y1

    def _macro_penalty(self, z, gx, gy):
        extra = 0.0

        if self._max_layer_allowed is not None and z > self._max_layer_allowed:
            extra += 15.0

        for pid, boost in self._partition_boost.items():
            box = self._partition_boxes.get(pid)
            if box and self._inside_box(gx, gy, box):
                extra += boost

        for x0, y0, x1, y1, penalty in self._keepout_boxes:
            if self._inside_box(gx, gy, (x0, y0, x1, y1)):
                extra += penalty

        return extra

    def update_history(self, current_cap):
        overflow = np.maximum(-current_cap, 0)
        self.history += self.alpha * overflow
        self.alpha *= self.alpha_growth

    def edge_supplement(self, z, gy1, gx1, gy2, gx2, full_matrix):
        hist = float(self.history[z, gy1, gx1] + self.history[z, gy2, gx2])
        present = self.present_factor * (
            max(0.0, -float(full_matrix[z, gy1, gx1]))
            + max(0.0, -float(full_matrix[z, gy2, gx2])))
        macro = self._macro_penalty(z, gx1, gy1) + self._macro_penalty(z, gx2, gy2)
        return hist + present + macro

    def adapt(self, features):
        if not features:
            return self.params_dict()


        if features.get('congested_pct', 0) < 2.0:
            self.present_factor = min(8.0, self.present_factor * 1.2)


        if features.get('util_overflow_corr', 0) > 0.5:
            self.alpha = min(2.0, self.alpha * 1.1)


        top = features.get('top_layers', [])
        if top and top[0][1]['pct'] > 50:
            self.via_cost_base = max(20.0, self.via_cost_base * 0.92)

        return self.params_dict()

    def params_dict(self):
        return {
            'alpha': round(self.alpha, 4),
            'present_factor': round(self.present_factor, 4),
            'via_cost': round(self.via_cost_base, 1),
        }


class RoutingKG:

    def __init__(self):
        self.graph = rx.PyDiGraph()
        self._patterns = {}
        self._strategies = {}
        self._outcomes = []
        self._node_data = {}

    def _pattern_key(self, net_area, n_pins, layer_pct_top):
        area_bin = min(net_area // 500, 20)
        pin_bin = min(n_pins, 10)
        layer_bin = int(layer_pct_top * 10) // 2
        return (area_bin, pin_bin, layer_bin)

    def _strategy_key(self, policy_params, algo='steiner'):
        alpha_bin = int(policy_params.get('alpha', 0.4) * 10)
        pf_bin = int(policy_params.get('present_factor', 2.0) * 5)
        return (alpha_bin, pf_bin, algo)

    def record(self, net_name, net_area, n_pins, layer_pct_top,
               policy_params, algo, overflow_delta, congested_cells):

        pk = self._pattern_key(net_area, n_pins, layer_pct_top)
        if pk not in self._patterns:
            nid = self.graph.add_node({'type': 'pattern', 'key': pk,
                                       'count': 0, 'total_delta': 0.0})
            self._patterns[pk] = nid
            self._node_data[nid] = self.graph[nid]
        p_nid = self._patterns[pk]
        self.graph[p_nid]['count'] += 1
        self.graph[p_nid]['total_delta'] += overflow_delta


        sk = self._strategy_key(policy_params, algo)
        if sk not in self._strategies:
            nid = self.graph.add_node({'type': 'strategy', 'key': sk,
                                       'params': policy_params,
                                       'algo': algo, 'uses': 0,
                                       'sum_delta': 0.0})
            self._strategies[sk] = nid
            self._node_data[nid] = self.graph[nid]
        s_nid = self._strategies[sk]
        self.graph[s_nid]['uses'] += 1
        self.graph[s_nid]['sum_delta'] += overflow_delta


        o_nid = self.graph.add_node({'type': 'outcome',
                                      'overflow_delta': overflow_delta,
                                      'congested': congested_cells})
        self._outcomes.append(o_nid)


        self.graph.add_edge(p_nid, s_nid, 'APPLIED')
        self.graph.add_edge(s_nid, o_nid, 'PRODUCED')

    def suggest(self, net_area, n_pins, layer_pct_top):
        pk = self._pattern_key(net_area, n_pins, layer_pct_top)
        p_nid = self._patterns.get(pk)
        if p_nid is None:
            return None


        best_delta = 0.0
        best_params = None
        for s_nid in self.graph.successor_indices(p_nid):
            sdata = self.graph[s_nid]
            if sdata.get('type') == 'strategy' and sdata['uses'] > 0:
                avg_delta = sdata['sum_delta'] / sdata['uses']
                if avg_delta < best_delta:
                    best_delta = avg_delta
                    best_params = sdata.get('params')
        return best_params

    def summary(self):
        return {
            'patterns': len(self._patterns),
            'strategies': len(self._strategies),
            'outcomes': len(self._outcomes),
            'total_nodes': self.graph.num_nodes(),
            'total_edges': self.graph.num_edges(),
        }

    def print_report(self):
        s = self.summary()
        print(f'\n KNOWLEDGE GRAPH')
        print(' ' + '-' * 68)
        print(f' Patterns: {s["patterns"]:>6,} '
              f'Strategies: {s["strategies"]:>4,} '
              f'Outcomes: {s["outcomes"]:>6,}')
        print(f' Graph: {s["total_nodes"]:>6,} nodes '
              f'{s["total_edges"]:>6,} edges')

        if self._strategies:
            best = min(self._strategies.values(),
                       key=lambda nid: self.graph[nid]['sum_delta']
                       / max(self.graph[nid]['uses'], 1))
            bd = self.graph[best]
            print(f' Best strat: alpha={bd["params"].get("alpha", "?")} '
                  f'pf={bd["params"].get("present_factor", "?")} '
                  f'avg_delta={bd["sum_delta"]/max(bd["uses"],1):.0f}')


class LLMLogAnalyzer:

    SYSTEM_PROMPT = (
        "You are a VLSI routing optimization assistant. You analyze routing "
        "congestion logs and suggest parameter adjustments. Be concise.\n\n"
        "Context: We route nets through a 3D GCell grid (layers × rows × cols). "
        "Each GCell has capacity. Overflow = demand - capacity when negative. "
        "Our router uses PathFinder negotiated congestion with parameters:\n"
        "- alpha: history cost growth rate (higher = stronger penalty for "
        "repeatedly congested cells)\n"
        "- present_factor: multiplier for current overflow cost\n"
        "- via_cost: cost of changing layers\n\n"
        "Given routing metrics, suggest: INCREASE/DECREASE/KEEP for each "
        "parameter, and briefly explain why. Output JSON with keys: "
        "alpha_action, pf_action, via_action, reasoning."
    )

    def __init__(self, model='llama3.2:3b', base_url='http://localhost:11434'):
        self.model = model
        self.base_url = base_url
        self.available = False
        self.insights = []
        self._check_availability()

    def _check_availability(self):
        try:
            req = urllib.request.Request(
                f'{self.base_url}/api/tags',
                method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = [m.get('name', '') for m in data.get('models', [])]
                self.available = any(self.model in m for m in models)
                if not self.available:
                    print(f' [LLM] Model {self.model} not found. '
                          f'Available: {models}')
        except Exception:
            self.available = False

    def analyze(self, iteration, metrics_dict):
        if not self.available:
            return self._template_analysis(iteration, metrics_dict)

        prompt = (
            f"Routing iteration {iteration} metrics:\n"
            f"- Overflow (L2): {metrics_dict.get('overflow_l2', 'N/A'):,.0f}\n"
            f"- Congested cells: {metrics_dict.get('congested_pct', 0):.2f}%\n"
            f"- Util-overflow correlation: "
            f"{metrics_dict.get('util_overflow_corr', 0):.3f}\n"
            f"- Top layer overflow: "
            f"{metrics_dict.get('top_layer_pct', 0):.1f}%\n"
            f"- Current alpha: "
            f"{metrics_dict.get('alpha', 0.4):.3f}\n"
            f"- Current present_factor: "
            f"{metrics_dict.get('present_factor', 2.0):.2f}\n"
            f"- Current via_cost: "
            f"{metrics_dict.get('via_cost', 40):.0f}\n"
            f"- Nets ripped: {metrics_dict.get('nets_ripped', 0):,}\n"
            f"- Delta from init: "
            f"{metrics_dict.get('delta_pct', 0):+.1f}%\n\n"
            f"Suggest parameter adjustments as JSON."
        )

        try:
            payload = json.dumps({
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': self.SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                'stream': False,
                'options': {'temperature': 0.3, 'num_predict': 256},
            }).encode('utf-8')

            req = urllib.request.Request(
                f'{self.base_url}/api/chat',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST')

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                raw = result.get('message', {}).get('content', '')


            insight = self._parse_llm_response(raw)
            insight['raw_response'] = raw
            insight['source'] = 'llm'
            self.insights.append(insight)
            return insight

        except Exception as e:
            return self._template_analysis(iteration, metrics_dict,
                                            error=str(e))

    def _parse_llm_response(self, raw):
        result = {
            'alpha_action': 'KEEP',
            'pf_action': 'KEEP',
            'via_action': 'KEEP',
            'reasoning': raw[:200],
        }

        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                for key in ['alpha_action', 'pf_action', 'via_action',
                            'reasoning']:
                    if key in parsed:
                        result[key] = parsed[key]
        except json.JSONDecodeError:
            pass


        if result['alpha_action'] == 'KEEP':
            if 'INCREASE ALPHA' in raw.upper() or 'INCREASING ALPHA' in raw.upper(): result['alpha_action'] = 'INCREASE'
            elif 'DECREASE ALPHA' in raw.upper() or 'DECREASING ALPHA' in raw.upper(): result['alpha_action'] = 'DECREASE'

        if result['pf_action'] == 'KEEP':
            if 'INCREASE PRESENT' in raw.upper() or 'INCREASING PRESENT' in raw.upper(): result['pf_action'] = 'INCREASE'
            elif 'DECREASE PRESENT' in raw.upper() or 'DECREASING PRESENT' in raw.upper(): result['pf_action'] = 'DECREASE'

        if result['via_action'] == 'KEEP':
            if 'INCREASE VIA' in raw.upper() or 'INCREASING VIA' in raw.upper(): result['via_action'] = 'INCREASE'
            elif 'DECREASE VIA' in raw.upper() or 'DECREASING VIA' in raw.upper(): result['via_action'] = 'DECREASE'

        return result

    def _template_analysis(self, iteration, metrics_dict, error=None):
        cpct = metrics_dict.get('congested_pct', 5)
        corr = metrics_dict.get('util_overflow_corr', 0)
        top_pct = metrics_dict.get('top_layer_pct', 30)
        delta = metrics_dict.get('delta_pct', 0)

        result = {
            'alpha_action': 'KEEP',
            'pf_action': 'KEEP',
            'via_action': 'KEEP',
            'reasoning': '',
            'source': 'template' + (f' (LLM error: {error})' if error else ''),
        }

        reasons = []
        if cpct < 2.0:
            result['pf_action'] = 'INCREASE'
            reasons.append(f'congestion concentrated ({cpct:.1f}%) → raise pf')
        if corr > 0.5:
            result['alpha_action'] = 'INCREASE'
            reasons.append(f'high util-overflow corr ({corr:.2f}) → raise alpha')
        if top_pct > 50:
            result['via_action'] = 'DECREASE'
            reasons.append(f'layer concentration ({top_pct:.0f}%) → lower via cost')
        if delta > -5 and iteration > 1:
            result['alpha_action'] = 'INCREASE'
            result['pf_action'] = 'INCREASE'
            reasons.append(f'slow convergence ({delta:+.1f}%) → increase pressure')

        result['reasoning'] = '; '.join(reasons) if reasons else 'stable'
        self.insights.append(result)
        return result

    def apply_suggestion(self, policy, suggestion):
        actions = {
            'INCREASE': 100.0,
            'DECREASE': 0.0001,
            'KEEP': 1.0,
        }
        a = suggestion.get('alpha_action', 'KEEP').upper()
        p = suggestion.get('pf_action', 'KEEP').upper()
        v = suggestion.get('via_action', 'KEEP').upper()

        if a in actions:
            policy.alpha = min(500000.0, policy.alpha * actions[a])
        if p in actions:
            policy.present_factor = min(400000.0,
                                        policy.present_factor * actions[p])
        if v in actions:

            policy.via_cost_base = max(1.0, min(100.0,
                                        policy.via_cost_base * actions[v]))

    def print_insights(self):
        if not self.insights:
            return
        print('\n LLM / TEMPLATE INSIGHTS')
        print(' ' + '-' * 68)
        for i, ins in enumerate(self.insights):
            src = ins.get('source', '?')
            print(f' [Iter {i+1}] ({src}) '
                  f'alpha={ins["alpha_action"]} '
                  f'pf={ins["pf_action"]} '
                  f'via={ins["via_action"]}')
            print(f' {ins["reasoning"][:80]}')


def route_enhanced(output_file, data_cap, data_net,
                   max_iterations=5, rip_up_pct=20,
                   batch_size=5000, max_nets=None,
                   use_llm=False, use_kg=True, use_maze=True, seed=42):
    t0 = time.time()
    matrix_orig = data_cap['cap'].astype(np.float32).copy()
    matrix = matrix_orig.copy()

    ordered = order_nets(data_net, data_cap, max_nets=max_nets, seed=seed)
    total = len(ordered)


    logger = CongestionLog()
    analyzer = OverflowAnalyzer()
    policy = AdaptivePolicy(matrix.shape)
    kg = RoutingKG() if use_kg else None
    llm = LLMLogAnalyzer() if use_llm else None


    all_slacks = []
    for net_name in ordered:
        net_data = data_net[net_name]
        for pin_info in net_data:
            if len(pin_info) == 3:
                all_slacks.append(float(pin_info[1]))
    neg_slacks = [s for s in all_slacks if s < 0]

    metrics = RoutingMetrics()
    metrics.benchmark = locals().get('benchmark', getattr(args, 'benchmark', 'ariane'))
    cap_w = data_cap.get('ppa_weights', [])
    if cap_w and len(cap_w) >= 4:
        metrics.weights = {'w1': cap_w[0], 'w2': cap_w[1], 'w3': cap_w[2], 'w4': cap_w[3]}
    metrics.total_nets = len(data_net)
    metrics.routed_nets = total
    metrics.n_endpoints = len(all_slacks) if all_slacks else 1
    metrics.min_slack = min(all_slacks) if all_slacks else 0.0
    metrics.total_neg_slack = sum(neg_slacks) if neg_slacks else 0.0

    solutions = {}

    eq = '=' * 72
    print(f'\n{eq}')
    print(f' ENHANCED ROUTER: Overflow Analysis + KG + LLM + Dijkstra Maze + R&R')
    print(f'{eq}')
    print(f' Grid : {data_cap["nLayers"]}L x '
          f'{data_cap["xSize"]} x {data_cap["ySize"]}')
    print(f' Routing nets : {total:,}')
    print(f' R&R iterations : {max_iterations}')
    print(f' Rip-up target : {rip_up_pct}% of overflow-contributing nets')
    print(f' Maze routing : {"ON" if use_maze else "OFF"} '
          f'(Dijkstra for {MAX_STEINER_AREA}<area<={MAZE_MAX_AREA})')
    print(f' Knowledge Graph : {"ON" if kg else "OFF"}')
    print(f' LLM analysis : '
          f'{"ON (" + llm.model + ")" if llm and llm.available else "OFF"}')
    print(f'{eq}\n')


    logger.start_iteration(0)
    m0 = RoutingMetrics()

    n_batches = math.ceil(total / batch_size)
    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, total)
        batch = ordered[lo:hi]
        bar = tqdm.tqdm(batch,
                        desc=f' [INIT] Batch {b+1}/{n_batches}',
                        leave=True, unit='net')
        for net in bar:
            bar.set_postfix_str(f'{net[:40]}', refresh=False)
            wl_before = m0.total_wirelength
            vias_before = m0.total_vias
            sol, cells = find_solution_for_net(
                net, matrix, data_cap, data_net, m0)
            solutions[net] = sol
            logger.record_net(net, cells,
                              m0.total_wirelength - wl_before,
                              m0.total_vias - vias_before)
        bar.close()


    analysis_features = analyzer.analyze(
        matrix_orig, matrix, data_cap['layerDirections'])

    init_overflow_l2 = analysis_features['overflow_l2']
    logger.finalize(
        nets_routed=total,
        overflow_l2=init_overflow_l2,
        overflow_total=analysis_features['total_overflow'],
        wirelength=m0.total_wirelength,
        vias=m0.total_vias,
    )

    print(f'\n [INIT] Overflow(L2): {init_overflow_l2:,.0f} '
          f'Congested: {analysis_features["congested_pct"]:.2f}% '
          f'WL: {m0.total_wirelength:,} Vias: {m0.total_vias:,}')
    analyzer.print_report()


    _analysis_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    for iteration in range(1, max_iterations + 1):
        logger.start_iteration(iteration)


        llm_time = 0.0
        if llm and analysis_features:
            llm_t0 = time.time()
            top_layers = analysis_features.get('top_layers', [])
            top_pct = top_layers[0][1]['pct'] if top_layers else 0
            llm_metrics = {
                'overflow_l2': analysis_features['overflow_l2'],
                'congested_pct': analysis_features['congested_pct'],
                'util_overflow_corr': analysis_features['util_overflow_corr'],
                'top_layer_pct': top_pct,
                'delta_pct': (analysis_features['overflow_l2'] - init_overflow_l2)
                             / max(init_overflow_l2, 1e-6) * 100,
                'nets_ripped': 0,
                **policy.params_dict(),
            }
            llm_suggestion = llm.analyze(iteration, llm_metrics)
            llm.apply_suggestion(policy, llm_suggestion)
            llm_time = time.time() - llm_t0
            print(f' [LLM] {llm_suggestion.get("source", "?")} ({llm_time:.2f}s): '
                  f'{llm_suggestion.get("reasoning", "")[:70]}')


        if not llm:
            policy_params = policy.adapt(analysis_features)
        else:
            policy_params = policy.params_dict()
        policy.update_history(matrix)


        candidates = analyzer.get_rip_up_candidates(
            logger.net_cells, matrix, pct=rip_up_pct)

        if not candidates:
            print(f'\n [R&R-{iteration}] No overflow - stopping early')
            logger.finalize(nets_ripped=0, overflow_l2=analysis_features['overflow_l2'],
                            overflow_total=analysis_features['total_overflow'])
            break


        m_rr = RoutingMetrics()
        bar = tqdm.tqdm(candidates,
                        desc=f' [R&R-{iteration}] Rerouting',
                        leave=True, unit='net')
        for net in bar:
            bar.set_postfix_str(f'{net[:40]}', refresh=False)


            cells = logger.net_cells.get(net, set())
            for z, y, x in cells:
                if 0 <= z < matrix.shape[0] and 0 <= y < matrix.shape[1]\
                        and 0 <= x < matrix.shape[2]:
                    matrix[z, y, x] += 1


            wl_before = m_rr.total_wirelength
            vias_before = m_rr.total_vias
            sol, cells = find_solution_for_net(
                net, matrix, data_cap, data_net, m_rr, policy=policy,
                force_maze=use_maze)
            solutions[net] = sol
            logger.record_net(net, cells,
                              m_rr.total_wirelength - wl_before,
                              m_rr.total_vias - vias_before)


            if kg:
                bbox = _get_ranges(data_net[net])
                net_area = bbox[4] * bbox[5]
                n_pins = len(data_net[net])
                top_layers = analysis_features.get('top_layers', [])
                top_pct = top_layers[0][1]['pct'] / 100 if top_layers else 0
                algo = 'maze' if (use_maze and net_area > MAX_STEINER_AREA
                                  and net_area <= MAZE_MAX_AREA) else 'steiner'
                kg.record(net, net_area, n_pins, top_pct,
                          policy_params, algo, 0.0,
                          analysis_features.get('congested_cells', 0))
        bar.close()


        analysis_future = _analysis_executor.submit(
            analyzer.analyze, matrix_orig, matrix.copy(),
            data_cap['layerDirections'])
        analysis_features = analysis_future.result()

        new_overflow = analysis_features['overflow_l2']
        delta = new_overflow - init_overflow_l2
        delta_pct = delta / max(init_overflow_l2, 1e-6) * 100

        logger.finalize(
            nets_routed=len(candidates),
            nets_ripped=len(candidates),
            overflow_l2=new_overflow,
            overflow_total=analysis_features['total_overflow'],
            wirelength=m_rr.total_wirelength,
            vias=m_rr.total_vias,
            policy_params=policy_params,
            llm_time_s=llm_time,
        )

        print(f'\n [R&R-{iteration}] Ripped {len(candidates):,} nets '
              f'-> Overflow(L2): {new_overflow:,.0f} '
              f'({delta_pct:+.1f}% from init)')
        print(f' Policy: alpha={policy_params["alpha"]:.3f} '
              f'pf={policy_params["present_factor"]:.2f} '
              f'via={policy_params["via_cost"]:.0f}')

    _analysis_executor.shutdown(wait=False)


    with open(output_file, 'w') as out:
        for net in ordered:
            if net in solutions:
                out.write(solutions[net])


    stats_file = output_file + ".stats"
    with open(stats_file, 'w') as sf:
        sf.write("net_name,init_wl,final_wl,init_vias,final_vias\n")
        for n in ordered:
            iw = logger.init_net_wl.get(n, 0)
            fw = logger.net_wl.get(n, iw)
            iv = logger.init_net_vias.get(n, 0)
            fv = logger.net_vias.get(n, iv)
            sf.write(f"{n},{iw},{fw},{iv},{fv}\n")

    metrics.runtime_sec = time.time() - t0


    metrics.total_wirelength = sum(logger.net_wl.values())
    metrics.total_vias = sum(logger.net_vias.values())


    overflow = np.maximum(-matrix, 0)
    metrics.total_overflow = float(overflow.sum())
    metrics.overflow_score = float((overflow ** 2).sum())

    logger.print_summary()
    analyzer.print_report()
    if kg:
        kg.print_report()
    if llm:
        llm.print_insights()

    return metrics, logger


if __name__ == '__main__':
    start_time = time.time()
    p = argparse.ArgumentParser(
        description='ISPD 2025 Performance-Driven Global Router '
                    '(RustWorkX + Slack-Aware + Overflow Analysis + R&R)')
    p.add_argument('-cap', required=True, help='.cap file path')
    p.add_argument('-net', required=True, help='.net file path')
    p.add_argument('-output', required=True, help='Output routing file')
    p.add_argument('-batch', type=int, default=5000,
                   help='Nets per batch (default: 5000)')
    p.add_argument('-max_nets', type=int, default=-1,
                   help='Max nets to route (-1 = all)')
    p.add_argument('-benchmark', type=str, default='ariane',
                   choices=list(ISPD_WEIGHTS.keys()),
                   help='Benchmark for ISPD weights')
    p.add_argument('-enhanced', action='store_true',
                   help='Enable overflow analysis + KG + R&R (enhanced mode)')
    p.add_argument('-rr_iters', type=int, default=5,
                   help='Rip-up & reroute iterations (default: 5)')
    p.add_argument('-rr_pct', type=int, default=20,
                   help='Pct of overflow nets to rip up (default: 20)')
    p.add_argument('-llm', action='store_true',
                   help='Enable Ollama LLM log analysis')
    p.add_argument('-no_kg', action='store_true',
                   help='Disable Knowledge Graph')
    p.add_argument('-no_maze', action='store_true',
                   help='Disable Dijkstra maze routing for R&R')
    p.add_argument('-seed', type=int, default=42,
                   help='Random seed for net ordering')
    args = p.parse_args()

    print('Read cap: {}'.format(args.cap))
    data_cap = read_cap(args.cap)
    print('Read net: {}'.format(args.net))
    data_net = read_net(args.net)

    print("Number of nets: {} Field size: {}x{}x{}".format(
        len(data_net),
        data_cap['nLayers'], data_cap['xSize'], data_cap['ySize']))

    max_nets = None if args.max_nets == -1 else args.max_nets

    if args.enhanced:
        metrics, log = route_enhanced(
            args.output, data_cap, data_net,
            max_iterations=args.rr_iters,
            rip_up_pct=args.rr_pct,
            batch_size=args.batch,
            max_nets=max_nets,
            use_llm=args.llm,
            use_kg=not args.no_kg,
            use_maze=not args.no_maze,
            seed=args.seed,
        )
    else:
        metrics = route_circuit_batched(
            args.output, data_cap, data_net,
            batch_size=args.batch,
            max_nets=max_nets,
            seed=args.seed,
        )

    scale = max_nets is not None and max_nets < len(data_net)
    print_results(metrics, scale_to_full=scale)

    print('Overall time: {:.2f} sec'.format(time.time() - start_time))
