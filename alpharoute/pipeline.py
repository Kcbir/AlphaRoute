"""Micro-loop and macro-loop orchestration."""
import argparse
import math
import os
import sys
import time
import numpy as np
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_cap, read_net
from router import (
    find_solution_for_net, find_solution_with_rustworkx,
    order_nets, _get_ranges, _net_min_slack,
    RoutingMetrics, AdaptivePolicy, OverflowAnalyzer, CongestionLog,
    RoutingKG, print_results, ISPD_WEIGHTS,
)
from alpharoute.core import (
    Hypergraph, HypergraphMessagePassing, SpatialPartitioner,
)
from alpharoute.optimizer import AugmentedLagrangianOptimizer
from alpharoute.spatial_llm import (
    MacroLoop, EphemeralKnowledgeGraph, BBoxPayloadBuilder,
)


class ALPolicy(AdaptivePolicy):

    def __init__(self, shape):
        super().__init__(shape)
        self.al_penalty = np.zeros(shape, dtype=np.float32)
        self.hgat_congestion = np.zeros(shape, dtype=np.float32)
        self._hgat_weight = 5.0

    def set_al_penalty(self, penalty_map: np.ndarray):
        self.al_penalty = penalty_map

    def set_hgat_congestion(self, cong_map: np.ndarray):
        self.hgat_congestion = cong_map

    def edge_supplement(self, z, gy1, gx1, gy2, gx2, full_matrix):

        base = super().edge_supplement(z, gy1, gx1, gy2, gx2, full_matrix)


        al = float(self.al_penalty[z, gy1, gx1]
                    + self.al_penalty[z, gy2, gx2])


        hgat = self._hgat_weight * float(
            self.hgat_congestion[z, gy1, gx1]
            + self.hgat_congestion[z, gy2, gx2])

        return base + al + hgat


class AlphaRoutePipeline:

    def __init__(self, data_cap: dict, data_net: dict,
                 micro_iters: int = 8, rip_up_pct: int = 20,
                 n_partitions: int = 4):
        self.data_cap = data_cap
        self.data_net = data_net
        self.micro_iters = micro_iters
        self.rip_up_pct = rip_up_pct
        self.n_partitions = n_partitions

        self.matrix_orig = data_cap['cap'].astype(np.float32).copy()
        self.matrix = self.matrix_orig.copy()
        self.grid_shape = self.matrix.shape

    def run(self, max_nets: int = None, output_file: str = None,
            batch_size: int = 5000) -> RoutingMetrics:
        t0 = time.time()


        print("\n [INIT] Building hypergraph...")
        hg = Hypergraph.from_netlist(self.data_net, self.data_cap)
        hgat = HypergraphMessagePassing(hg, n_hops=2)
        print(f" [INIT] Hypergraph: {hg.n_nets:,} nets, {hg.n_pins:,} pins")


        partitions, cut_nets = SpatialPartitioner.partition(
            hg, self.n_partitions)
        print(f" [INIT] Partitions: {len(partitions)}, "
              f"cut-nets: {len(cut_nets):,}")


        import concurrent.futures
        self.llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.llm_future = None

        al = AugmentedLagrangianOptimizer(self.grid_shape)
        policy = ALPolicy(self.grid_shape)
        policy.set_partitions(partitions)
        analyzer = OverflowAnalyzer()
        logger = CongestionLog()
        kg = RoutingKG()
        ekg = EphemeralKnowledgeGraph()
        macro = MacroLoop(ekg)


        ordered = order_nets(self.data_net, self.data_cap, max_nets=max_nets)
        total = len(ordered)


        all_slacks = []
        for net_data in self.data_net.values():
            for pin_info in net_data:
                if len(pin_info) == 3:
                    all_slacks.append(float(pin_info[1]))
        neg_slacks = [s for s in all_slacks if s < 0]

        metrics = RoutingMetrics()
        metrics.total_nets = len(self.data_net)
        metrics.routed_nets = total
        metrics.n_endpoints = len(all_slacks) if all_slacks else 1
        metrics.min_slack = min(all_slacks) if all_slacks else 0.0
        metrics.total_neg_slack = sum(neg_slacks) if neg_slacks else 0.0

        solutions = {}

        eq = '=' * 72
        print(f'\n{eq}')
        print(f' AlphaRoute: LLM-guided global router')
        print(f'{eq}')
        print(f' Grid : {self.grid_shape[0]}L x '
              f'{self.grid_shape[2]}W x {self.grid_shape[1]}H')
        print(f' Routing nets : {total:,}')
        print(f' Micro iterations: {self.micro_iters}')
        print(f' Rip-up target : {self.rip_up_pct}%')
        print(f' Partitions : {len(partitions)}')
        print(f' Cut-nets : {len(cut_nets):,}')
        print(f'{eq}\n')


        logger.start_iteration(0)
        m0 = RoutingMetrics()


        cong = hgat.forward(self.matrix_orig, self.matrix,
                            self.data_cap['layerDirections'])
        policy.set_hgat_congestion(cong)

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
                    net, self.matrix, self.data_cap, self.data_net, m0)
                solutions[net] = sol
                logger.record_net(net, cells,
                                  m0.total_wirelength - wl_before,
                                  m0.total_vias - vias_before)
            bar.close()


        analysis_features = analyzer.analyze(
            self.matrix_orig, self.matrix,
            self.data_cap['layerDirections'])
        al_loss, overflow_g, al_info = al.compute_loss(
            self.matrix_orig, self.matrix)

        init_of_l2 = al_info['overflow_l2']
        prev_of_l2 = init_of_l2

        logger.finalize(
            nets_routed=total,
            overflow_l2=init_of_l2,
            overflow_total=al_info['overflow_l1'],
            wirelength=m0.total_wirelength,
            vias=m0.total_vias,
        )

        print(f'\n [INIT] Overflow(L2): {init_of_l2:,.0f} '
              f'Overflow(L1): {al_info["overflow_l1"]:,.0f} '
              f'Congested: {al_info["congested_cells"]:,}')
        analyzer.print_report()


        macro_triggered_count = 0
        local_net_counter = 0

        for iteration in range(1, self.micro_iters + 1):
            logger.start_iteration(iteration)
            iter_t0 = time.time()


            cong = hgat.forward(self.matrix_orig, self.matrix,
                                self.data_cap['layerDirections'])
            policy.set_hgat_congestion(cong)


            penalty = al.get_penalty_map(self.matrix)
            policy.set_al_penalty(penalty)


            hgat_net_scores = hgat.get_net_congestion(self.matrix)


            candidates = analyzer.get_rip_up_candidates(
                logger.net_cells, self.matrix, pct=self.rip_up_pct)

            if not candidates:
                print(f'\n [R&R-{iteration}] No overflow - done')
                break


            overflow = np.maximum(-self.matrix, 0)
            nL, H, W = self.grid_shape
            combined_scores = {}
            for net in candidates:
                cells = logger.net_cells.get(net, set())
                overflow_score = sum(
                    overflow[z, y, x] for z, y, x in cells
                    if 0 <= z < nL and 0 <= y < H and 0 <= x < W)
                hgat_score = hgat_net_scores.get(net, 0.0)
                combined_scores[net] = overflow_score + 0.3 * hgat_score


            candidates = sorted(combined_scores.keys(),
                                key=lambda n: -combined_scores[n])


            policy_params = policy.adapt(analysis_features)
            policy.update_history(self.matrix)


            for net_name in candidates:
                cells = logger.net_cells.get(net_name, set())
                for z, y, x in cells:
                    if 0 <= z < nL and 0 <= y < H and 0 <= x < W:
                        self.matrix[z, y, x] += 1


            m_rr = RoutingMetrics()
            bar = tqdm.tqdm(candidates,
                            desc=f' [R&R-{iteration}] Rerouting',
                            leave=True, unit='net')
            for net in bar:
                local_net_counter += 1


                if self.llm_future and self.llm_future.done():
                    try:
                        result = self.llm_future.result()
                        actions = result.get('actions', [])
                        if actions:
                            macro.apply_actions(actions, policy, al)
                            print(f"\n [MACRO-ASYNC] Applied {len(actions)} EKG actions dynamically mid-iteration!")
                    except Exception as e:
                        pass
                    self.llm_future = None


                if local_net_counter % 1000 == 0 and self.llm_future is None:
                    print(f'\n [MACRO] {local_net_counter} nets routed. Triggering asynchronous LLM payload...')
                    payload = BBoxPayloadBuilder.build(
                        partitions, self.matrix_orig, self.matrix,
                        self.data_cap['layerDirections'],
                        cut_nets=cut_nets, hg=hg)
                    self.llm_future = self.llm_executor.submit(
                        macro.trigger, payload, iteration, init_of_l2, init_of_l2
                    )

                bar.set_postfix_str(f'{net[:40]}', refresh=False)
                wl_before = m_rr.total_wirelength
                vias_before = m_rr.total_vias
                sol, cells = find_solution_for_net(
                    net, self.matrix, self.data_cap, self.data_net, m_rr,
                    policy=policy, force_maze=True)
                solutions[net] = sol
                logger.record_net(net, cells,
                                  m_rr.total_wirelength - wl_before,
                                  m_rr.total_vias - vias_before)
            bar.close()


            al.update_multipliers(self.matrix)
            al_loss, overflow_g, al_info = al.compute_loss(
                self.matrix_orig, self.matrix)


            boundary_overflow = {}
            for cn in cut_nets:
                cn_of = 0.0
                for pin in hg.net_pins.get(cn, []):
                    z, x, y, _ = pin
                    if 0 <= z < nL and 0 <= y < H and 0 <= x < W:
                        cn_of += max(0, -float(self.matrix[z, y, x]))
                if cn_of > 0:
                    boundary_overflow[cn] = cn_of
            al.update_cut_multipliers(cut_nets, boundary_overflow)


            analysis_features = analyzer.analyze(
                self.matrix_orig, self.matrix,
                self.data_cap['layerDirections'])

            cur_of_l2 = al_info['overflow_l2']
            delta_pct = (cur_of_l2 - init_of_l2) / max(init_of_l2, 1e-6) * 100

            logger.finalize(
                nets_routed=len(candidates),
                nets_ripped=len(candidates),
                overflow_l2=cur_of_l2,
                overflow_total=al_info['overflow_l1'],
                wirelength=m_rr.total_wirelength,
                vias=m_rr.total_vias,
                policy_params=policy_params,
            )

            print(f'\n [R&R-{iteration}] Ripped {len(candidates):,} nets '
                  f'-> OF(L2): {cur_of_l2:,.0f} ({delta_pct:+.1f}%) '
                  f'OF(L1): {al_info["overflow_l1"]:,.0f} '
                  f'rho={al.rho:.2f} '
                  f't={time.time()-iter_t0:.1f}s')


            if self.llm_future and self.llm_future.done():
                try:
                    result = self.llm_future.result()
                    actions = result.get('actions', [])
                    if actions:
                        macro.apply_actions(actions, policy, al)
                        print(f" [MACRO-ASYNC] Applied {len(actions)} EKG actions from previous plateau!")
                except Exception as e:
                    print(f" [MACRO-ASYNC] Error compiling LLM output: {e}")
                self.llm_future = None

            if iteration > 1:
                macro_triggered_count += 1
                if self.llm_future is None:
                    print(f'\n [MACRO] Plateau detected at iter {iteration} (trigger #{macro_triggered_count}). Submitting async job...')

                    payload = BBoxPayloadBuilder.build(
                        partitions, self.matrix_orig, self.matrix,
                        self.data_cap['layerDirections'],
                        cut_nets=cut_nets, hg=hg)

                    self.llm_future = self.llm_executor.submit(
                        macro.trigger, payload, iteration, prev_of_l2, cur_of_l2
                    )
                else:
                    print(f'\n [MACRO] Plateau detected at iter {iteration}, but an async job is already running...')
            else:

                ekg.update_confidence(prev_of_l2, cur_of_l2, iteration)

            prev_of_l2 = cur_of_l2


            for net in candidates[:100]:
                bbox = _get_ranges(self.data_net[net])
                net_area = bbox[4] * bbox[5]
                n_pins = len(self.data_net[net])
                top_layers = analysis_features.get('top_layers', [])
                top_pct = top_layers[0][1]['pct'] / 100 if top_layers else 0
                kg.record(net, net_area, n_pins, top_pct,
                          policy_params, 'maze', al.get_overflow_delta(),
                          al_info['congested_cells'])


        if output_file:
            with open(output_file, 'w') as out:
                for net in ordered:
                    if net in solutions:
                        out.write(solutions[net])

        metrics.runtime_sec = time.time() - t0
        metrics.total_wirelength = sum(logger.net_wl.values())
        metrics.total_vias = sum(logger.net_vias.values())

        final_overflow = np.maximum(-self.matrix, 0)
        metrics.total_overflow = float(final_overflow.sum())
        metrics.overflow_score = float((final_overflow ** 2).sum())


        logger.print_summary()
        analyzer.print_report()
        kg.print_report()

        print(f'\n [MACRO] Triggered {macro_triggered_count} times')
        print(f' [EKG] Final: {ekg.summary()}')

        return metrics


def main():
    p = argparse.ArgumentParser(
        description='AlphaRoute: LLM-guided global router')
    p.add_argument('-cap', required=True, help='.cap file')
    p.add_argument('-net', required=True, help='.net file')
    p.add_argument('-output', default='alpharoute_output.txt', help='Output file')
    p.add_argument('-max_nets', type=int, default=-1,
                   help='Max nets (-1 = all)')
    p.add_argument('-micro_iters', type=int, default=8,
                   help='Micro-loop R&R iterations')
    p.add_argument('-rip_pct', type=int, default=20,
                   help='Rip-up percentage')
    p.add_argument('-batch', type=int, default=5000, help='Batch size')
    p.add_argument('-partitions', type=int, default=4,
                   help='Number of spatial partitions')
    args = p.parse_args()

    print(f'Read cap: {args.cap}')
    data_cap = read_cap(args.cap)
    print(f'Read net: {args.net}')
    data_net = read_net(args.net)
    print(f'Nets: {len(data_net):,} Grid: {data_cap["nLayers"]}L x '
          f'{data_cap["xSize"]}W x {data_cap["ySize"]}H')

    max_nets = None if args.max_nets == -1 else args.max_nets

    pipeline = AlphaRoutePipeline(
        data_cap, data_net,
        micro_iters=args.micro_iters,
        rip_up_pct=args.rip_pct,
        n_partitions=args.partitions,
    )

    metrics = pipeline.run(
        max_nets=max_nets,
        output_file=args.output,
        batch_size=args.batch,
    )

    scale = max_nets is not None and max_nets < len(data_net)
    print_results(metrics, scale_to_full=scale)


if __name__ == '__main__':
    main()
