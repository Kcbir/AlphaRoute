"""LLM macro-loop and ephemeral knowledge graph."""
import json
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set


class BBoxPayloadBuilder:

    @staticmethod
    def build(partitions, cap_orig, cap_cur, layer_dirs,
              cut_nets: Set[str] = None, hg=None) -> str:
        nL, H, W = cap_cur.shape
        overflow = np.maximum(-cap_cur, 0)
        total_of = float(overflow.sum())

        sections = []
        sections.append(f"GRID: {nL}L x {W}W x {H}H TOTAL_OVERFLOW: {total_of:.0f}")


        layer_lines = []
        for z in range(nL):
            lo = float(overflow[z].sum())
            d = 'H' if z < len(layer_dirs) and layer_dirs[z] == 0 else 'V'
            if lo > 0:
                layer_lines.append(f" L{z}({d}): of={lo:.0f} "
                                   f"max_cell={float(overflow[z].max()):.0f}")
        if layer_lines:
            sections.append("LAYER_OVERFLOW:\n" + "\n".join(layer_lines))


        for p in partitions:
            x0, y0 = max(0, p.x_lo), max(0, p.y_lo)
            x1, y1 = min(W - 1, p.x_hi), min(H - 1, p.y_hi)

            sub_of = overflow[:, y0:y1+1, x0:x1+1]
            sub_cap = cap_cur[:, y0:y1+1, x0:x1+1]
            sub_orig = cap_orig[:, y0:y1+1, x0:x1+1]

            p_total = float(sub_of.sum())
            p_l2 = float((sub_of ** 2).sum())
            p_max = float(sub_of.max())
            p_congested = int((sub_cap < 0).sum())
            p_total_cells = sub_cap.size

            util = np.clip(1.0 - sub_cap / (sub_orig + 1e-6), 0, 2)
            util_mean = float(util.mean())

            cut_of = 0.0
            n_cut = len(p.cut_nets)
            if hg and p.cut_nets:
                for cn in p.cut_nets:
                    for pin in hg.net_pins.get(cn, []):
                        z, x, y, _ = pin
                        if 0 <= z < nL and 0 <= y < H and 0 <= x < W:
                            cut_of += max(0, -float(cap_cur[z, y, x]))

            sections.append(
                f"PARTITION P{p.id} [{x0},{y0}]-[{x1},{y1}]:\n"
                f" overflow_total={p_total:.0f} overflow_L2={p_l2:.0f} "
                f"max_cell={p_max:.0f}\n"
                f" congested_cells={p_congested}/{p_total_cells} "
                f"util_mean={util_mean:.3f}\n"
                f" cut_nets={n_cut} cut_overflow={cut_of:.0f}\n"
                f" local_nets={len(p.nets)}"
            )

        return "\n\n".join(sections)


class EKGRule:
    __slots__ = ('rule_id', 'text', 'action', 'target_partition', 'payload',
                 'confidence', 'epoch_created', 'epoch_committed',
                 'epoch_killed', 'status')

    def __init__(self, rule_id: int, text: str, action: str,
                 target_partition: int, epoch: int, payload: dict):
        self.rule_id = rule_id
        self.text = text
        self.action = action
        self.target_partition = target_partition
        self.payload = payload
        self.confidence = 0.0
        self.epoch_created = epoch
        self.epoch_committed = -1
        self.epoch_killed = -1
        self.status = 'active'


class EphemeralKnowledgeGraph:

    def __init__(self, gamma: float = 10.0,
                 commit_thresh: float = 0.3,
                 kill_thresh: float = -0.2):
        self.gamma = gamma
        self.commit_thresh = commit_thresh
        self.kill_thresh = kill_thresh
        self.rules: Dict[int, EKGRule] = {}
        self._next_id = 0
        self.committed_rules: List[EKGRule] = []
        self.killed_rules: List[EKGRule] = []

    def add_rules(self, parsed_rules: List[dict], epoch: int):
        added = []
        for r in parsed_rules:
            rid = self._next_id
            self._next_id += 1
            rule = EKGRule(
                rule_id=rid,
                text=r.get('text', ''),
                action=r.get('action', 'unknown'),
                target_partition=r.get('partition', -1),
                epoch=epoch,
                payload=r,
            )
            self.rules[rid] = rule
            added.append(rid)
        return added

    def update_confidence(self, overflow_prev: float, overflow_cur: float,
                          epoch: int) -> dict:
        active = [r for r in self.rules.values() if r.status == 'active']
        n_active = len(active)
        if n_active == 0:
            return {'active': 0, 'committed': len(self.committed_rules),
                    'killed': len(self.killed_rules), 'delta_per_rule': 0.0}


        if overflow_prev > 1e-6:
            delta_rel = (overflow_prev - overflow_cur) / overflow_prev
        else:
            delta_rel = 0.0


        delta_per_rule = self.gamma * delta_rel / n_active

        newly_committed = []
        newly_killed = []

        for rule in active:
            rule.confidence += delta_per_rule

            if rule.confidence >= self.commit_thresh:
                rule.status = 'committed'
                rule.epoch_committed = epoch
                self.committed_rules.append(rule)
                newly_committed.append(rule.rule_id)

            elif rule.confidence <= self.kill_thresh:
                rule.status = 'killed'
                rule.epoch_killed = epoch
                self.killed_rules.append(rule)
                newly_killed.append(rule.rule_id)


        for rid in newly_committed + newly_killed:
            if rid in self.rules:
                del self.rules[rid]

        return {
            'active': len(self.rules),
            'committed': len(self.committed_rules),
            'killed': len(self.killed_rules),
            'newly_committed': len(newly_committed),
            'newly_killed': len(newly_killed),
            'delta_per_rule': delta_per_rule,
            'delta_rel': delta_rel,
        }

    def get_active_actions(self) -> List[dict]:
        actions = []
        for r in list(self.rules.values()) + self.committed_rules:
            actions.append({
                'rule_id': r.rule_id,
                'action': r.action,
                'partition': r.target_partition,
                'confidence': r.confidence,
                'status': r.status,
                'payload': r.payload,
            })
        return actions

    def summary(self) -> str:
        active = [r for r in self.rules.values()]
        lines = [f"EKG: {len(active)} active, "
                 f"{len(self.committed_rules)} committed, "
                 f"{len(self.killed_rules)} killed"]
        for r in active[:5]:
            lines.append(f" [{r.rule_id}] C={r.confidence:+.3f} "
                         f"act={r.action} P{r.target_partition}")
        for r in self.committed_rules[-3:]:
            lines.append(f" [{r.rule_id}] COMMITTED C={r.confidence:+.3f} "
                         f"act={r.action}")
        return "\n".join(lines)


GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

SYSTEM_PROMPT = """\
You are a VLSI global routing optimization expert. You analyze spatial \
congestion data from an ISPD 2025 benchmark and propose actionable rules.

The router uses:
- Augmented Lagrangian optimization with per-GCell multipliers (lambda, rho)
- PathFinder negotiated congestion (history cost + present overflow)
- Hypergraph message passing for congestion prediction
- Spatial partitioning with cut-net tracking

Given partition-level congestion statistics, propose 3-5 rules as JSON array.
Each rule must have:
  "text": human-readable description
    "action": one of [increase_rho, decrease_rho, increase_alpha, decrease_alpha, \
increase_pf, decrease_pf, decrease_via_cost, reroute_partition, spread_layers, \
enforce_keepout, restrict_layer]
  "partition": target partition ID (-1 for global)
  "reasoning": why this will help

Optional fields for geometric rules:
    "bbox": [x_min, y_min, x_max, y_max]
    "max_layer": integer

Be specific. Reference partition IDs, layer numbers, and overflow values.
Output ONLY the JSON array, no markdown fences."""


class MacroLoop:

    def __init__(self, ekg: EphemeralKnowledgeGraph = None):
        self.ekg = ekg or EphemeralKnowledgeGraph()
        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self):
        try:
            self._available = False; self._client = None
        except ImportError:
            print(" [MACRO] groq package not installed. pip install groq")
            self._available = False
        except Exception as e:
            print(f" [MACRO] Groq init failed: {e}")
            self._available = False

    def query_llm(self, payload: str, epoch: int) -> List[dict]:
        if not self._available or self._client is None:
            return self._template_rules(payload, epoch)

        try:
            response = self._client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.3,
                max_tokens=1024,
                stream=False,
            )

            raw = response.choices[0].message.content.strip()
            rules = self._parse_rules(raw)
            if rules:
                print(f" [MACRO] LLM returned {len(rules)} rules")
                return rules
            else:
                print(f" [MACRO] LLM response unparseable, using templates")
                return self._template_rules(payload, epoch)

        except Exception as e:
            print(f" [MACRO] Groq API error: {e}")
            return self._template_rules(payload, epoch)

    def _parse_rules(self, raw: str) -> List[dict]:

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [r for r in parsed if 'action' in r]
        except json.JSONDecodeError:
            pass


        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
                if isinstance(parsed, list):
                    return [r for r in parsed if 'action' in r]
            except json.JSONDecodeError:
                pass

        return []

    def _template_rules(self, payload: str, epoch: int) -> List[dict]:
        rules = []


        total_of = 0.0
        lines = payload.split('\n')
        for line in lines:
            if 'TOTAL_OVERFLOW' in line:
                try:
                    total_of = float(line.split('TOTAL_OVERFLOW:')[1].strip())
                except (ValueError, IndexError):
                    pass


        partitions = []
        current_pid = -1
        current_of = 0.0
        for line in lines:
            if line.startswith('PARTITION P'):
                try:
                    pid = int(line.split('P')[1].split(' ')[0])
                    current_pid = pid
                except (ValueError, IndexError):
                    pass
            if 'overflow_total=' in line and current_pid >= 0:
                try:
                    of_str = line.split('overflow_total=')[1].split()[0]
                    current_of = float(of_str)
                    partitions.append((current_pid, current_of))
                    current_pid = -1
                except (ValueError, IndexError):
                    pass

        if not partitions:

            rules.append({
                'text': f'Epoch {epoch}: increase rho globally to penalize overflow',
                'action': 'increase_rho',
                'partition': -1,
            })
            rules.append({
                'text': f'Epoch {epoch}: increase alpha for stronger history signal',
                'action': 'increase_alpha',
                'partition': -1,
            })
            return rules


        worst = max(partitions, key=lambda x: x[1])
        best = min(partitions, key=lambda x: x[1])

        rules.append({
            'text': f'P{worst[0]} has highest overflow ({worst[1]:.0f}). '
                    f'Focus rerouting there.',
            'action': 'reroute_partition',
            'partition': worst[0],
        })

        if total_of > 0 and worst[1] / total_of > 0.4:
            rules.append({
                'text': f'P{worst[0]} has >{40}% of total overflow. '
                        f'Increase present_factor to penalize.',
                'action': 'increase_pf',
                'partition': worst[0],
            })

        rules.append({
            'text': 'Spread traffic across layers to reduce per-layer congestion.',
            'action': 'decrease_via_cost',
            'partition': -1,
        })

        return rules

    def trigger(self, payload: str, epoch: int,
                overflow_prev: float, overflow_cur: float) -> dict:

        new_rules = self.query_llm(payload, epoch)


        rule_ids = self.ekg.add_rules(new_rules, epoch)


        ekg_status = self.ekg.update_confidence(
            overflow_prev, overflow_cur, epoch)


        actions = self.ekg.get_active_actions()

        print(f" [EKG] {ekg_status['active']} active, "
              f"{ekg_status['newly_committed']} newly committed, "
              f"{ekg_status['newly_killed']} killed "
              f"ΔO_rel={ekg_status['delta_rel']:+.4f} "
              f"credit/rule={ekg_status['delta_per_rule']:+.4f}")

        return {
            'new_rules': len(new_rules),
            'rule_ids': rule_ids,
            'ekg_status': ekg_status,
            'actions': actions,
        }

    def apply_actions(self, actions: List[dict], policy, al_optimizer):
        for act in actions:
            c = act['confidence']
            if c <= 0:
                continue


            strength = min(1.0, c / self.ekg.commit_thresh)

            action = act['action']
            payload = act.get('payload', {})

            if action == 'increase_rho':
                al_optimizer.rho = min(al_optimizer.rho_max,
                                       al_optimizer.rho * (1.0 + 0.5 * strength))
            elif action == 'decrease_rho':
                al_optimizer.rho *= (1.0 - 0.4 * strength)
            elif action == 'increase_alpha':
                policy.alpha = min(2.0, policy.alpha * (1.0 + 0.5 * strength))
            elif action == 'decrease_alpha':
                policy.alpha *= (1.0 - 0.4 * strength)
            elif action == 'increase_pf':
                policy.present_factor = min(
                    8.0, policy.present_factor * (1.0 + 0.8 * strength))
            elif action == 'decrease_pf':
                policy.present_factor *= (1.0 - 0.6 * strength)
            elif action == 'decrease_via_cost':
                policy.via_cost_base = max(
                    15.0, policy.via_cost_base * (1.0 - 0.4 * strength))
            elif action == 'increase_via_cost':
                policy.via_cost_base = min(
                    100.0, policy.via_cost_base * (1.0 + 0.5 * strength))
            elif action == 'reroute_partition':
                pid = int(payload.get('partition', act.get('partition', -1)))
                if hasattr(policy, 'boost_partition') and pid >= 0:
                    policy.boost_partition(pid, strength)
            elif action == 'spread_layers':

                policy.via_cost_base = max(
                    15.0, policy.via_cost_base * (1.0 - 0.06 * strength))
                policy.present_factor = max(
                    1.0, policy.present_factor * (1.0 - 0.03 * strength))
            elif action == 'enforce_keepout':
                bbox = payload.get('bbox')
                if hasattr(policy, 'add_keepout_bbox') and bbox is not None:
                    policy.add_keepout_bbox(bbox, strength)
            elif action == 'restrict_layer':
                max_layer = payload.get('max_layer')
                if hasattr(policy, 'restrict_max_layer') and max_layer is not None:
                    policy.restrict_max_layer(max_layer, strength)
