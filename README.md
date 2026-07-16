# AlphaRoute

A global router for VLSI physical design that lets a language model tune the routing penalties instead of hand-coding a fixed schedule.

Global routing decides how every signal net gets a path across a chip's grid of routing cells. The hard part is congestion: too many nets fighting for the same cells. Most routers deal with this using penalty schedules that are fixed up front, but the penalty that actually works depends on what the congestion looks like at each step. AlphaRoute keeps that decision open. It measures the overflow after each pass, asks an LLM what to do about it, and applies the answer only if a knowledge graph says it's within bounds. The LLM never touches wires directly, so the routing stays deterministic and reproducible.

It's all Python, built on open-source libraries. On the ISPD 2025 ARIANE benchmark it gets overflow down to 146,109 with an original score of `S_orig = 0.0538`.

This is the code behind our paper, *AlphaRoute: Large Language Models as Semantic Optimizers for Multi-Objective Routing* (IEEE LAD).

## The pipeline

Here's what happens on a run, start to finish:

1. **Parse and build the graph.** Read the `.cap` and `.net` files into per-cell capacities and pin locations, then build a RustWorkX grid graph. Edge weights already factor in capacity and net slack.
2. **Order the nets.** Big nets route first since they're the ones that create congestion for everyone else. When two nets are close in size, the more timing-critical one wins and takes the shortest path.
3. **Route everything once.** Small nets get Steiner trees. Big nets use a Prim MST plus flexible L-routing, picking the layer with the most room. This first pass alone cuts ARIANE's initial overflow by 2.2x.
4. **Figure out where the overflow is.** A SHAP-style analyzer (we wrote our own, no external SHAP dependency) breaks the overflow down per net, per layer, and by how well utilization predicts it.
5. **Rip up and reroute.** Take the worst 20% of nets by overflow contribution and route them again. Medium nets go through 3D Dijkstra maze routing to get around hotspots. The history and present-overflow penalties climb as iterations go.
6. **Let the LLM tune the knobs.** The overflow metrics go to an LLM, which suggests new values for the history penalty, present factor, and via cost, with a short explanation. Before anything is applied, a RustWorkX knowledge graph checks it against hard limits. If the suggestion doesn't pass, the deterministic rules take over instead.

So the LLM only ever adjusts high-level penalty parameters. The geometry is fixed math.

## Setup

You'll need Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

`groq` is optional; you only need it if you want the LLM macro-loop. Ollama runs over HTTP, so there's no package to install for it.

## Getting the data

The benchmarks are the ISPD 2025 Global Routing Contest designs: ARIANE, BSG, MEMPOOL, and NVDLA. They're too big to ship here, so grab the `.cap` and `.net` files from the contest and drop them in `data/`:

```
data/ariane.cap
data/ariane.net
```

## Running it

Just the initial routing, no rip-up:

```bash
python router.py -cap data/ariane.cap -net data/ariane.net \
                 -output artifacts/ariane.route
```

With overflow analysis, the knowledge graph, and rip-up/reroute:

```bash
python router.py -cap data/ariane.cap -net data/ariane.net \
                 -output artifacts/ariane.route -enhanced
```

And with the LLM tuning the penalties on top of that:

```bash
python router.py -cap data/ariane.cap -net data/ariane.net \
                 -output artifacts/ariane.route -enhanced -llm
```

### Flags worth knowing

| Flag | Default | What it does |
| --- | --- | --- |
| `-cap` | required | Path to the `.cap` capacity file |
| `-net` | required | Path to the `.net` netlist file |
| `-output` | required | Where to write the routing result |
| `-benchmark` | `ariane` | Which scoring weights to use (`ariane`, `bsg`, `nvdla`, `mempool`) |
| `-enhanced` | off | Turn on overflow analysis, the knowledge graph, and rip-up/reroute |
| `-llm` | off | Let the LLM adjust penalties (needs `-enhanced`) |
| `-rr_iters` | 5 | How many rip-up and reroute passes |
| `-rr_pct` | 20 | What fraction of overflow nets to rip up each pass |
| `-batch` | 5000 | Nets per batch, keeps memory in check |
| `-max_nets` | -1 | Cap the net count for a quick test (-1 means all of them) |
| `-no_kg` | off | Skip the knowledge graph |
| `-no_maze` | off | Skip Dijkstra maze rerouting |

There's also the `alpharoute` package, which runs a two-tier version: hypergraph message passing and an augmented-Lagrangian loop underneath, with the LLM loop on top.

```bash
python -m alpharoute.pipeline -cap data/ariane.cap -net data/ariane.net \
                              -output artifacts/ariane.route
```

### LLM setup

The macro-loop picks up your Groq key from the environment:

```bash
export GROQ_API_KEY=your_key_here
```

The `-llm` flag on `router.py` talks to a local Ollama instance instead (`http://localhost:11434`, model `llama3.2:3b`). If Ollama isn't running, it quietly falls back to a template-based analysis, so nothing breaks.

## Reproducing the ablations

`run_experiments.sh` runs the baseline, enhanced, and LLM configurations at a few net counts and dumps the logs into `artifacts/`:

```bash
./run_experiments.sh
```

## What's where

```
router.py            The main router: parsing, ordering, rip-up/reroute, scoring
utils.py             Parsers for the .cap and .net formats
alpharoute/          The two-tier pipeline
  core.py            Hypergraph message passing and spatial partitioning
  optimizer.py       Augmented-Lagrangian penalty optimizer
  spatial_llm.py     The LLM macro-loop and knowledge graph
  pipeline.py        Ties the micro and macro loops together
run_experiments.sh   Runs the ARIANE ablations
```

## Citing

If AlphaRoute is useful in your work:

```bibtex
@inproceedings{murjani2026alpharoute,
  title     = {AlphaRoute: Large Language Models as Semantic Optimizers for Multi-Objective Routing},
  author    = {Murjani, Kabir and Bhavsar, Mishri and Patel, Manish I. and Talukdar, Jonti},
  booktitle = {IEEE LAD},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
