# Paper references

MinerU `standard`-tier extractions of the channel-fusion source papers. PDFs and
extracted figure images are git-ignored; the markdown, this README, and the
commands below are enough to reproduce the conversion.

```
<paper-name>/
├── <paper-name>.pdf   # source paper (git-ignored)
├── paper.md           # full markdown, tables included, figures referenced locally
└── images/            # figure/plot blocks extracted as PNG (git-ignored)
```

| Paper | Source | Extraction | Figures | MinerU doc |
|---|---|---|---|---|
| Channel Vision Transformers: An Image Is Worth 1 x 16 x 16 Words | [2309.16108](https://arxiv.org/abs/2309.16108) | [`channelvit/paper.md`](channelvit/paper.md) | 14 | `doc:b89d1b5` |
| ChAda-ViT: Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images | [2311.15264](https://arxiv.org/abs/2311.15264) | [`chada-vit/paper.md`](chada-vit/paper.md) | 18 | `doc:4d707b7` |
| Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation | [2503.10845](https://arxiv.org/abs/2503.10845) | [`panopticon/paper.md`](panopticon/paper.md) | 8 | `doc:127996d` |
| Perceiver: General Perception with Iterative Attention | [2103.03206](https://arxiv.org/abs/2103.03206) | [`perceiver/paper.md`](perceiver/paper.md) | 37 | `doc:edbfd63` |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs | [2107.14795](https://arxiv.org/abs/2107.14795) | [`perceiver-io/paper.md`](perceiver-io/paper.md) | 7 | `doc:3b5a822` |
| Masked Autoencoders Are Scalable Vision Learners (MAE) | [2111.06377](https://arxiv.org/abs/2111.06377) | [`mae/paper.md`](mae/paper.md) | 17 | `doc:1b49044` |
| ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers | [2503.19331](https://arxiv.org/abs/2503.19331) | [`cha-maevit/paper.md`](cha-maevit/paper.md) | 25 | `doc:762255e` |
| Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | [1810.00825](https://arxiv.org/abs/1810.00825) | [`set-transformer/paper.md`](set-transformer/paper.md) | 6 | `doc:36b5dc3` |
| Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation (DOFA) | [2403.15356](https://arxiv.org/abs/2403.15356) | [`dofa/paper.md`](dofa/paper.md) | 45 | `doc:5a8fed9` |
| HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery | [2503.21841](https://arxiv.org/abs/2503.21841) | [`hyperfree/paper.md`](hyperfree/paper.md) | 107 | `doc:1e4e58f` |
| FlexiViT: One Model for All Patch Sizes | [2212.08013](https://arxiv.org/abs/2212.08013) | [`flexivit/paper.md`](flexivit/paper.md) | 62 | `doc:516935e` |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | [2104.09864](https://arxiv.org/abs/2104.09864) | [`roformer/paper.md`](roformer/paper.md) | 4 | `doc:e9a481f` |
| Rotary Position Embedding for Vision Transformer | [2403.13298](https://arxiv.org/abs/2403.13298) | [`rope-vit/paper.md`](rope-vit/paper.md) | 7 | `doc:416333c` |
| MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy | [2602.24222](https://arxiv.org/abs/2602.24222) | [`muvit/paper.md`](muvit/paper.md) | 10 | `doc:d025dc2` |
| Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning | [2212.14532](https://arxiv.org/abs/2212.14532) | [`scale-mae/paper.md`](scale-mae/paper.md) | 8 | `doc:f227430` |
| Evolutionary-scale prediction of atomic level protein structure with a language model (ESM-2) | [bioRxiv 2022.07.20.500902](https://doi.org/10.1101/2022.07.20.500902) | [`esm-2/paper.md`](esm-2/paper.md) | 31 | `doc:23c994c` |
| Reducing cost of fluorescent microscopy through dye exclusion, distillation techniques, and deep learning (CR-DINO) | [Sci Rep 2026, DOI](https://doi.org/10.1038/s41598-026-69424-3) | [`cr-dino/paper.md`](cr-dino/paper.md) | 3 | `doc:95fc5bd` |
| Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics | [2602.02093](https://arxiv.org/abs/2602.02093) | [`cell-jepa/paper.md`](cell-jepa/paper.md) | 6 | `doc:2d88ff7` |
| SCJEPA: Dual-Space Self-Supervised Learning for Single-Cell Representations | [OpenReview KadCjvcLOz](https://openreview.net/forum?id=KadCjvcLOz) | [`scjepa/paper.md`](scjepa/paper.md) | 4 | `doc:594c318` |
| Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) | [2301.08243](https://arxiv.org/abs/2301.08243) | [`i-jepa/paper.md`](i-jepa/paper.md) | 8 | `doc:eddbdc0` |
| Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA) | [2404.08471](https://arxiv.org/abs/2404.08471) | [`v-jepa/paper.md`](v-jepa/paper.md) | 16 | `doc:fb84f79` |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | [2506.09985](https://arxiv.org/abs/2506.09985) | [`v-jepa-2/paper.md`](v-jepa-2/paper.md) | 35 | `doc:9cfcfde` |
| scMorphJEPA: Self-Supervised Cell Morphology Learning via Spatial Joint-Embedding Predictive Architecture | [GitHub](https://github.com/simo1946/scMorphJEPA) | [`scmorphjepa/README.md`](scmorphjepa/README.md) | — | — |

Tables are inline in each `paper.md` (markdown and/or HTML). Figure blocks that
MinerU marked as images/charts are exported as PNG from the PDF, and the
markdown links are rewritten to local `images/*.png` paths.

scMorphJEPA has no paper or PDF; its reference implementation is checked out
in place (commit `a0e028c`, 2026-07-17), with `README.md` as the entry point.
SCJEPA is an LMRL workshop submission at ICLR 2026 and should be treated as
provisional. The CR-DINO PDF is the publisher's Crossref text-mining copy.

## Reproduce

All source PDFs are public. Most are arXiv papers; ESM-2 is a bioRxiv
preprint, CR-DINO is a Scientific Reports article, and SCJEPA is hosted on
OpenReview. The directories are named after the papers, so the download loop
maps arXiv id to directory explicitly:

```bash
mkdir -p references
while read -r id name; do
  mkdir -p "references/$name"
  curl -L --fail -o "references/$name/$name.pdf" "https://arxiv.org/pdf/$id"
done <<'IDS'
2309.16108 channelvit
2311.15264 chada-vit
2503.10845 panopticon
2103.03206 perceiver
2107.14795 perceiver-io
2111.06377 mae
2503.19331 cha-maevit
1810.00825 set-transformer
2403.15356 dofa
2503.21841 hyperfree
2212.08013 flexivit
2104.09864 roformer
2403.13298 rope-vit
2602.24222 muvit
2212.14532 scale-mae
2602.02093 cell-jepa
2301.08243 i-jepa
2404.08471 v-jepa
2506.09985 v-jepa-2
IDS
# ESM-2 is a bioRxiv preprint, not arXiv:
curl -L --fail -o references/esm-2/esm-2.pdf \
  'https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full.pdf'

# CR-DINO is a Scientific Reports article; use the Crossref text-mining PDF:
curl -L --fail -o references/cr-dino/cr-dino.pdf \
  'https://www.nature.com/articles/s41598-026-69424-3_reference.pdf'
# SCJEPA is on OpenReview. If curl receives a browser challenge, open the PDF
# in a browser session and save it to references/scjepa/scjepa.pdf.
curl -L --fail -o references/scjepa/scjepa.pdf \
  'https://openreview.net/pdf?id=KadCjvcLOz'
# scMorphJEPA has no paper/PDF; check out the reference implementation in place.
# Drop the nested .git so the checkout is tracked as part of this repo.
rm -rf references/scmorphjepa
git clone --depth 1 https://github.com/simo1946/scMorphJEPA references/scmorphjepa
rm -rf references/scmorphjepa/.git
```

Set up local MinerU `standard` parsing (one-time; ~2 GB models):

```bash
mineru server start
mineru-kit models download --tier standard
mineru-kit models verify --tier standard
mineru config set parse_server.local.managed_tier standard
mineru config set parse_server.local.mode managed
mineru server restart
mineru server status --json   # wait for parse_server.local.healthy == true
```

Parse each paper to markdown:

```bash
while read -r id name; do
  mineru parse "references/$name/$name.pdf" --pages all --tier standard \
    --output "references/$name/paper.md"
done <<'IDS'
2309.16108 channelvit
2311.15264 chada-vit
2503.10845 panopticon
2103.03206 perceiver
2107.14795 perceiver-io
2111.06377 mae
2503.19331 cha-maevit
1810.00825 set-transformer
2403.15356 dofa
2503.21841 hyperfree
2212.08013 flexivit
2104.09864 roformer
2403.13298 rope-vit
2602.24222 muvit
2212.14532 scale-mae
23c994c esm-2
95fc5bd cr-dino
2d88ff7 cell-jepa
594c318 scjepa
eddbdc0 i-jepa
fb84f79 v-jepa
9cfcfde v-jepa-2
IDS
```

Materialize figure blocks and rewrite the `doc:` image placeholders to local
files:

```bash
python3 - <<'PY'
import re, subprocess
from pathlib import Path

root = Path("references")
pattern = re.compile(r"!\[([^\]]*)\]\((doc:[^)]+)\)")

for md in sorted(root.glob("*/paper.md")):
    text = md.read_text()
    out = text
    img_dir = md.parent / "images"
    img_dir.mkdir(exist_ok=True)
    for alt, locator in pattern.findall(text):
        match = re.search(r"/page:(\d+)/block:(\d+)", locator)
        name = (
            f"page{int(match.group(1)):03d}-block{int(match.group(2)):02d}.png"
            if match
            else f"image{len(list(img_dir.glob('*.png'))):03d}.png"
        )
        subprocess.run(
            ["mineru", "read", locator, "--format", "image",
             "--output", str(img_dir / name)],
            check=True,
        )
        out = out.replace(f"![{alt}]({locator})", f"![{alt}](images/{name})")
    md.write_text(out)
PY
```

## Reading

Markdown files are self-contained for agents; the `doc:` ids remain useful for
block-level citations or exporting images on demand:

```bash
mineru read "doc:b89d1b5/tier:standard/page:4"
mineru read "doc:4d707b7/tier:standard/page:12/block:5" --format image --output /tmp/block.png
```