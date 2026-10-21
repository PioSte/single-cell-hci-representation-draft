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

| Paper | arXiv | Extraction | Figures | MinerU doc |
|---|---|---|---|---|
| Channel Vision Transformers: An Image Is Worth 1 x 16 x 16 Words | [2309.16108](https://arxiv.org/abs/2309.16108) | [`channelvit/paper.md`](channelvit/paper.md) | 14 | `doc:b89d1b5` |
| ChAda-ViT: Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images | [2311.15264](https://arxiv.org/abs/2311.15264) | [`chada-vit/paper.md`](chada-vit/paper.md) | 18 | `doc:4d707b7` |
| Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation | [2503.10845](https://arxiv.org/abs/2503.10845) | [`panopticon/paper.md`](panopticon/paper.md) | 8 | `doc:127996d` |
| Perceiver: General Perception with Iterative Attention | [2103.03206](https://arxiv.org/abs/2103.03206) | [`perceiver/paper.md`](perceiver/paper.md) | 37 | `doc:edbfd63` |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs | [2107.14795](https://arxiv.org/abs/2107.14795) | [`perceiver-io/paper.md`](perceiver-io/paper.md) | 7 | `doc:3b5a822` |
| ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers | [2503.19331](https://arxiv.org/abs/2503.19331) | [`cha-maevit/paper.md`](cha-maevit/paper.md) | 25 | `doc:762255e` |
| Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks | [1810.00825](https://arxiv.org/abs/1810.00825) | [`set-transformer/paper.md`](set-transformer/paper.md) | 6 | `doc:36b5dc3` |
| Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation (DOFA) | [2403.15356](https://arxiv.org/abs/2403.15356) | [`dofa/paper.md`](dofa/paper.md) | 45 | `doc:5a8fed9` |
| HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery | [2503.21841](https://arxiv.org/abs/2503.21841) | [`hyperfree/paper.md`](hyperfree/paper.md) | 107 | `doc:1e4e58f` |
| Evolutionary-scale prediction of atomic level protein structure with a language model (ESM-2) | [bioRxiv 2022.07.20.500902](https://doi.org/10.1101/2022.07.20.500902) | [`esm-2/paper.md`](esm-2/paper.md) | 31 | `doc:23c994c` |

Tables are inline in each `paper.md` (markdown and/or HTML). Figure blocks that
MinerU marked as images/charts are exported as PNG from the PDF, and the
markdown links are rewritten to local `images/*.png` paths.

## Cited but not yet extracted

The ESM-2 marker-token claim comes from **VirTues**, which is still not
extracted. PMA terminology is now covered by `set-transformer/paper.md`.

| Source | Used for | Links |
|---|---|---|
| VirTues (ESM-2 marker tokens) | Sequence-grounded identity / zero-shot new markers | [DOI](https://doi.org/10.21203/rs.3.rs-8865586/v1) · [model page](https://bio.rodeo/models/virtues) |

## Reproduce

All source PDFs are public. Most are arXiv papers; ESM-2 is a bioRxiv
preprint. The directories are named after the papers, so the download loop maps
arXiv id to directory explicitly:

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
2503.19331 cha-maevit
1810.00825 set-transformer
2403.15356 dofa
2503.21841 hyperfree
IDS
# ESM-2 is a bioRxiv preprint, not arXiv:
curl -L --fail -o references/esm-2/esm-2.pdf \
  'https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full.pdf'
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
2503.19331 cha-maevit
1810.00825 set-transformer
2403.15356 dofa
2503.21841 hyperfree
23c994c esm-2
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