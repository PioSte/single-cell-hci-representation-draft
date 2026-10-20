# Paper references

MinerU `standard`-tier extractions of the channel-fusion source papers. PDFs and
extracted figure images are git-ignored; the markdown, this README, and the
commands below are enough to reproduce the conversion.

```
<arxiv-id>/
├── paper.md      # full markdown, tables included, figures referenced locally
└── images/       # figure/plot blocks extracted as PNG (git-ignored)
```

| Paper | arXiv | Extraction | Figures | MinerU doc |
|---|---|---|---|---|
| Channel Vision Transformers: An Image Is Worth 1 x 16 x 16 Words | [2309.16108](https://arxiv.org/abs/2309.16108) | [`2309.16108/paper.md`](2309.16108/paper.md) | 14 | `doc:b89d1b5` |
| ChAda-ViT: Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images | [2311.15264](https://arxiv.org/abs/2311.15264) | [`2311.15264/paper.md`](2311.15264/paper.md) | 18 | `doc:4d707b7` |
| Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation | [2503.10845](https://arxiv.org/abs/2503.10845) | [`2503.10845/paper.md`](2503.10845/paper.md) | 8 | `doc:127996d` |
| Perceiver: General Perception with Iterative Attention | [2103.03206](https://arxiv.org/abs/2103.03206) | [`2103.03206/paper.md`](2103.03206/paper.md) | 37 | `doc:edbfd63` |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs | [2107.14795](https://arxiv.org/abs/2107.14795) | [`2107.14795/paper.md`](2107.14795/paper.md) | 7 | `doc:3b5a822` |

Tables are inline in each `paper.md` (markdown and/or HTML). Figure blocks that
MinerU marked as images/charts are exported as PNG from the PDF, and the
markdown links are rewritten to local `images/*.png` paths.

## Reproduce

All source PDFs are public arXiv papers:

```bash
mkdir -p references
for id in 2309.16108 2311.15264 2503.10845 2103.03206 2107.14795; do
  curl -L --fail -o "references/$id.pdf" "https://arxiv.org/pdf/$id"
done
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
for id in 2309.16108 2311.15264 2503.10845 2103.03206 2107.14795; do
  mineru parse "references/$id.pdf" --pages all --tier standard \
    --output "references/$id/paper.md"
done
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