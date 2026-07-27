# Yelp POI-graph dataset

Reproduces the **Yelp** dataset from the SelMAG paper (Appendix A). One graph
per city; a node is a point-of-interest (POI / business), an edge is a
*co-review* relationship (≥ `min_common_reviewers` users reviewed both POIs),
node features are averaged **GLOVE** word embeddings of a POI's review text,
and labels are 5 POI categories: **{Food, Shop, Home Service, Health Service,
Finance}**.

Cities (six of different scales): **Madison, Glendale, Gilbert, Las Vegas,
Toronto, Phoenix** — **Phoenix is the target** graph.

On the first run the loader **auto-downloads the raw dump from Google Drive**
(via gdown), reads `business.json`, and **streams `review.json` straight out of
the `.tgz`** (never fully extracted), caching every city graph to
`data/yelp/<City>/processed_fgw.pt`. Later runs load the cache directly.

## ⚠️ Why not Kaggle / yelp.com?

The current Yelp Open Dataset (the `yelp_dataset.tar` from yelp.com, Jan 2022)
**does not contain the paper's cities** — Phoenix / Las Vegas / Toronto /
Gilbert are gone (it now ships Philadelphia, Tucson, Tampa, … instead). Those
cities live only in a **pre-2021 round**. That old round is preserved as Kaggle
*version 1*, but **Kaggle is IP-blocked (403) from some regions (e.g. Iran)**.
So the loader downloads a copy of that old round hosted on a **public Google
Drive** (Drive works where Kaggle doesn't):

```
https://drive.google.com/file/d/1wzwSjAtQJaQa8hWT9WIHQrTH4m-zk4gO/view
```

### Recommended: auto-download (gdown)

```bash
pip install gdown
python main_yelp_fgw.py          # downloads the .tgz, builds, trains
```

The loader fetches the file into `data/yelp/raw/yelp_dataset.tgz` once and
streams from it. Use `--gdrive_id <ID>` to point at a different Drive file, or
`--no_download` to require a manually placed dump (below).

### Manual alternative

If you'd rather fetch it yourself, drop the `.tgz` in `data/yelp/raw/` — **leave
it compressed**, the loader streams from it:

```bash
mkdir -p data/yelp/raw
gdown 1wzwSjAtQJaQa8hWT9WIHQrTH4m-zk4gO -O data/yelp/raw/yelp_dataset.tgz
python main_yelp_fgw.py --no_download
```

Extracted `*.json`, a `.zip`, or a `.tar[.gz]` in `data/yelp/raw/` or `data/yelp/`
all work too; extracted files take precedence over archives.

## GLOVE vectors (~862 MB zip) → `data/yelp/glove/`

**What/why:** GLOVE is a table of pre-trained word vectors (Stanford). Each POI's
feature is the *average GLOVE vector of the words in its reviews* — that's how the
graph nodes get features. Needed once, at build time.

This is **auto-downloaded** too, from the Hugging Face mirror (reachable where
`nlp.stanford.edu` is blocked), and only the needed dimension is extracted. No
action required — just run. To do it by hand (macOS has no `wget`, use `curl`):

```bash
mkdir -p data/yelp/glove
curl -L https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip \
  -o data/yelp/glove/glove.6B.zip
unzip data/yelp/glove/glove.6B.zip -d data/yelp/glove
```

This yields `glove.6B.{50,100,200,300}d.txt`. The loader defaults to
`glove.6B.300d.txt` (300-dim features); pass `--glove_path` for another dim, or
`--glove_url` for a different mirror.

## Run

```bash
pip install gdown
python main_yelp_fgw.py                       # auto-downloads dump+GLOVE, target = Phoenix
python main_yelp_fgw.py --target Toronto      # different target
python main_yelp_fgw.py --sources Madison Glendale Gilbert --target Phoenix
```

On the first run the loader prints the kept-POI count per city — confirm Phoenix
and the others are non-zero (this verifies the Drive `.tgz` is the old round with
the paper's cities).
