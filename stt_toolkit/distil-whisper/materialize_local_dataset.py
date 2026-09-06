#!/usr/bin/env python
"""
Materialize (fetch + decode + WER-filter) one or more Italian sources from
bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual into real,
non-streaming datasets.Dataset objects, then push EACH ONE to its OWN Hub
dataset repo (Parquet-with-embedded-audio, via Dataset.push_to_hub - no
manual WAV files / metadata.jsonl / audiofolder needed). Group the resulting
repos into a Collection on your profile afterward (a Collection is just a
grouping of existing repos, not something this script needs to create) -
adding a new source later just means running this again with that source
added to --sources; already-pushed repos are skipped automatically (see
--force).

This is deliberately one full repo per source rather than one combined
dataset: each source is small enough (7-28GB) to download and train on
individually, without the shuffle-buffer/random-shard-access cost a single
~74GB combined dataset hits when streamed, or the "must fully download 74GB
before any training can start" cost when loaded non-streaming. It also
matches training one source at a time (a sequential curriculum) rather than
an interleaved mix.

Intended for the sources small enough to fit on local disk - NOT for YODAS
(~710GB). Default sources are the four that comfortably fit a 100GB budget:
mcv (~28GB), mls (~28GB), voxpopuli (~7GB), mtedx (~11GB).

Sources are processed ONE AT A TIME: materialize -> push -> (by default)
delete that source's local cache -> move to the next source. This keeps peak
local disk usage down to whichever single source is largest (~28GB), not the
sum of all of them, since nothing needs multiple sources in memory/on-disk
at once anymore (there's no concatenation step).

Each repo gets a single "train" split only - no train/validation split is
carved out of this data. These manifests are all train-only pseudo-labelled
data with no accompanying held-out split, and bofenghuang's own distillation
run didn't eval against a slice of it either (their TensorBoard logs show
eval against independent standard validation sets - MLS-validation, FLEURS,
Common Voice Italian-validation). Point `--eval_dataset_name`/
`--eval_split_name` at one of those in run_distillation_patch.py instead of
expecting an eval split here.

Usage:
  uv run python materialize_local_dataset.py \\
    --sources mcv mls voxpopuli mtedx \\
    --wer-threshold 20 \\
    --cache-dir /content/materialize-cache \\
    --push-repo-template "YOUR-USERNAME/it-{source}-pseudo-labeled-v0.1"

Adding a source later, e.g. once you're ready for YODAS:
  uv run python materialize_local_dataset.py \\
    --sources mcv mls voxpopuli mtedx yodas-it000 \\
    --cache-dir /content/materialize-cache \\
    --push-repo-template "YOUR-USERNAME/it-{source}-pseudo-labeled-v0.1"
  # mcv/mls/voxpopuli/mtedx are skipped automatically - their repos already exist.
"""

import argparse
import os
import shutil

from datasets import Dataset
from huggingface_hub import HfApi

from stream_pseudo_labeled_dataset import FEATURES, IT_MANIFESTS, REPO_ID, stream_pseudo_labeled


def materialize_source(
    repo_id: str,
    manifest_path: str,
    source: str,
    wer_threshold: float,
    cache_dir: str,
    max_in_flight: int,
    max_samples: int | None = None,
) -> Dataset:
    print(f"[materialize] starting source={source!r} (this blocks until the whole source is fetched)")
    ds = Dataset.from_generator(
        stream_pseudo_labeled,
        features=FEATURES,
        cache_dir=cache_dir,
        gen_kwargs={
            "repo_id": repo_id,
            "manifest_path": manifest_path,
            "wer_threshold": wer_threshold,
            "source": source,
            "max_in_flight": max_in_flight,
            "max_samples": max_samples,
        },
    )
    print(f"[materialize] source={source!r} done: {len(ds)} examples")
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["mcv", "mls", "voxpopuli", "mtedx"],
        help="Keys into IT_MANIFESTS to materialize, each pushed to its own repo. Defaults to "
        "the four sources small enough to fit a 100GB disk budget - deliberately excludes "
        "yodas-it000/100/101 (~710GB).",
    )
    parser.add_argument("--wer-threshold", type=float, default=20.0)
    parser.add_argument("--max-in-flight", type=int, default=8)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle complete rows reproducibly before upload. Omit to preserve manifest order. "
        "The published order is fixed across training passes; this does not disable trainer shuffling.",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=None,
        help="Cap examples per source (useful for a quick test run before committing to the full fetch).",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Local directory for the Arrow cache built while materializing each source. Since "
        "sources are processed one at a time and cleaned up between them (see --keep-cache), "
        "this only needs to fit the LARGEST single source (~28GB), not all of them combined.",
    )
    parser.add_argument(
        "--push-repo-template",
        required=True,
        help="Template for each source's own Hub repo, with {source} as a placeholder, e.g. "
        "'YOUR-USERNAME/it-{source}-pseudo-labeled-v0.1' -> "
        "'YOUR-USERNAME/it-mcv-pseudo-labeled-v0.1', 'YOUR-USERNAME/it-mtedx-pseudo-labeled-v0.1', etc.",
    )
    parser.add_argument("--private", action="store_true", help="Push as private dataset repos.")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Don't delete each source's local cache after it's pushed. By default it's removed "
        "right after each source's push succeeds (before moving to the next source), since "
        "load_dataset() will download its own separate copy from the Hub for training anyway.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-materialize and re-push a source even if its repo already exists. By default, "
        "existing repos are skipped - so adding a new source later (e.g. --sources mcv mls "
        "voxpopuli mtedx yodas-it000) doesn't redo the ones already pushed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api = HfApi()

    unknown = set(args.sources) - set(IT_MANIFESTS)
    if unknown:
        raise ValueError(f"Unknown source(s) {unknown}; valid choices are {list(IT_MANIFESTS)}")

    for name in args.sources:
        push_repo_id = args.push_repo_template.format(source=name)

        if not args.force and api.repo_exists(push_repo_id, repo_type="dataset"):
            print(f"[materialize] skipping source={name!r} - {push_repo_id} already exists (pass --force to redo it)")
            continue

        source_cache_dir = os.path.join(args.cache_dir, name)
        os.makedirs(source_cache_dir, exist_ok=True)

        ds = materialize_source(
            args.repo_id,
            IT_MANIFESTS[name],
            name,
            args.wer_threshold,
            source_cache_dir,
            args.max_in_flight,
            max_samples=args.max_samples_per_source,
        )
        print(ds)

        if args.shuffle_seed is not None:
            print(f"[materialize] shuffling source={name!r} with seed={args.shuffle_seed}")
            # Shuffle row indices, keeping audio, labels and explicit previous
            # context together. Parquet export writes this logical row order;
            # no extra full-sized flatten_indices cache is needed here.
            ds = ds.shuffle(seed=args.shuffle_seed)

        print(f"[materialize] pushing source={name!r} to {push_repo_id} (private={args.private})")
        ds.push_to_hub(push_repo_id, private=args.private)
        print(f"[materialize] push succeeded: {push_repo_id}")

        if args.keep_cache:
            print(f"[materialize] --keep-cache set, leaving local cache at {source_cache_dir}")
        else:
            print(f"[materialize] removing local cache at {source_cache_dir} (pass --keep-cache to skip this)")
            # Drop our reference first - on Linux this isn't strictly required
            # (unlinking an open file just keeps the inode alive until the
            # last fd closes), but it's cheap insurance and avoids relying on
            # that.
            del ds
            try:
                shutil.rmtree(source_cache_dir)
            except OSError as e:
                print(f"[materialize] WARNING: failed to remove {source_cache_dir}: {e}")

    print("[materialize] all sources done.")


if __name__ == "__main__":
    main()
