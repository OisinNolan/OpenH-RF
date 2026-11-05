# Fix the ULMShare files (open-h/Open-H-RF#59)

`fix_file.py` corrects a ULMShare file in place. Run it from a checkout of this repo. `uv` installs what it needs, including the pinned `zea` commit that has the IQ fix.

## 1. Get the files

Download the `.hdf5` files from `nvidia/Open-H-RF`, under `ulmshare/data/`, into `data/`.
Keep their original names, since the script uses them to find each file's settings.

## 2. Get the FRDR metadata

Each file needs its acquisition's `sequence.json` and `acquisition.json` from the ULMShare FRDR record:

```bash
BASE=https://g-8bdcf1.cd4fe.0ec8.data.globus.org/33/published/publication_1545/submitted_data
for f in data/*.hdf5; do
  n=$(basename "$f"); m=${n%%_acquisition*}; a=acquisition_${n#*_acquisition_}; a=${a%%_zea*}
  for j in sequence acquisition; do
    curl -sSf --create-dirs -o "metadata/$m/$a/$j.json" "$BASE/$m/$a/$j.json" || echo "FAILED $m/$a/$j"
  done
done
```

## 3. Run

```bash
uv run datasets/ulmshare/fix/fix_file.py data/*.hdf5 --metadata metadata --dry-run
uv run datasets/ulmshare/fix/fix_file.py data/*.hdf5 --metadata metadata
```

The dry run checks every file and changes nothing. The real run takes about an hour per 160,000
frames run alone, and about twice that with three files running at once. It writes a new copy of
each file, so it needs free disk equal to that file's size.

A file that already has `custom/nb_acq` is skipped, so an interrupted run can simply be restarted.
Back up anything you can't download again, because the fix replaces the original.

Files run one at a time. If you have many cores and fast storage, you could run them in parallel:

```bash
ls data/*.hdf5 | xargs -P 8 -I{} uv run datasets/ulmshare/fix/fix_file.py {} --metadata metadata
```

This is untested at this scale. Each process's FFT uses every core, so they compete, and capping it
with `scipy.fft.set_workers` in `fix_file.py` may help. Free disk must cover every file in progress,
up to about 3 TB for 8 files. A GPU wouldn't help much: on one file here, about 80% of the time went
to compressing and writing.

## What changes

| Field | Change | Probes |
|---|---|---|
| `probe/probe_geometry` | pitch 0.08 -> 0.10 mm | L22-14v, L22-14vX |
| `scan/t0_delays` | recomputed from the new geometry | L22-14v, L22-14vX |
| `scan/initial_times` | 0 -> `2 * startDepth / fc` | all |
| `metadata/annotations/anatomy` | new, `brain` | all |
| `metadata/text_report` | mice 55-57 only (6 files): air-puff text -> FRDR's `procedure_info` | L22-14vX |
| `custom/nb_acq` | new, frames per acquisition buffer | all |
| `raw_data` | Q shifted to the I sample times (zea#672) | all |
