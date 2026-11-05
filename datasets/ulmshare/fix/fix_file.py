# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5py",
#   "hdf5plugin",
#   "jax",
#   "numpy",
#   "zea @ git+https://github.com/tue-bmd/zea@e0aaee93b8c100ab560b2d07c3f161b30ed5999b",
# ]
# ///
"""Fix ULMShare zea files in place (open-h/Open-H-RF#59).

    uv run fix_file.py FILE.hdf5 [FILE.hdf5 ...] --metadata FRDR_DIR [--dry-run]

For every file:
- probe_geometry: the L22-14v(X) pitch is 0.10 mm, not 0.08 mm (GEL818iD is unchanged);
- t0_delays: recomputed from that geometry;
- initial_times: 2 * startDepth / fc instead of 0;
- metadata/annotations/anatomy: new, "brain";
- metadata/text_report: mice 55-57 were not air-puff; replaced by FRDR's procedure_info;
- custom/nb_acq: new, the frames per acquisition buffer;
- raw_data: Q shifted back to the I sample times (https://github.com/tue-bmd/zea/pull/672).

raw_data is compressed, so this writes a full new copy next to the file (peak disk: twice the
file) and then replaces the file.

Each file's settings come from sequence.json and acquisition.json in
FRDR_DIR/mouse_N/acquisition_M (see README.md).
Anything unexpected raises before a file is changed. Files that already have
custom/nb_acq are skipped, so an interrupted run over the dataset can simply be restarted.
"""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("BLOSC_NTHREADS", "16")

import h5py
import hdf5plugin  # noqa: F401  (blosc filter)
import numpy as np
from zea.beamform.delays import compute_t0_delays_planewave
from zea.data.convert.verasonics import _align_baseband_quadrature

# Probe: (pitch stored in the published files, correct pitch), in metres.
# L22-14v:
# https://verasonics.com/wp-content/uploads/2017/08/L22-14v_data_sheet_2017U.pdf
# L22-14vX:
# https://verasonics.com/wp-content/uploads/2021/11/Single-Crystal-High-Frequency-Linear-Arrays-Data-Sheet-October-2021.pdf
# GEL818iD: no manufacturer source found. ULMShare's 168 elements at 0.15 mm (25.2 mm) fits the
# 25 mm aperture listed by a reseller: https://www.standardultrasound.com/probes/l8-18i-d
PITCH = {
    "L22-14v": (0.08e-3, 0.10e-3),
    "L22-14vX": (0.08e-3, 0.10e-3),
    "GEL818iD": (0.15e-3, 0.15e-3),
}
NOT_AIR_PUFF = {"mouse_55", "mouse_56", "mouse_57"}
GEOMETRY = "probe/probe_geometry"
REPORT = "metadata/text_report"
SCAN = "tracks/track_0/scan"
RAW = "tracks/track_0/data/raw_data"
FRAMES_PER_READ = 400
LIBVER = ("earliest", "v114")  # as zea writes, so older HDF5 installs can open the files


def load_sequence(path, metadata):
    mouse, acquisition = re.match(r"(mouse_\d+)_(acquisition_\d+)_zea", Path(path).name).groups()
    folder = Path(metadata) / mouse / acquisition
    return json.loads((folder / "acquisition.json").read_text()) | json.loads(
        (folder / "sequence.json").read_text()
    )


def corrected_values(f, seq):
    """Check an unfixed file and return its corrected datasets as {path: values}."""
    scan = f[SCAN]
    fc = float(scan["center_frequency"][()])
    probe = f["probe/name"][()].decode()
    assert probe == seq["transducer"], (probe, seq["transducer"])
    assert abs(fc - float(seq["frequency_MHz"]) * 1e6) < 1e3, (fc, seq["frequency_MHz"])
    assert not np.any(scan["initial_times"][()]), "initial_times already set"
    assert "annotations" not in f["metadata"], "annotations already set"

    def t0_delays(geometry):
        return compute_t0_delays_planewave(
            probe_geometry=geometry,
            polar_angles=scan["polar_angles"][()],
            azimuth_angles=scan["azimuth_angles"][()],
            sound_speed=float(scan["sound_speed"][()]),
        ).astype(np.float32)

    # Reproducing the stored delays shows this is the recipe the conversion used.
    geometry = f[GEOMETRY][()]
    np.testing.assert_allclose(t0_delays(geometry), scan["t0_delays"][()], atol=1e-10)
    old_pitch, new_pitch = PITCH[probe]
    assert not np.any(geometry[:, 1:]), "expected a linear array along x"
    np.testing.assert_allclose(np.diff(geometry[:, 0]), old_pitch, atol=1e-9)

    start_time = 2 * float(seq["startDepth_wavelengths"]) / fc
    values = {f"{SCAN}/initial_times": np.full_like(scan["initial_times"][()], start_time)}
    if new_pitch != old_pitch:
        new_geometry = np.zeros_like(geometry)
        new_geometry[:, 0] = (np.arange(len(geometry)) - (len(geometry) - 1) / 2) * new_pitch
        values[GEOMETRY] = new_geometry
        values[f"{SCAN}/t0_delays"] = t0_delays(new_geometry)
    report = f[REPORT][()].decode() if REPORT in f else ""
    if f["metadata/subject/id"][()].decode() in NOT_AIR_PUFF and "air-puff" in report:
        assert "puff" not in seq["procedure_info"].lower(), seq["procedure_info"]
        values[REPORT] = seq["procedure_info"]
    return values


def write_values(f, values, seq):
    for path, value in values.items():
        f[path][...] = value
    anatomy = (
        f["metadata"]
        .create_group("annotations")
        .create_dataset("anatomy", data="brain", dtype=f["metadata/credit"].dtype)
    )
    anatomy.attrs.update(unit="–", description="Anatomical structure imaged (e.g. carotid, liver).")
    # Stored as text like the other custom fields. Skipping a finished file relies on it.
    custom = f["custom"]
    custom.create_dataset("nb_acq", data=seq["acquisition"]["nbAcq"], dtype=custom["npulse"].dtype)


def copy_except_raw(src, dst):
    """Copy everything, with attributes, creating raw_data empty with its chunks and filters."""
    dst.attrs.update(src.attrs)
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            copy_except_raw(item, dst.create_group(name))
        elif item.name == f"/{RAW}":
            plist = item.id.get_create_plist()
            new = h5py.h5d.create(
                dst.id, name.encode(), item.id.get_type(), item.id.get_space(), dcpl=plist
            )
            h5py.Dataset(new).attrs.update(item.attrs)
        else:
            src.copy(item, dst, name=name)


def q_delay(sampling):
    """How far Q lags I, in IQ samples, for BS100BW or BS50BW."""
    sample_mode = int(re.fullmatch(r"BS(100|50)BW", sampling).group(1))
    # Same formula as zea, which splits it over these two lines:
    # https://github.com/tue-bmd/zea/blob/e0aaee93b8c100ab560b2d07c3f161b30ed5999b/zea/data/convert/verasonics.py#L810
    # https://github.com/tue-bmd/zea/blob/e0aaee93b8c100ab560b2d07c3f161b30ed5999b/zea/data/convert/verasonics.py#L223
    return 0.25 / (100 // sample_mode)


def write_aligned_copy(src, dst, delay):
    copy_except_raw(src, dst)
    raw_in, raw_out = src[RAW], dst[RAW]
    for start in range(0, raw_in.shape[0], FRAMES_PER_READ):
        raw = raw_in[start : start + FRAMES_PER_READ]
        iq = _align_baseband_quadrature(raw.astype(np.float32), axis=2, q_delay=delay)
        raw[..., 1] = np.clip(np.round(iq[..., 1]), -32768, 32767)
        raw_out[start : start + FRAMES_PER_READ] = raw


def fix_file(path, metadata, dry_run=False):
    path = Path(path)
    seq = load_sequence(path, metadata)
    with h5py.File(path, "r") as f:
        if "nb_acq" in f["custom"]:
            print(f"{path.name}: already fixed, skipping")
            return
        values = corrected_values(f, seq)
        changed = ", ".join(sorted(k.rsplit("/", 1)[-1] for k in values))
        print(
            f"{path.name}: {seq['transducer']}, updating {changed}, anatomy, nb_acq, raw_data",
            flush=True,
        )
        if dry_run:
            return
        tmp = path.with_name(path.name + ".tmp")
        with h5py.File(tmp, "w", libver=LIBVER) as out:
            write_aligned_copy(f, out, q_delay(seq["sampling"]))
            write_values(out, values, seq)
    tmp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--metadata", required=True, help="directory with the FRDR json files")
    parser.add_argument("--dry-run", action="store_true", help="run all checks, change nothing")
    args = parser.parse_args()
    for file in args.files:
        fix_file(file, args.metadata, args.dry_run)
