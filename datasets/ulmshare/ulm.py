# SPDX-License-Identifier: Apache-2.0
"""A minimal ULM chain on clutter-filtered IQ buffers (NumPy/SciPy, Keras for the Hessian).

Per buffer of the clutter-filtered movie from ``reconstruct.py``: enhance, localize
bubbles in every frame, link them into tracks, and draw the tracks into a density
map. The first two steps follow ULMShare's own chain (``example_script.m`` with
the TAL toolbox).

1. **Enhance** (:func:`lag1`). Lag-1 autocorrelation over a short window: noise
   decorrelates from one frame to the next and averages out, a bubble does not.
2. **Mask** (:func:`vesselness`). A bubble moving along a vessel traces a bright
   tube through the ``(z, x, t)`` volume; TAL's spatio-temporal Jerman filter
   marks such tubes, and only peaks inside them count.
3. **Localize** (:func:`localize`). The noise of the filtered movie changes with
   depth (TGC, attenuation, the skull), so each depth row has its own floor,
   taken from frame-to-frame differences (:func:`noise_floor`); local maxima
   above a multiple of it are bubbles, refined to sub-pixel by a 3-point Gaussian
   fit per axis.
4. **Track** (:func:`track`). The simple tracker of deep-ulm-pipeline
   (``tracking/simple/tracker.py``): frame by frame, a Hungarian assignment of
   detections to the last position of each open track, gated at
   ``max_linking_distance``; tracks missing more than ``max_gap`` frames close,
   unmatched detections open new tracks, and only tracks of ``>= min_length``
   detections are kept.
5. **Render** (:func:`density_map`). Each track is smoothed with a moving
   average, resampled densely along its path, and adds 1 to every super-resolved
   pixel it crosses -- once per track, so a slow bubble does not paint a blob
   (as deep-ulm-pipeline's ``tracks_to_map``).

Everything works in **beamforming-grid pixel coordinates** ``(z, x)`` and frames;
distances and thresholds are therefore in pixels.
"""

import keras
import numpy as np
from scipy.ndimage import convolve1d, maximum_filter, uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def lag1(iq_cf, window=7):
    """Lag-1 autocorrelation enhancement of one filtered buffer ``(n, Nz, Nx)``.

    As ULMShare's ``example_script.m`` (``useLag1``): ``sqrt(|h * (x_t conj(x_{t+1}))|)``
    with ``h`` a normalised ``hanning(window)`` along time. Noise decorrelates from one
    frame to the next and averages out; a bubble, moving less than its PSF per frame,
    does not. Returns a real ``(n - 1, Nz, Nx)`` magnitude.
    """
    h = np.hanning(window + 2)[1:-1]  # MATLAB's hanning(n) excludes the zero endpoints
    prod = iq_cf[:-1] * np.conj(iq_cf[1:])
    return np.sqrt(np.abs(convolve1d(prod, h / h.sum(), axis=0, mode="nearest")))


def _gaussian_kernels(sigma):
    """Sampled Gaussian and its first and second derivatives, radius 3 sigma."""
    r = int(np.ceil(3 * sigma))
    u = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (u / sigma) ** 2)
    g /= g.sum()
    return [g, -u / sigma**2 * g, (u**2 / sigma**4 - 1 / sigma**2) * g]


def _filter_axis(x, kernel, axis):
    """Correlate ``x`` with a 1D ``kernel`` along ``axis`` (edges replicated), on the backend."""
    r = len(kernel) // 2
    x = keras.ops.moveaxis(x, axis, -1)
    shape = keras.ops.shape(x)
    x = keras.ops.reshape(x, (-1, shape[-1], 1))
    x = keras.ops.concatenate(
        [keras.ops.repeat(x[:, :1], r, 1), x, keras.ops.repeat(x[:, -1:], r, 1)], 1
    )
    k = keras.ops.convert_to_tensor(kernel[::-1].reshape(-1, 1, 1).astype("float32"))
    x = keras.ops.conv(x, k, padding="valid")
    return keras.ops.moveaxis(keras.ops.reshape(x, shape), -1, axis)


def _eigenvalues_3x3(a, b, c, d, e, f):
    """Eigenvalues of the symmetric ``[[a, d, e], [d, b, f], [e, f, c]]``, sorted by ``|.|``."""
    ops = keras.ops
    q = (a + b + c) / 3
    p = ops.sqrt(
        ops.maximum(
            ((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * (d * d + e * e + f * f)) / 6, 1e-30
        )
    )
    A, B, C, D, E, F = (a - q) / p, (b - q) / p, (c - q) / p, d / p, e / p, f / p
    half_det = (A * (B * C - F * F) - D * (D * C - F * E) + E * (D * F - B * E)) / 2
    phi = ops.arccos(ops.clip(half_det, -1.0, 1.0)) / 3
    e1 = q + 2 * p * ops.cos(phi)
    e3 = q + 2 * p * ops.cos(phi + 2 * np.pi / 3)
    ev = ops.stack([e1, 3 * q - e1 - e3, e3], axis=0)
    return ops.take_along_axis(ev, ops.argsort(ops.abs(ev), axis=0), axis=0)


def vesselness(mag, sigma=4.0, epsilon=1.4, tau=0.4, tile=64):
    """Spatio-temporal vesselness of one enhanced buffer ``(n, Nz, Nx)``, in ``[0, 1]``.

    A bubble moving along a vessel traces a bright tube through the ``(z, x, t)``
    volume; speckle and flicker do not. This is the Jerman filter TAL runs on that
    volume before tracking (``lib/trajectories_filter.m``): the Hessian at scale
    ``sigma`` (pixels and frames), its spatial terms weighted by ``epsilon`` against
    the temporal ones, and ``tau`` the response uniformity. The Hessian runs on the
    Keras backend (the GPU when there is one), ``tile`` depth rows at a time to
    bound memory.
    """
    k = _gaussian_kernels(sigma)
    r = len(k[0]) // 2
    mag = np.asarray(mag, dtype=np.float32)
    nz = mag.shape[1]
    l2, l3 = np.empty_like(mag), np.empty_like(mag)
    s, e2 = sigma**2, epsilon**2
    for z0 in range(0, nz, tile):
        z1 = min(z0 + tile, nz)
        lo, hi = max(z0 - r, 0), min(z1 + r, nz)  # halo: the depth filter needs r rows each side
        x = keras.ops.convert_to_tensor(mag[:, lo:hi])
        by_t = [_filter_axis(x, k[o], 0) for o in range(3)]

        def h(ot, oz, ox):
            return _filter_axis(_filter_axis(by_t[ot], k[oz], 1), k[ox], 2)

        ev = _eigenvalues_3x3(
            s * e2 * h(0, 2, 0),
            s * e2 * h(0, 0, 2),
            s * h(2, 0, 0),
            s * e2 * h(0, 1, 1),
            s * epsilon * h(1, 1, 0),
            s * epsilon * h(1, 0, 1),
        )
        ev = keras.ops.convert_to_numpy(ev)[:, :, z0 - lo : z1 - lo]
        l2[:, z0:z1], l3[:, z0:z1] = -ev[1], -ev[2]  # bright structures on a dark background
    # Jerman's response, with the regularised third eigenvalue
    rho = np.where((l3 > 0) & (l3 <= tau * l3.max()), tau * l3.max(), l3)
    rho[l3 <= 0] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        response = l2 * l2 * (rho - l2) * 27 / (l2 + rho) ** 3
    response[(l2 >= rho / 2) & (rho > 0)] = 1
    response[(l2 <= 0) | (rho <= 0) | ~np.isfinite(response)] = 0
    return response / max(response.max(), 1e-30)


def noise_floor(iq_cf):
    """Noise level of each depth row of one filtered buffer ``(n, Nz, Nx)``, shape ``(1, Nz, 1)``.

    From frame-to-frame differences: white noise changes completely from one frame to
    the next, while bubbles (moving less than their PSF per frame) and tissue mostly
    cancel. So unlike the median ``|IQ|``, which the bubble cloud lifts two- to
    threefold during the bolus, this stays close to the noise before, during and
    after it.
    """
    diff = np.abs(np.diff(iq_cf, axis=0))
    return np.median(diff, axis=(0, 2), keepdims=True).astype(np.float32) / np.sqrt(2)


def localize(iq_cf, threshold, min_distance, mask=None, noise=None):
    """Bubble positions in every frame of one filtered buffer ``(n, Nz, Nx)``.

    A bubble is a local maximum in a ``(2 * min_distance + 1)`` square window whose
    ``|IQ|`` exceeds ``threshold`` times the ``noise`` of its depth row (by default
    the row's median ``|IQ|``), and, given a boolean ``mask`` of the same shape, lies
    inside it. Returns a list with, per frame, a ``(k, 2)`` array of sub-pixel ``(z, x)``.
    """
    mag = np.abs(iq_cf).astype(np.float32)
    if noise is None:
        noise = np.median(mag, axis=(0, 2), keepdims=True)  # per depth row
    snr = mag / np.maximum(noise, 1e-3 * np.median(noise))
    candidates = snr if mask is None else snr * mask
    width = 2 * min_distance + 1
    local_max = candidates == maximum_filter(candidates, size=(1, width, width))
    peak = local_max & (candidates > threshold)
    peak[:, [0, -1], :] = peak[:, :, [0, -1]] = False  # the fit needs both neighbours
    f, z, x = np.nonzero(peak)

    log = np.log(np.maximum(snr, 1e-6))  # some pixels are exactly 0

    def offset(before, at, after):  # vertex of the parabola through 3 log samples
        curv = before - 2 * at + after
        return np.clip(0.5 * (before - after) / np.where(curv < 0, curv, -np.inf), -0.5, 0.5)

    dz = offset(log[f, z - 1, x], log[f, z, x], log[f, z + 1, x])
    dx = offset(log[f, z, x - 1], log[f, z, x], log[f, z, x + 1])
    points = np.stack([z + dz, x + dx], axis=1)
    return np.split(points, np.searchsorted(f, np.arange(1, len(mag))))


def _assign(last, points, max_distance):
    """Minimum-cost matching of track ends to detections closer than ``max_distance``.

    The same matching as a Hungarian assignment on the full gated cost matrix (as
    many links as possible, then the shortest), solved per connected group of
    nearby ends and detections: after the bolus a frame holds thousands of
    detections, and the full matrix would be cubic in that.
    """
    pairs = cKDTree(last).sparse_distance_matrix(
        cKDTree(points), max_distance, output_type="coo_matrix"
    )
    if pairs.nnz == 0:
        return [], []
    m, k = len(last), len(points)
    graph = coo_matrix((np.ones(pairs.nnz), (pairs.row, m + pairs.col)), shape=(m + k, m + k))
    _, label = connected_components(graph, directed=False)
    rows, cols = [], []
    order = np.argsort(label[pairs.row], kind="stable")
    r, c, d = pairs.row[order], pairs.col[order], pairs.data[order]
    for group in np.split(np.arange(len(r)), np.flatnonzero(np.diff(label[r])) + 1):
        ur, ir = np.unique(r[group], return_inverse=True)
        uc, ic = np.unique(c[group], return_inverse=True)
        cost = np.full((len(ur), len(uc)), 1e9)  # out of reach: never linked
        cost[ir, ic] = d[group]
        a, b = linear_sum_assignment(cost)
        keep = cost[a, b] < 1e9
        rows += list(ur[a[keep]])
        cols += list(uc[b[keep]])
    return rows, cols


def track(locs, max_linking_distance, max_gap, min_length):
    """Link per-frame detections into tracks.

    ``locs`` is :func:`localize`'s output. Returns a list of ``(len, 3)`` arrays of
    ``(frame, z, x)``, with ``frame`` counted from the start of ``locs``.
    """
    open_tracks, closed = [], []  # each track: [list of (frame, z, x), frames missed]
    for frame, points in enumerate(locs):
        matched = np.zeros(len(points), bool)
        for t in open_tracks:
            t[1] += 1
        if open_tracks and len(points):
            last = np.array([t[0][-1][1:] for t in open_tracks])
            # Gated before the assignment, so a far detection can't steal a track
            # from a near one.
            for r, c in zip(*_assign(last, points, max_linking_distance)):
                open_tracks[r][0].append((frame, *points[c]))
                open_tracks[r][1] = 0
                matched[c] = True
        closed += [t for t in open_tracks if t[1] > max_gap]
        open_tracks = [t for t in open_tracks if t[1] <= max_gap]
        open_tracks += [[[(frame, *p)], 0] for p in points[~matched]]
    closed += open_tracks
    return [np.array(t[0], dtype=np.float32) for t in closed if len(t[0]) >= min_length]


def density_map(tracks, grid_shape, super_res, smooth):
    """Number of tracks crossing each pixel of a ``super_res``-times finer grid.

    ``grid_shape`` is the beamforming grid ``(Nz, Nx)``; each track is smoothed
    with a ``smooth``-frame moving average first.
    """
    shape = (grid_shape[0] * super_res, grid_shape[1] * super_res)
    density = np.zeros(shape[0] * shape[1])
    for t in tracks:
        zx = t[:, 1:].astype(np.float64)
        window = min(smooth, len(zx) - 1 + len(zx) % 2)  # odd, at most the track length
        zx = uniform_filter1d(zx, window, axis=0, mode="nearest")
        # Resample every half super-res pixel along the path, so no pixel is skipped.
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(zx, axis=0), axis=1))])
        s = np.arange(0, arc[-1] + 1e-9, 0.5 / super_res)
        pix = np.round(np.stack([np.interp(s, arc, zx[:, k]) for k in (0, 1)], 1) * super_res)
        pix = pix.astype(int)
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < shape[0]) & (pix[:, 1] >= 0) & (pix[:, 1] < shape[1])
        density[np.unique(pix[ok, 0] * shape[1] + pix[ok, 1])] += 1
    return density.reshape(shape)
