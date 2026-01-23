# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from inspect import getfullargspec

import numpy as np

from ..annotations import _sync_onset
from ..utils import _check_eeglabio_installed

_check_eeglabio_installed()
import eeglabio.epochs  # noqa: E402
import eeglabio.raw  # noqa: E402


def _export_raw(fname, raw, *, ica=None):
    # load data first
    raw.load_data()

    # remove extra epoc and STI channels
    drop_chs = ["epoc"]
    # filenames attribute of RawArray is filled with None
    if raw.filenames[0] and raw.filenames[0].suffix != ".fif":
        drop_chs.append("STI 014")

    ch_names = [ch for ch in raw.ch_names if ch not in drop_chs]
    cart_coords = _get_als_coords_from_chs(raw.info["chs"], drop_chs)

    if raw.annotations:
        annotations = [
            raw.annotations.description,
            # subtract raw.first_time because EEGLAB marks events starting from
            # the first available data point and ignores raw.first_time
            _sync_onset(raw, raw.annotations.onset, inverse=False),
            raw.annotations.duration,
        ]
    else:
        annotations = None

    kwargs = dict()
    have_kwargs = getfullargspec(eeglabio.raw.export_set).kwonlyargs
    if ica is not None:
        if "icaweights" in have_kwargs:
            icaweights, icasphere, icawinv = _mne_ica_to_eeglab(ica)
            kwargs["icaweights"] = icaweights
            kwargs["icasphere"] = icasphere
            kwargs["icawinv"] = icawinv
        else:
            # TODO: confirm that 0.1.4 is the correct pin before merge
            raise RuntimeError(
                "To export ICA to eeglab format, eeglabio version 0.1.4 is required. "
                f"You have version {eeglabio.__version__}"
            )

    eeglabio.raw.export_set(
        fname,
        data=raw.get_data(picks=ch_names),
        sfreq=raw.info["sfreq"],
        ch_names=ch_names,
        ch_locs=cart_coords,
        annotations=annotations,
        **kwargs,
    )


def _export_epochs(fname, epochs, *, ica=None):
    _check_eeglabio_installed()
    # load data first
    epochs.load_data()

    # remove extra epoc and STI channels
    drop_chs = ["epoc", "STI 014"]
    ch_names = [ch for ch in epochs.ch_names if ch not in drop_chs]
    cart_coords = _get_als_coords_from_chs(epochs.info["chs"], drop_chs)

    if epochs.annotations:
        annot = [
            epochs.annotations.description,
            epochs.annotations.onset,
            epochs.annotations.duration,
        ]
    else:
        annot = None

    # https://github.com/jackz314/eeglabio/pull/18
    kwargs = dict()
    have_kwargs = getfullargspec(eeglabio.epochs.export_set).kwonlyargs
    if "epoch_indices" in have_kwargs:
        kwargs["epoch_indices"] = epochs.selection
    if ica is not None:
        if "icaweights" in have_kwargs:
            icaweights, icasphere, icawinv = _mne_ica_to_eeglab(ica)
            kwargs["icaweights"] = icaweights
            kwargs["icasphere"] = icasphere
            kwargs["icawinv"] = icawinv
        else:
            # TODO: confirm that 0.1.4 is the correct pin before merge
            raise RuntimeError(
                "To export ICA to eeglab format, eeglabio version 0.1.4 is required. "
                f"You have version {eeglabio.__version__}"
            )

    eeglabio.epochs.export_set(
        fname,
        data=epochs.get_data(picks=ch_names),
        sfreq=epochs.info["sfreq"],
        events=epochs.events,
        tmin=epochs.tmin,
        tmax=epochs.tmax,
        ch_names=ch_names,
        event_id=epochs.event_id,
        ch_locs=cart_coords,
        annotations=annot,
        **kwargs,
    )


def _mne_ica_to_eeglab(ica):
    n_comp = ica.n_components_
    n_ch = len(ica.ch_names)
    pre_whitener = ica.pre_whitener_
    if pre_whitener.shape == (n_ch, 1):
        pre_whitener = np.diag(pre_whitener[:, 0])
    elif pre_whitener.shape != (n_ch, n_ch):
        raise RuntimeError()

    P = ica.pca_components_[:n_comp, :]
    icasphere = P @ pre_whitener
    icaweights = ica.unmixing_matrix_

    W_mne = icaweights @ icasphere
    icawinv = np.linalg.pinv(W_mne)
    return icaweights, icasphere, icawinv


def _get_als_coords_from_chs(chs, drop_chs=None):
    """Extract channel locations in ALS format (x, y, z) from a chs instance.

    Returns
    -------
    None if no valid coordinates are found (all zeros)
    """
    if drop_chs is None:
        drop_chs = []
    cart_coords = np.array([d["loc"][:3] for d in chs if d["ch_name"] not in drop_chs])
    if cart_coords.any():  # has coordinates
        # (-y x z) to (x y z)
        cart_coords[:, 0] = -cart_coords[:, 0]  # -y to y
        # swap x (1) and y (0)
        cart_coords[:, [0, 1]] = cart_coords[:, [1, 0]]
    else:
        cart_coords = None
    return cart_coords
