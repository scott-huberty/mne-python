# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
#          simplified BSD-3 license
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
import time

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ..._fiff.utils import _create_chs, _read_segments_file
from ...utils import _check_fname, _validate_type, logger, verbose, warn
from ..base import BaseRaw
from .egimff import _read_raw_egi_mff
from .events import _combine_triggers


def _read_header(fid):
    """Read EGI binary header."""
    version = np.fromfile(fid, "<i4", 1)[0]

    if version > 6 & ~np.bitwise_and(version, 6):
        version = version.byteswap().astype(np.uint32)
    else:
        raise ValueError("Watchout. This does not seem to be a simple binary EGI file.")

    def my_fread(*x, **y):
        return int(np.fromfile(*x, **y)[0])

    info = dict(
        version=version,
        year=my_fread(fid, ">i2", 1),
        month=my_fread(fid, ">i2", 1),
        day=my_fread(fid, ">i2", 1),
        hour=my_fread(fid, ">i2", 1),
        minute=my_fread(fid, ">i2", 1),
        second=my_fread(fid, ">i2", 1),
        millisecond=my_fread(fid, ">i4", 1),
        samp_rate=my_fread(fid, ">i2", 1),
        n_channels=my_fread(fid, ">i2", 1),
        gain=my_fread(fid, ">i2", 1),
        bits=my_fread(fid, ">i2", 1),
        value_range=my_fread(fid, ">i2", 1),
    )

    unsegmented = 1 if np.bitwise_and(version, 1) == 0 else 0
    precision = np.bitwise_and(version, 6)
    if precision == 0:
        raise RuntimeError("Floating point precision is undefined.")

    if unsegmented:
        info.update(
            dict(
                n_categories=0,
                n_segments=1,
                n_samples=int(np.fromfile(fid, ">i4", 1)[0]),
                n_events=int(np.fromfile(fid, ">i2", 1)[0]),
                event_codes=[],
                category_names=[],
                category_lengths=[],
                pre_baseline=0,
            )
        )
        for event in range(info["n_events"]):
            event_codes = "".join(np.fromfile(fid, "S1", 4).astype("U1"))
            info["event_codes"].append(event_codes)
    else:
        raise NotImplementedError("Only continuous files are supported")
    info["unsegmented"] = unsegmented
    info["dtype"], info["orig_format"] = {
        2: (">i2", "short"),
        4: (">f4", "float"),
        6: (">f8", "double"),
    }[precision]
    info["dtype"] = np.dtype(info["dtype"])
    return info


def _read_events(fid, info):
    """Read events."""
    events = np.zeros([info["n_events"], info["n_segments"] * info["n_samples"]])
    fid.seek(36 + info["n_events"] * 4, 0)  # skip header
    for si in range(info["n_samples"]):
        # skip data channels
        fid.seek(info["n_channels"] * info["dtype"].itemsize, 1)
        # read event channels
        events[:, si] = np.fromfile(fid, info["dtype"], info["n_events"])
    return events


@verbose
def read_raw_egi(
    input_fname,
    eog=None,
    misc=None,
    include=None,
    exclude=None,
    preload=False,
    channel_naming="E%d",
    *,
    events_as_annotations=False,
    verbose=None,
) -> "RawEGI":
    """Read EGI simple binary as raw object.

    .. note:: This function attempts to create a synthetic trigger channel.
              See the Notes section below.

    Parameters
    ----------
    input_fname : path-like
        Path to the raw file. Files with an extension ``.mff`` are
        automatically considered to be EGI's native MFF format files.
    eog : list or tuple
        Names of channels or list of indices that should be designated
        EOG channels. Default is None.
    misc : list or tuple
        Names of channels or list of indices that should be designated
        MISC channels. Default is None.
    include : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None.
       Note. Overrides ``exclude`` parameter.
    exclude : None | list
       The event channels to be ignored when creating the synthetic
       trigger. Defaults to None. If None, channels that have more than
       one event and the ``sync`` and ``TREV`` channels will be
       ignored.
    %(preload)s

        .. versionadded:: 0.11
    channel_naming : str
        Channel naming convention for the data channels. Defaults to ``'E%%d'``
        (resulting in channel names ``'E1'``, ``'E2'``, ``'E3'``...). The
        effective default prior to 0.14.0 was ``'EEG %%03d'``.
        .. versionadded:: 0.14.0

    events_as_annotations : bool
        If True, annotations are created from experiment events. If False (default),
        synthetic trigger channels are created from experiment events. See the Notes
        section for details.

        .. versionadded:: 1.7.0
    %(verbose)s

    Returns
    -------
    raw : instance of RawEGI
        A Raw object containing EGI data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawEGI.

    Notes
    -----
    The trigger channel names are based on the arbitrary user dependent event
    codes used. However this function will attempt to generate a **synthetic
    trigger channel** named ``STI 014`` in accordance with the general
    Neuromag / MNE naming pattern.

    The event_id assignment equals ``np.arange(n_events) + 1``. The resulting
    ``event_id`` mapping is stored as attribute to the resulting raw object but
    will be ignored when saving to a fiff. Note. The trigger channel is
    artificially constructed based on timestamps received by the Netstation.
    As a consequence, triggers have only short durations.

    This step will fail if events are not mutually exclusive.
    """
    _validate_type(input_fname, "path-like", "input_fname")
    input_fname = str(input_fname)
    if input_fname.rstrip("/\\").endswith(".mff"):  # allows .mff or .mff/
        return _read_raw_egi_mff(
            input_fname,
            eog,
            misc,
            include,
            exclude,
            preload,
            channel_naming,
            events_as_annotations,
            verbose,
        )
    return RawEGI(
        input_fname, eog, misc, include, exclude, preload, channel_naming, verbose
    )


class RawEGI(BaseRaw):
    """Raw object from EGI simple binary file."""

    _extra_attributes = ("event_id",)

    @verbose
    def __init__(
        self,
        input_fname,
        eog=None,
        misc=None,
        include=None,
        exclude=None,
        preload=False,
        channel_naming="E%d",
        verbose=None,
    ):
        input_fname = str(_check_fname(input_fname, "read", True, "input_fname"))
        if eog is None:
            eog = []
        if misc is None:
            misc = []
        with open(input_fname, "rb") as fid:  # 'rb' important for py3k
            logger.info(f"Reading EGI header from {input_fname}...")
            egi_info = _read_header(fid)
            logger.info("    Reading events ...")
            egi_events = _read_events(fid, egi_info)  # update info + jump
            if egi_info["value_range"] != 0 and egi_info["bits"] != 0:
                cal = egi_info["value_range"] / 2.0 ** egi_info["bits"]
            else:
                cal = 1e-6

        logger.info("    Assembling measurement info ...")

        event_codes = []
        if egi_info["n_events"] > 0:
            event_codes = list(egi_info["event_codes"])
            if include is None:
                exclude_list = ["sync", "TREV"] if exclude is None else exclude
                exclude_inds = [
                    i for i, k in enumerate(event_codes) if k in exclude_list
                ]
                more_excludes = []
                if exclude is None:
                    for ii, event in enumerate(egi_events):
                        if event.sum() < 1 and event_codes[ii]:
                            more_excludes.append(ii)
                if len(exclude_inds) + len(more_excludes) == len(event_codes):
                    warn(
                        "Did not find any event code with more than one event.",
                        RuntimeWarning,
                    )
                else:
                    exclude_inds.extend(more_excludes)

                exclude_inds.sort()
                include_ = [
                    i for i in np.arange(egi_info["n_events"]) if i not in exclude_inds
                ]
                include_names = [k for i, k in enumerate(event_codes) if i in include_]
            else:
                include_ = [i for i, k in enumerate(event_codes) if k in include]
                include_names = include

            for kk, v in [("include", include_names), ("exclude", exclude)]:
                if isinstance(v, list):
                    for k in v:
                        if k not in event_codes:
                            raise ValueError(f'Could find event named "{k}"')
                elif v is not None:
                    raise ValueError(f"`{kk}` must be None or of type list")

            event_ids = np.arange(len(include_)) + 1
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            excl_events = ", ".join(
                k for i, k in enumerate(event_codes) if i not in include_
            )
            logger.info(f"    Excluding events {{{excl_events}}} ...")
            egi_info["new_trigger"] = _combine_triggers(
                egi_events[include_], remapping=event_ids
            )
            self.event_id = dict(
                zip([e for e in event_codes if e in include_names], event_ids)
            )
        else:
            # No events
            self.event_id = None
            egi_info["new_trigger"] = None
        info = _empty_info(egi_info["samp_rate"])
        my_time = datetime.datetime(
            egi_info["year"],
            egi_info["month"],
            egi_info["day"],
            egi_info["hour"],
            egi_info["minute"],
            egi_info["second"],
        )
        my_timestamp = time.mktime(my_time.timetuple())
        info["meas_date"] = (my_timestamp, 0)
        ch_names = [channel_naming % (i + 1) for i in range(egi_info["n_channels"])]
        ch_names.extend(list(egi_info["event_codes"]))
        if egi_info["new_trigger"] is not None:
            ch_names.append("STI 014")  # our new_trigger
        nchan = len(ch_names)
        cals = np.repeat(cal, nchan)
        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)
        sti_ch_idx = [
            i
            for i, name in enumerate(ch_names)
            if name.startswith("STI") or name in event_codes
        ]
        for idx in sti_ch_idx:
            chs[idx].update(
                {
                    "unit_mul": FIFF.FIFF_UNITM_NONE,
                    "cal": 1.0,
                    "kind": FIFF.FIFFV_STIM_CH,
                    "coil_type": FIFF.FIFFV_COIL_NONE,
                    "unit": FIFF.FIFF_UNIT_NONE,
                    "loc": np.zeros(12),
                }
            )
        info["chs"] = chs
        info._unlocked = False
        info._update_redundant()
        orig_format = (
            egi_info["orig_format"] if egi_info["orig_format"] != "float" else "single"
        )
        super().__init__(
            info,
            preload,
            orig_format=orig_format,
            filenames=[input_fname],
            last_samps=[egi_info["n_samples"] - 1],
            raw_extras=[egi_info],
            verbose=verbose,
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        egi_info = self._raw_extras[fi]
        dtype = egi_info["dtype"]
        n_chan_read = egi_info["n_channels"] + egi_info["n_events"]
        offset = 36 + egi_info["n_events"] * 4
        trigger_ch = egi_info["new_trigger"]
        _read_segments_file(
            self,
            data,
            idx,
            fi,
            start,
            stop,
            cals,
            mult,
            dtype=dtype,
            n_channels=n_chan_read,
            offset=offset,
            trigger_ch=trigger_ch,
        )
