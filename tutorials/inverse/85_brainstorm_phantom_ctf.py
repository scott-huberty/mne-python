"""
.. _plot_brainstorm_phantom_ctf:

=======================================
Brainstorm CTF phantom dataset tutorial
=======================================

Here we compute the evoked from raw for the Brainstorm CTF phantom
tutorial dataset. For comparison, see :footcite:`TadelEtAl2011` and:

    https://neuroimage.usc.edu/brainstorm/Tutorials/PhantomCtf

References
----------
.. footbibliography::
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import warnings

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import fit_dipole
from mne.datasets.brainstorm import bst_phantom_ctf
from mne.io import read_raw_ctf

print(__doc__)

# %%
# The data were collected with a CTF system at 2400 Hz.
data_path = bst_phantom_ctf.data_path(verbose=True)

# Switch to these to use the higher-SNR data:
# raw_path = op.join(data_path, 'phantom_200uA_20150709_01.ds')
# dip_freq = 7.
raw_path = data_path / "phantom_20uA_20150603_03.ds"
dip_freq = 23.0
erm_path = data_path / "emptyroom_20150709_01.ds"
raw = read_raw_ctf(raw_path, preload=True)

# %%
# The sinusoidal signal is generated on channel HDAC006, so we can use
# that to obtain precise timing.

sinusoid, times = raw[raw.ch_names.index("HDAC006-4408")]
plt.figure()
plt.plot(times[times < 1.0], sinusoid.T[times < 1.0])

# %%
# Let's create some events using this signal by thresholding the sinusoid.

events = np.where(np.diff(sinusoid > 0.5) > 0)[1] + raw.first_samp
events = np.vstack((events, np.zeros_like(events), np.ones_like(events))).T

# %%
# The CTF software compensation works reasonably well:

raw.plot()

# %%
# But here we can get slightly better noise suppression, lower localization
# bias, and a better dipole goodness of fit with spatio-temporal (tSSS)
# Maxwell filtering:

raw.apply_gradient_compensation(0)  # must un-do software compensation first
mf_kwargs = dict(origin=(0.0, 0.0, 0.0), st_duration=10.0, st_overlap=True)
raw = mne.preprocessing.maxwell_filter(raw, **mf_kwargs)
raw.plot()

# %%
# Our choice of tmin and tmax should capture exactly one cycle, so
# we can make the unusual choice of baselining using the entire epoch
# when creating our evoked data. We also then crop to a single time point
# (@t=0) because this is a peak in our signal.

tmin = -0.5 / dip_freq
tmax = -tmin
epochs = mne.Epochs(
    raw, events, event_id=1, tmin=tmin, tmax=tmax, baseline=(None, None)
)
evoked = epochs.average()
evoked.plot(time_unit="s")
evoked.crop(0.0, 0.0)

# %%
# .. _plt_brainstorm_phantom_ctf_eeg_sphere_geometry:
#
# Let's use a :ref:`sphere head geometry model <eeg_sphere_model>`
# and let's see the coordinate alignment and the sphere location.
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.08)

mne.viz.plot_alignment(
    raw.info, subject="sample", meg="helmet", bem=sphere, dig=True, surfaces=["brain"]
)
del raw, epochs

# %%
# To do a dipole fit, let's use the covariance provided by the empty room
# recording.

raw_erm = read_raw_ctf(erm_path).apply_gradient_compensation(0)
raw_erm = mne.preprocessing.maxwell_filter(raw_erm, coord_frame="meg", **mf_kwargs)
cov = mne.compute_raw_covariance(raw_erm)
del raw_erm

with warnings.catch_warnings(record=True):
    # ignore warning about data rank exceeding that of info (75 > 71)
    warnings.simplefilter("ignore")
    dip, residual = fit_dipole(evoked, cov, sphere, verbose=True)

# %%
# Compare the actual position with the estimated one.

expected_pos = np.array([18.0, 0.0, 49.0])
diff = np.sqrt(np.sum((dip.pos[0] * 1000 - expected_pos) ** 2))
print(f"Actual pos:     {np.array_str(expected_pos, precision=1)} mm")
print(f"Estimated pos:  {np.array_str(dip.pos[0] * 1000, precision=1)} mm")
print(f"Difference:     {diff:0.1f} mm")
print(f"Amplitude:      {1e9 * dip.amplitude[0]:0.1f} nAm")
print(f"GOF:            {dip.gof[0]:0.1f} %")
