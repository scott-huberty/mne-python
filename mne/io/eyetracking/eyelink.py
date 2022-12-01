# Authors: Dominik Welke <dominik.welke@web.de>
#          Scott Huberty <seh33@uw.edu>
#
# License: BSD-3-Clause

from pathlib import Path
import numpy as np
import pandas as pd

from ..base import BaseRaw
from ..meas_info import create_info
from ..constants import FIFF
from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc


# from ...preprocessing.interpolate import interpolate_nan
# from mne.preprocessing import annotate_nan


MODE = {'CR': 'corneal reflection', 'P': 'pupil'}

FILTER_TYPE = {'0': 'filter off',
               '1': 'standard filter',
               '2': 'extra filter'}

# Leading A is for Arm
# next R is for Remote
# B is for Binocular/Monocular.
# M is for Monocular only
# Final R means illuminator on Right (legacy systems)
MOUNT = {'MTABLER': 'Desktop, Stabilized Head, Monocular',
         'BTABLER': 'Desktop, Stabilized Head, Binocular/Monocular',
         'RTABLER': 'Desktop (Remote mode), Target Sticker, Monocular',
         'RBTABLER': ('Desktop (Remote mode), Target Sticker, '
                      'Binocular/Monocular'),
         'AMTABLER': 'Arm Mount, Stabilized Head, Monocular',
         'ABTABLER': 'Arm Mount Stabilized Head, Binocular/Monocular',
         'ARTABLER': ('Arm Mount (Remote mode), Target Sticker, '
                      'Monocular'),
         'ABRTABLE': ('Arm Mount (Remote mode), Target Sticker,'
                      'Binocular/Monocular'),
         'BTOWER': ('Binocular Tower Mount, Stabilized Head,'
                    'Binocular/Monocular'),
         'TOWER': 'Tower Mount, Stabilized Head, Monocular',
         'MPRIM': 'Primate Mount, Stabilized Head, Monocular',
         'BPRIM': ('Primate Mount, Stabilized Head,'
                   'Binocular/Monocular'),
         'MLRR': 'Long-Range Mount, Stabilized Head, Monocular, Camera Level',
         'BLRR': ('Long-Range Mount, Stabilized Head,'
                  'Binocular/Monocular, Camera Angled')}


EYELINK_COLS = {'timestamp': ('start_time',),
                'ocular': {'monocular': ('x', 'y', 'pupil'),
                           'binocular': ('x_left', 'y_left', 'pupil_left',
                                         'x_right', 'y_right', 'pupil_right')},
                'velocity': {'monocular': ('x_vel', 'y_vel'),
                             'binocular': ('x_vel_left', 'y_vel_left',
                                           'x_vel_right', 'y_vel_right')},
                'resolution': ('x_res', 'y_res'),
                'input': ('serial_port_input',),
                'flags': ('flags',),
                'remote': ('head_target_x', 'head_target_y',
                           'head_target_distance', 'remote_flags'),
                'block_num': ('recording_block',),
                'eye_event': ('eye', 'start_time', 'end_time', 'duration'),
                'fixation': ('fix_avg_x', 'fix_avg_y',
                             'fix_avg_pupil_size'),
                'saccade': ('sacc_start_x', 'sacc_start_y',
                            'sacc_end_x', 'sacc_end_y',
                            'sacc_visual_angle', 'peak_velocity')}


def _isfloat(token):
    '''boolean test for whether string can be of type float.

       token (str): single element from tokens list'''

    if isinstance(token, str):
        try:
            float(token)
            return True
        except ValueError:
            return False
    else:
        raise ValueError('input should be a string,'
                         f' but {token} is of type {type(token)}')


def _set_dtypes(tokens):
    """Sets dtypes of tokens in list, which are read in as strings.
       Posix timestamp strings can be integers, eye gaze position and
       pupil size can be floats. flags token ("...") remains as string.
       Missing eye/head-target data (indicated by '.' or 'MISSING_DATA')
       are replaced by np.nan.

       Parameters
       ----------
       tokens (list): list of string elements.

       returns: tokens list with dtypes set."""

    return [int(token) if token.isdigit()  # execute this before _isfloat()
            else float(token) if _isfloat(token)
            else np.nan if token in ('.', 'MISSING_DATA')
            else token
            for token in tokens]


def _parse_line(line):
    """takes a tab deliminited string from eyelink file,
       splits it into a list of tokens, and sets the dtype
       for each token in the list"""

    if len(line):
        tokens = line.split()
        return _set_dtypes(tokens)
    else:
        raise ValueError('line is empty, nothing to parse')


def _is_sys_msg(line):
    """Some lines in eyelink files are system outputs usually
       only meant for Eyelinks DataViewer application to read.
       These shouldn't need to be parsed.

       Parameters
       ----------
       line (string): single line from Eyelink asc file

       Returns: True if any of the following strings that are
       known to indicate a system message are in the line"""

    # TODO more elegent way to identify system messages
    return any(['!V' in line,
                '!MODE' in line,
                ';' in line])


def _set_times(df, first_samp):
    """Converts posix timestamp to time in seconds relative to
       the start of the recording, in place.
       Each sample in an Eyelink file has a posix timestamp string.
       Subtracts the "first" samples timestamp from each timestamp.
       The "first" sample is inferred to be the first sample of
       the first recording block, i.e. the first "START" line.

       df (pandas DataFrame):
           One of the dataframes returned by  self.return_dataframes.

       first_samp (int):
           timestamp of the first sample of the recording. This should
           be the first sample of the first recording block."""

    for col in df.columns:
        if col.endswith('time'):
            df[col] -= first_samp
            df[col] /= 1000
        if col == 'duration':
            df[col] /= 1000


@fill_doc
def read_raw_eyelink(fname, preload=False, verbose=None,
                     annotate_missing=False, interpolate_missing=False,
                     read_eye_events=True):
    """Reader for an Eyelink .asc file.

    Parameters
    ----------
    fname : str
        Path to the eyelink file (.asc).
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawEyetrack
        A Raw object containing eyetracker data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    extension = Path(fname).suffix
    if extension not in '.asc':
        raise ValueError('This reader can only read eyelink .asc files.'
                         f' Got extension {extension} instead. consult eyelink'
                         ' manual for converting eyelink data format (.edf)'
                         ' files to .asc format.')

    return RawEyelink(fname, preload=preload, verbose=verbose,
                      annotate_missing=annotate_missing,
                      interpolate_missing=interpolate_missing,
                      read_eye_events=read_eye_events)


@fill_doc
class RawEyelink(BaseRaw):
    """Raw object from an XXX file.

    Parameters
    ----------
    fname : str
        Path to the data file (.XXX).
    %(preload)s
    %(verbose)s

    Attributes
       ----------
    fname (pathlib.Path):
        Eyelink filename
    header_info (dictionary):
        File/session information derived from lines starting with '**'
        in the beginning of the eyelink .asc file.
    tracking_info (dictionary):
        System/recording information such as mount configuration,
        recording mode (binocular/monocular), sampling frequency etc.
    sample_lines (list):
        List of lists, each list is one sample containing eyetracking
        X/Y and pupil channel data (+ other channels, if they exist)
    event_lines (list):
        List of lists, each list is one event that occured during the
        recording period. Events can vary, from occular events (blinks,
        saccades, fixations), to messages from the stimulus PC, or info
        from a response controller.
    system_lines (list):
        List of lists, each list is a string of a system message line,
        that in most cases aren't needed. system messages occur
        for eyelinks DataViewer application.
    dataframes (dictionary):
        dictionary of pandas DataFrames. One for eyetracking samples,
        and one for each type of eyelink event (blinks, messages, etc)

    Methods
    -------
    return_dataframes:
        Creates pandas DataFrame instances from self.sample_lines and
        self.event_lines. Returns a dictionary of DataFrames.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None,
                 annotate_missing=True, interpolate_missing=True,
                 read_eye_events=True):

        logger.info('Loading {}'.format(fname))

        self.fname = Path(fname)
        self.header_info = None
        self.tracking_info = None
        self.sample_lines = None
        self.event_lines = None
        self.system_lines = None
        self._eye_ch_names = None
        self.dataframes = None

        self._get_header()  # sets header_info
        self._get_tracking_info()  # sets tracking_info
        self._parse_recording_blocks()  # sets sample, event, & system lines

        self.dataframes = self.return_dataframes()  # _eye_ch_names is set here
        info = self._create_info()
        eye_ch_data = self.dataframes['samples'][self._eye_ch_names]
        eye_ch_data = eye_ch_data.to_numpy().T

        # create mne object
        super(RawEyelink, self).__init__(info, preload=eye_ch_data,
                                         filenames=[self.fname],
                                         verbose=verbose)
        # self.set_meas_date(meas_date)
        annots = self._make_eyelink_annots(self.dataframes, read_eye_events)
        self.set_annotations(annots)

        # annotate missing data
        if annotate_missing:  # TODO touch this block.
            raise NotImplementedError()
            annot_bad = annotate_nan(self)
            self.set_annotations(self.annotations.__add__(annot_bad))
        # interpolate missing data
        if interpolate_missing:  # TODO touch this block
            raise NotImplementedError()
            self = interpolate_nan(self)

    def _get_header(self):
        """the first series of lines in eyelink .asc files start with '**'.
           these usually aren't needed for analyses but can provide useful
           information."""

        header_info = {}
        is_header = False
        for line in self.fname.open():
            if line.startswith('**'):
                is_header = True
            else:
                if len(header_info):
                    self.header_info = header_info
                    break
                else:
                    raise ValueError(f'couldnt parse {self.fname} header')
            if is_header:
                tokens = line.lstrip('** ').split(':', maxsplit=1)
                if len(tokens) and tokens[0].isupper():
                    this_key, this_value = tokens[0], tokens[1].strip()
                    header_info[this_key] = this_value

    def _get_tracking_info(self):
        self.tracking_info = self._get_data_spec()
        self.tracking_info['camera'] = self.header_info['CAMERA']

    def _get_data_spec(self):
        """extract info from the RECCFG and ELCLCFG lines
           that occur before the first START block"""
        data_spec = {}
        with self.fname.open() as file:
            is_data_spec = False
            for line in file:
                if line.isspace():
                    continue
                if 'RECCFG' in line:
                    is_data_spec = True
                if is_data_spec:
                    if 'RECCFG' in line:
                        rec_info = line.split('RECCFG')[1].split()
                        data_spec['tracking_mode'] = MODE[rec_info[0]]
                        data_spec['srate'] = int(rec_info[1])
                        data_spec['sample_filter'] = FILTER_TYPE[rec_info[2]]
                        data_spec['analog_filter'] = FILTER_TYPE[rec_info[3]]
                        data_spec['eyes_tracked'] = rec_info[4]
                    elif 'ELCLCFG' in line:
                        mount_info = line.split('ELCLCFG')[1].split()
                        data_spec['mount_config'] = MOUNT[mount_info[0]]
                    else:
                        return data_spec

    def _parse_recording_blocks(self):
        '''Eyelink samples occur within START and END blocks.
           samples lines start with a posix-like string,
           and contain eyetracking sample info. Event Lines
           start with an upper case string and contain info
           about occular events (i.e. blink/saccade), or experiment
           messages sent by the stimulus presentation software.'''

        with self.fname.open() as file:
            block_num = 1
            self.sample_lines = []
            self.event_lines = {'START': [], 'END': [], 'SAMPLES': [],
                                'EVENTS': [], 'ESACC': [], 'EBLINK': [],
                                'EFIX': [], 'MSG': [], 'INPUT': [],
                                'BUTTON': []}
            self.system_lines = []

            is_recording_block = False
            for line in file:
                if line.startswith('START'):  # start of recording block
                    is_recording_block = True
                if is_recording_block:
                    if _is_sys_msg(line):
                        self.system_lines.append(line)
                        continue  # system messages don't need to be parsed.
                    tokens = _parse_line(line)
                    tokens.append(block_num)  # adding recording block number
                    if isinstance(tokens[0], int):  # Samples start with num.
                        self.sample_lines.append(tokens)
                    elif tokens[0] in self.event_lines.keys():
                        event_name, event_info = tokens[0], tokens[1:]
                        self.event_lines[event_name].append(event_info)
                    if tokens[0] == 'END':  # end of recording block
                        is_recording_block = False
                        block_num += 1
            if (not self.sample_lines and not self.event_lines):
                raise ValueError('could not identify any samples or events'
                                 f' in {self.fname}')

    def _infer_col_names(self):
        """Returns the expected column names for the sample lines and event
           lines, to be passed into pd.DataFrame. Also sets the eye channel
           names in self._eye_ch_names. Sample and event lines in eyelink files
           have a fixed order of columns, but the columns that are present can
           vary. The order that col_names is built below should NOT change."""

        col_names = {}
        event_keys = self.event_lines['SAMPLES'][0]

        # initiate the column names for the SAMPLE lines
        col_names['sample'] = list(EYELINK_COLS['timestamp'])

        # and for the eye event lines
        col_names['blink'] = list(EYELINK_COLS['eye_event'])
        col_names['fixation'] = list(EYELINK_COLS['eye_event'] +
                                     EYELINK_COLS['fixation'])
        col_names['saccade'] = list(EYELINK_COLS['eye_event'] +
                                    EYELINK_COLS['saccade'])

        # Recording was either binocular or monocular mode
        eyes_tracked = ('binocular' if
                        self.tracking_info['eyes_tracked'] == 'LR'
                        else 'monocular')
        self._eye_ch_names = list(EYELINK_COLS['ocular'][eyes_tracked])
        col_names['sample'].extend(EYELINK_COLS['ocular'][eyes_tracked])

        # The order of these if statements should not be changed.
        # if velocity data are reported
        if 'VEL' in event_keys:
            self._eye_ch_names.extend(EYELINK_COLS['velocity'][eyes_tracked])
            col_names['sample'].extend(EYELINK_COLS['velocity'][eyes_tracked])
        # if resolution data are reported
        if 'RES' in event_keys:
            self._eye_ch_names.extend(EYELINK_COLS['resolution'])
            col_names['sample'].extend(EYELINK_COLS['resolution'])
            col_names['fixation'].extend(EYELINK_COLS['fixation'] +
                                         EYELINK_COLS['resolution'])
            col_names['saccade'].extend(EYELINK_COLS['saccade'] +
                                        EYELINK_COLS['resolution'])
        # if serial input port values are reported
        if 'INPUT' in event_keys:
            col_names['sample'].extend(EYELINK_COLS['input'])

        # add flags column
        col_names['sample'].extend(EYELINK_COLS['flags'])

        # if head target info was reported, add its cols after flags col.
        if 'HTARGET' in event_keys:
            col_names['sample'].extend(EYELINK_COLS['remote'])

        # finally add a column for recording block number
        # FYI this column does not exist in the asc file..
        # but it is added during _parse_recording_blocks
        for col in col_names.values():
            col.extend(EYELINK_COLS['block_num'])

        return col_names

    def return_dataframes(self):
        """creates a pandas DataFrame for self.sample_lines and for each
           non-empty key in self.event_lines"""

        df_dict = {}
        col_names = self._infer_col_names()

        df_dict['samples'] = pd.DataFrame(self.sample_lines,
                                          columns=col_names['sample'])
        df_dict['samples'] = df_dict['samples'].sort_values('start_time',
                                                            ascending=True)

        # dataframe for each type of eyelink event
        for eye_event, columns, label in zip(['EFIX', 'ESACC', 'EBLINK'],
                                             [col_names['fixation'],
                                              col_names['saccade'],
                                              col_names['blink']],
                                             ['fixations',
                                              'saccades',
                                              'blinks']):
            if self.event_lines[eye_event]:  # an empty list returns False
                df_dict[label] = pd.DataFrame(self.event_lines[eye_event],
                                              columns=columns)
                df_dict[label] = df_dict[label].sort_values('start_time',
                                                            ascending=True)

            else:
                logger.info(f'No {label} were found in this file.'
                            f'Not returning any info on {label}')

        # make dataframe for experiment messages
        if self.event_lines['MSG']:
            experiment_events = []
            for tokens in self.event_lines['MSG']:
                # if offset val exists, it will be the 1st index
                if isinstance(tokens[1], int):
                    text = ' '.join(str(x) for x in tokens[2:])
                    timestamp, offset, message = tokens[0], tokens[1], text
                else:
                    text = ' '.join(str(x) for x in tokens[1:])
                    timestamp, offset, message = tokens[0], np.nan, text
                experiment_events.append([timestamp, offset, message])

            df_dict['experiment_events'] = (pd.DataFrame(experiment_events,
                                                         columns=['start_time',
                                                                  'offset',
                                                                  'event_msg'])
                                            .sort_values('start_time',
                                                         ascending=True)
                                            )

        # TODO: Make dataframes for other eyelink events (Buttons, input..)

        # first sample should be the first line of the first recording block
        first_samp = self.event_lines['START'][0][0]
        for df in df_dict.values():
            _set_times(df, first_samp)

        return df_dict

    def _create_info(self):
        ch_types = ['eyetrack'] * len(self._eye_ch_names)
        info = create_info(self._eye_ch_names,
                           self.tracking_info['srate'],
                           ch_types)

        # set correct channel type and location
        loc = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        for i_ch, ch_name in enumerate(self._eye_ch_names):
            coil_type = (FIFF.FIFFV_COIL_EYETRACK_PUPIL if ('pupil' in ch_name)
                         else FIFF.FIFFV_COIL_EYETRACK_POS)
            unit = (FIFF.FIFF_UNIT_UNITLESS if ('pupil' in ch_name)
                    else FIFF.FIFF_UNIT_PX)

            loc[3] = (-1 if ('left' in ch_name) else
                      1 if ('right' in ch_name) else
                      np.nan)
            loc[4] = (-1 if ('x' in ch_name) else
                      1 if ('y' in ch_name) else
                      np.nan)

            info['chs'][i_ch]['coil_type'] = coil_type
            info['chs'][i_ch]['loc'] = loc.copy()
            info['chs'][i_ch]['unit'] = unit
        return info

    def _make_eyelink_annots(self, df_dict, read_eye_events=True):
        """creates mne Annotations for each df in self.dataframes"""

        annots = None

        for key, df in df_dict.items():
            if key in ['blinks', 'fixations', 'saccades'] and read_eye_events:
                onsets = df['start_time']
                durations = df['duration']
                descriptions = [key[:-1]] * len(onsets)
                this_annot = Annotations(onset=onsets,
                                         duration=durations,
                                         description=descriptions,
                                         orig_time=None)
            elif key in ['experiment_events']:
                onsets = df['start_time']
                durations = [0] * onsets
                descriptions = df['event_msg']
                this_annot = Annotations(onset=onsets,
                                         duration=durations,
                                         description=descriptions)
            else:
                continue  # TODO make annotations for other eyelink dfs
            if not annots:
                annots = this_annot
            elif annots:
                annots += this_annot
        return annots
