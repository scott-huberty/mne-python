# ParseEyeLinkAsc.py
# - Reads in .asc data files from EyeLink and produces pandas dataframes for
# further analysis
#
# Created 7/31/18-8/15/18 by DJ.
# Updated 7/4/19 by DJ - detects and handles monocular sample data.

from pathlib import Path
import numpy as np
import pandas as pd
from ...utils import logger

def _is_system_message(line):
    return any(['!V' in line,
               '!MODE' in line,
              ';' in line])


def ParseEyeLinkAsc_(elFilename):
    # dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)
    # -Reads in data files from EyeLink .asc file and produces readable
    # dataframes for further analysis.
    #
    # INPUTS:
    # -elFilename is a string indicating an EyeLink data file from an AX-CPT
    # task in the current path.
    #
    # OUTPUTS:
    # -dfRec contains information about recording periods (often trials)
    # -dfMsg contains information about messages (usually sent from stimulus
    # software)
    # -dfFix contains information about fixations
    # -dfSacc contains information about saccades
    # -dfBlink contains information about blinks
    # -dfSamples contains information about individual samples
    #
    # Created 7/31/18-8/15/18 by DJ.
    # Updated 11/12/18 by DJ - switched from "trials" to "recording periods"
    # for experiments with continuous recording
    # Updated 9/??/19 by Dominik Welke - fixed read-in of data

    # ===== READ IN FILES ===== #
    # Read in EyeLink file
    logger.info('Reading in EyeLink file %s...' % elFilename)

    with Path(elFilename).open() as file:
        samples = [] 
        events = {'trial_ids':[]}

    is_recording_block = False

    for line in file:
        if line.isspace() or  _is_system_message(line):
            continue
        line = line.split()
        if 'TRIALID' in line:
            events['trial_ids'].append([line[1], ' '.join(line[2:])])
        if line[0] == 'START':
            is_recording_block = True
        if is_recording_block:
            if line[0].isdigit():  # Sample lines start with a number.
                samples.append(line)
            elif line[0].isupper():  # Event strings are all-caps
                if line[0] not in events.keys():
                    events[f'{line[0]}'] = [line[1:]]
                else: # append line to existing key
                    events[f'{line[0]}'].append(line[1:])                
        if line[0] == 'END':
            is_recording_block = False
    return samples, events