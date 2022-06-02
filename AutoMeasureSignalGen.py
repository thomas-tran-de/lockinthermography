from sourceCode.AutoMeasure import AutoMeasure
from sourceCode.Config import getCustomLogger
from pathlib import Path
import numpy as np
import sys
sys.path.append('../hardwareinterfaces')
from Rigol1000 import Rigol1022a
import time

"""
This script runs a lock-in measurement series at different frequencies or
powers with the new IR camera and creates data files from the .irb files.
"""

# DASHBOARD
name = 'PEEK07_PowerSweep'
folder = Path('E:/Thomas/2022-05-05_PEEK07')
camera = 'dell'
Sentinel = getCustomLogger(name, folder)
signalGen = Rigol1022a('USB0::0x0400::0x09C4::DG1D194304542::INSTR')
cameraFrequency = 500
dutyCycle = 0.3

# For frequency sweep
doFreqSweep = False
if doFreqSweep:
    dropPeriods = None
    startFreq = 0.1
    stopFreq = 125
    stepsFreq = 50
    power = 2.5 / 2.5  # Relative power between 0 and 1 (= 0 and 0.9 W)

# For a power sweep
doPowerSweep = True
if doPowerSweep:
    lockInFrequency = 1
    dropPeriods = 60 * lockInFrequency
    measPeriods = 60 * lockInFrequency
    startPower = 0.5 / 2.5  # Relative power between 0 and 1 (= 0 and 0.9 W)
    stopPower = 0.025 / 2.5
    stepsPower = 39

# For a combined sweep
doCombinedSweep = False
if doCombinedSweep:
    startFreq = 0.1
    stopFreq = 30
    stepsFreq = 30
    startPower = 0.2
    stopPower = 1
    stepsPower = 5

# MAIN
automeasure = AutoMeasure(folder / name, camera)
camPeriod = 1 / cameraFrequency

# Do a frequency sweep
if doFreqSweep:
    amplitude = round(power * 2.5, 3)
    offset = amplitude / 2
    for f in np.logspace(np.log10(startFreq), np.log10(stopFreq), stepsFreq):
        # Calculate needed parameters
        lockInFrequency = 1 / (round(cameraFrequency / f) * camPeriod)
        tempDropPer = max(30 * lockInFrequency, 50)
        measPeriods = 60 * lockInFrequency

        # Set up instruments
        automeasure.configsave(measPeriods, lockInFrequency,
                               tempDropPer, dutyCycle * 100)
        signalGen.applySquare(lockInFrequency, amplitude, offset, dutyCycle)
        signalGen.setOutput(True)

        # Start measuerment
        automeasure.startMeasurement()
        automeasure.fileCheck()

# Do a power sweep
if doPowerSweep:
    automeasure.measureCount += 53
    input(2.5 * np.linspace(startPower, stopPower, stepsPower))
    lockInFrequency = 1 / \
        (round(cameraFrequency / lockInFrequency) * camPeriod)
    automeasure.configsave(measPeriods, lockInFrequency,
                           dropPeriods, dutyCycle * 100)
    for p in np.linspace(startPower, stopPower, stepsPower):
        # Calculate needed parameters
        amplitude = round(p * 2.5, 3)
        offset = amplitude / 2

        # Set up instruments
        signalGen.applySquare(lockInFrequency, amplitude, offset, dutyCycle)
        signalGen.setOutput(True)

        # Start measuerment
        automeasure.startMeasurement()
        automeasure.fileCheck()

# Do a combined sweep
if doCombinedSweep:
    for p in np.linspace(startPower, stopPower, stepsPower):
        amplitude = round(p * 2.5, 3)
        offset = amplitude / 2
        for f in np.logspace(np.log10(startFreq), np.log10(stopFreq), stepsFreq):
            # Calculate needed parameters
            lockInFrequency = 1 / (round(cameraFrequency / f) * camPeriod)
            dropPer = max(30 * f, 50)
            measPeriods = 60 * f

            # Set up instruments
            automeasure.configsave(
                measPeriods, lockInFrequency, dropPer, dutyCycle * 100)
            signalGen.applySquare(
                lockInFrequency, amplitude, offset, dutyCycle)
            signalGen.setOutput(True)

            # Start measuerment
            automeasure.startMeasurement()
            automeasure.fileCheck()

# Set signal generator to 0.5 V DC and put it into local mode
signalGen.applyDC(0.5)
signalGen.disconnect()
