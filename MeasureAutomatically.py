from sourceCode.AutoMeasure import AutoMeasure
from sourceCode.Config import getCustomLogger
from pathlib import Path

"""
This script runs a lock-in measurement series at different temperatures with
the old IR camera and creates data files from the .irb files.
"""

# DASHBOARD
name = 'AutoMeasureTest'
folder = Path('E:/Thomas/SoftwareTest')
measPeriods = 10
dropPeriods = 5
numOfMeasurements = 2
lockInFrequency = 1
camera = 'new'
Sentinel = getCustomLogger(name, folder)

# MAIN
automeasure = AutoMeasure(folder / name, camera)

# run the measurement
automeasure.configsave(measPeriods, lockInFrequency, dropPeriods)
for i in range(numOfMeasurements):
    automeasure.startMeasurement()
    automeasure.fileCheck()

# create the data files
automeasure.createDataFromIrbFiles(automeasure.measureCount - 1)
