from pathlib import Path
from sourceCode.LIT_FiberAnalyzer import LIT_FiberAnalyzer
from sourceCode.Config import getCustomLogger

# DASHBOARD
ExpName = "Test"
ExpFolder = Path("testing/Fiber Testing")
LineWidth = 0        # in pixels
MinDist = 0.2        # start point of evaluation in mm from center
MaxDist = 1.0        # stop point of evaluation in mm from center
AutoRun = False
Sentinel = getCustomLogger(ExpName, ExpFolder, 'DEBUG')

"""********************************************************"""
# MAIN
LIT_Instrument = LIT_FiberAnalyzer(ExpName, ExpFolder, AutoRun)

# analyze the calibration grid
LIT_Instrument.setRScale(1.3e-3)

# analyze the raw data for the isotropic measurement
LIT_Instrument.AnalyzeFiberLITMeas(ExpFolder / 'ASCII',
                                   MinDist, MaxDist, LineWidth,
                                   SaveRawData=False, SaveNumpyData=False)
