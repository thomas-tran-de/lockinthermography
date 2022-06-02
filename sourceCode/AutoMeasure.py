import pyautogui as augui
import time
import os
from logging import getLogger
from sourceCode.StepperMotor import rotationMotor
import sys
sys.path.append('../hardwareinterfaces')
from AutoHeatStageProgram import AutoHeatStageProgram


class AutoMeasure():
    """
    AutoMeasure to autorun IRBIS active online measurements with the old
    IR camera. Please avoid working on the computer while one of the IRBIS
    programs is in foreground.
    """
    # Click positions for the old and new camera
    positions = {'old': [(150, 115),    # RawData name in SaveConfig
                         (505, 235),    # SaveRawDataOff
                         (505, 515),    # AutoSaveOff
                         (505, 735),    # SaveResultsOn
                         (505, 825),    # Close SaveConfig window
                         (282, 209),    # Top of the scrolling bar
                         (239, 866),    # Measure Time Dropdown
                         (660, 88),     # Open save dropdown
                         (660, 150),    # open ConfigSave
                         (239, 866),    # MeasTime unit (same as Measure Time Dropdown?)
                         (1700, 90),    # Open DataReader
                         (528, 55),     # live checkmark in reader
                         (120, 90),     # Open file in reader
                         (990, 90),     # ExportASCII
                         ],
                 'new': [(145, 90), (400, 180), (400, 410),
                         (400, 600), (470, 685), (214, 189),
                         (200, 722), (535, 75), (530, 125),
                         (204, 727), (1128, 75), (443, 46),
                         (108, 96), (831, 66)
                         ],
                 'dell': [(224, 75), (345, 150), (345, 330),
                          (345, 490), (345, 550), (),
                          (180, 580), (423, 56), (428, 100),
                          (180, 580), (900, 55), (331, 36),
                          (78, 55), (660, 56)]}

    def __init__(self, fileName, camera):
        """
        Constructor for AutoMeasure

        Parameters
        ----------
        fileName : path
            Data files will be saved at the paths
            fileName_Results_measureCount.irb,
            fileName_Results_measureCount_Period0001_Amp.txt,
            fileName_Results_measureCount_Period0001_Phase.txt and
            fileName_Temperature_measureCount.xlsx.
            measureCount is put into the name as four digit number.
            If the heat stage is used, the temperature will be added to
            the fileName.
        camera : str
            Either 'old' for the VarioCam HD or 'new' for the ImageIR camera
        """
        # Default values for variables
        self.measureCount = 1
        self.measTime = None

        # Remember save name and camera
        self.fileName = fileName
        if camera not in ['old', 'new', 'dell']:
            raise ValueError('The camera must be either \'old\' or \'new\'.')
        self.camera = camera

        # Initial setup
        if self.camera == 'old':
            self.imgPath = 'sourceCode\\ReferencePicturesAutoGUI\\'
        elif self.camera == 'new':
            self.imgPath = 'sourceCode\\ReferencePicturesAutoGUI\\newCamera\\'
        elif self.camera == 'dell':
            self.imgPath = 'sourceCode\\ReferencePicturesAutoGUI\\dellLaptop\\'
        else:
            raise NotImplementedError
        self.sublime = augui.getWindowsWithTitle('*REPL* [python] - Sublime '
                                                 'Text (UNREGISTERED)')[0]
        self.Sentinel = getLogger('Watchtower')
        try:
            augui.getWindowsWithTitle('IRBIS active online')[0] == []
        except IndexError:
            exit('Cannot find IRBIS active online program window. '
                 'Please make sure the program is running and restart '
                 'the measurement.')

    def searchImg(self, imgFile):
        """
        Locates the center of a reference picture on the screen.

        Parameters
        ----------
        imgFile : string
            Name of the reference picture in folder ReferencePicturesAutoGUI.
        """
        # pyautogui and opencv must be installed for this to work
        return augui.locateCenterOnScreen(self.imgPath + imgFile,
                                          confidence=0.9)

    def IRBISToForeground(self):
        """
        Moves IRBIS to foreground and prepares it for the use of AutoGUI tools.
        This also closes the config save window if open.
        """
        # Do nothing if IRBIS is already active
        if 'IRBIS active online' in augui.getActiveWindow().title:
            self.Sentinel.debug('IRBIS is already active')
            return
        else:
            self.Sentinel.debug(augui.getActiveWindow().title + ' is active')

        # if Config Save is open, move it to a good position and close it.
        if augui.getWindowsWithTitle('Config Save') != []:
            augui.getWindowsWithTitle('Config Save')[0].activate()
            time.sleep(1)
            augui.moveTo(augui.getActiveWindow().topleft)
            augui.mouseDown()
            augui.moveTo(5, 5)
            augui.mouseUp()
            augui.click(*self.positions[self.camera][4])

        # minimize all windows except those without names (to prevent weird
        # things from happening) and maximize IRBIS active online.
        for i in augui.getAllTitles():
            if i != '' and i != 'TeamViewer Panel' and i != 'TeamViewer':
                augui.getWindowsWithTitle(i)[0].minimize()
        # bring IRBIS to foreground
        irbis = augui.getWindowsWithTitle('IRBIS active online')[0]
        irbis.maximize()
        time.sleep(0.2)
        if irbis.isActive is False:
            self.Sentinel.error('IRBIS still inactive. Please do not '
                                'open non-minimizable windows while '
                                'AutoMeasure is running.')
        # Connect to the camera?
        # augui.click(*self.positions[self.camera][5])

    def configsave(self, measTimeInput, lockInFrequency=1, skipPeriods=0,
                   dutyCycle=0.5):
        """
        Chooses suitable configs in the Config Save window and estimates the
        duration of one measurement using the information in the MeasTime
        section in Irbis Active Online.

        Parameters
        ----------
        measTimeInput : float
            Duration for writing into the measTime in Irbis Active Online in
            Seconds, Minutes, Hours or Periods.
        lockInFrequency : float, optional
            Frequency in Hz for the measurement to be written into the
            LockInFrequency section in Irbis Active Online. Default is 1.
        skipPeriods : int, optional
            Number of periods to skip before the measurement starts.
            Default is 0
        dutyCycle : float, optional
            Duty cycle for the measurement. 0 <= dutyCycle <= 1.
            Default is 0.5
        """
        # open save configs
        self.IRBISToForeground()
        augui.click(self.positions[self.camera][7])
        augui.press('c')

        # move Config Save window
        augui.moveTo(augui.getActiveWindow().topleft)
        augui.mouseDown()
        augui.moveTo(5, 5)
        augui.mouseUp()

        # change RawData file name
        augui.click(*self.positions[self.camera][0])
        augui.press('tab')
        augui.hotkey('shift', 'tab')
        nameToWrite = str(self.fileName)
        augui.typewrite(nameToWrite + '_raw.*')

        # change PeriodData file name
        for i in range(7):
            augui.press('tab')
        augui.typewrite(nameToWrite + '.*')

        # check save configs
        for i in ['SaveRawDataOff.PNG', 'AutoSaveOff.PNG',
                  'SaveResultsOn.PNG']:
            if self.searchImg(i) is None:
                if i == 'SaveRawDataOff.PNG':
                    augui.click(self.positions[self.camera][1])
                    augui.moveTo(5, 5)
                elif i == 'AutoSaveOff.PNG':
                    augui.click(self.positions[self.camera][2])
                    augui.moveTo(5, 5)
                elif i == 'SaveResultsOn.PNG':
                    augui.click(self.positions[self.camera][3])
            if self.searchImg(i) is None:
                self.Sentinel.error('Picture identification for {} '
                                    'failed! The measurement configs '
                                    'might be wrong!'.format(i))

        # close save configs
        augui.click(*self.positions[self.camera][4])
        time.sleep(1)
        if augui.getActiveWindowTitle() == 'Config Save':
            exit('Invalid data file path, please change it according to the '
                 'fileName parameter description in AutoMeasure\'s __init__')

        # estimate measurement time
        augui.doubleClick(self.positions[self.camera][9])
        augui.press('tab')
        augui.typewrite(str(measTimeInput))
        for i in ['Period.PNG', 'Second.PNG', 'Minute.PNG', 'Hour.PNG']:
            if self.searchImg(i):
                if i == 'Period.PNG':
                    for i in range(5):
                        augui.hotkey('shift', 'tab')
                    augui.typewrite(str(lockInFrequency).replace('.', ','))
                    augui.hotkey('enter')
                    self.measTime = measTimeInput / lockInFrequency
                    time.sleep(2)
                elif i == 'Second.PNG':
                    self.measTime = measTimeInput
                elif i == 'Minute.PNG':
                    self.measTime = 60 * measTimeInput
                elif i == 'Hour.PNG':
                    self.measTime = 3600 * measTimeInput
                else:
                    self.Sentinel.warning('Couldn\'t determine time unit')

        # edit drop periods
        augui.hotkey('shift', 'tab') #@Thomas: Musste das hier ändern, weil der bei mir die Drop Periods nicht mehr eingetragen hat.
        # for i in range(3):
        #     augui.hotkey('shift', 'tab')
        #     time.sleep(0.05)
        augui.typewrite(str(skipPeriods))
        time.sleep(0.05)
        self.measTime += (skipPeriods / lockInFrequency)
        # edit duty cycle
        if dutyCycle != 0.5:
            for i in range(6):
                augui.hotkey('tab')
            augui.typewrite(str(dutyCycle))
        # self.sublime.maximize()
        self.Sentinel.info('AutoGUI operations are done until Sublime is '
                           'minimized again.')
        self.Sentinel.info('measure time: ' + str(self.measTime))

    def startMeasurement(self):
        """
        Starts the IRBIS measurement if possible.
        """
        self.IRBISToForeground()
        posMeasure = self.searchImg('MeasureAktiv.PNG')
        if posMeasure is not None:
            augui.click(posMeasure)
            self.Sentinel.info('Measurement {} started.'
                               ''.format(self.measureCount))
            self.measureCount += 1
        else:
            self.Sentinel.error('Measurement starting failed, retrying to '
                                'start the measurement.')
            time.sleep(10)
            self.IRBISToForeground()
            posMeasure = self.searchImg('MeasureAktiv.PNG')
            if posMeasure is not None:
                augui.click(posMeasure)
                self.Sentinel.info('Measurement {} started.'
                                   ''.format(self.measureCount))
                self.measureCount += 1
            else:
                exit('Failed to start the measurement! Make sure the '
                     '\'Measure\' button is active.')
        # self.sublime.maximize()
        self.Sentinel.info('AutoGUI operations are done until Sublime is '
                           'minimized again.')

    def fileCheck(self):
        """
        Deletes incorrectly saved files and restarts the measurement.
        After that the program sleeps until the measurement is done.
        """
        # delete wrongly saved files and restart the measurement
        if (self.measureCount - 1) == 1:
            fileToCheck = str(self.fileName) + '_Results.irb'
        else:
            fileToCheck = '{}_Results_{:04d}.irb'.format(
                self.fileName, self.measureCount - 1)
        time.sleep(5)
        while os.path.isfile(fileToCheck):
            os.remove(fileToCheck)
            self.Sentinel.info('Measurement {} was wrongly saved and '
                               'therefore deleted. Measurement will be '
                               'restarted.'.format(self.measureCount - 1))
            self.measureCount -= 1
            time.sleep(2)
            self.startMeasurement()
            time.sleep(5)

        # sleep measurement time
        time.sleep(self.measTime + 20)

    def createDataFromIrbFiles(self, measurements):
        """
        Uses the .irb file from the measurement to create the Phase and Amp
        data files explained in AutoMeasure's __init__. It uses the program
        'IRBIS active online data reader' for that.

        Parameters
        ----------
        measurements : int
            Number of irbis files with consecutive numbering from which data
            files should be exported.
        """
        # ensure the data reader is open and bring it to foreground
        if augui.getWindowsWithTitle('Irbisactiveonlinedatareader') != []:
            for i in augui.getAllTitles():
                if i != '':
                    augui.getWindowsWithTitle(i)[0].minimize()
            reader = augui.getWindowsWithTitle(
                'Irbisactiveonlinedatareader')[0]
            reader.maximize()
            if reader.isActive is False:
                self.Sentinel.error('IRBIS still inactive. Please do not '
                                    'open non-minimizable windows while '
                                    'AutoMeasure is running.')
        else:
            self.IRBISToForeground()
            augui.click(self.positions[self.camera][10])
            self.sublime.maximize()
            self.Sentinel.info('Starting IRBIS data reader.')
            time.sleep(60)
            for i in augui.getAllTitles():
                if i != '':
                    augui.getWindowsWithTitle(i)[0].minimize()
            reader = augui.getWindowsWithTitle(
                'Irbisactiveonlinedatareader')[0]
            reader.maximize()
            if reader.isActive is False:
                self.Sentinel.error('IRBIS still inactive. Please do not '
                                    'open non-minimizable windows while '
                                    'AutoMeasure is running.')

        # choose the right configs
        augui.click(self.positions[self.camera][11])
        for i in ['live.png', 'amp.png', 'phase.png', 'comp.png', 'data.png',
                  'M90Deg.png', '0Degree.png']:
            augui.press('tab')
            if self.searchImg(i) is None:
                augui.hotkey('shift', 'tab')
                augui.press(' ')
                augui.press('tab')
            if self.searchImg(i) is None:
                self.Sentinel.error('Picture identification failed! The '
                                    'IRBIS reader configs might be wrong!')
            if i == 'comp.png':
                augui.press('tab')

        # create data files
        if measurements == 1:
            files = [str(self.fileName) + '_Results.irb']
        else:
            files = ['{}_Results_{:04d}.irb'.format(self.fileName, i + 1)
                     for i in range(measurements)]
        for i in range(len(files)):
            file = files[i]
            # open the .irb file
            augui.click(self.positions[self.camera][12])
            time.sleep(0.2)
            augui.typewrite(file)
            augui.press('enter')
            time.sleep(1)
            # export Phase and Amp data files
            augui.click(self.positions[self.camera][13])
            time.sleep(0.2)
            augui.typewrite(file[:-3] + 'txt')
            augui.press('enter')
            time.sleep(1)
            if os.path.isfile(file[:-4] + '_Period0001_Amp.txt') is None:
                self.Sentinel.error('Exporting data files for measurement {}'
                                    ' failed!'.format(i))
            else:
                self.Sentinel.info('Measurement {} successfully '
                                   'exported.'.format(i))
        self.sublime.maximize()
        self.Sentinel.info('AutoGUI operations are done until Sublime is '
                           'minimized again.')
        self.Sentinel.info('Data file creation finished!')

    def rotateMeasure(self, startAngle, angleStep, numberOfSteps, motorPort):
        """
        Starts a series of IRBIS measurements at different laser orientations.
        The laser is rotated with the motor by the selected angle.

        Parameters
        ----------
        startAngle : int
            Angle from starting orientation of the laser to zero position.
        angleStep : int
            Laser rotation angle between two measurements.
        numberOfSteps : int
            Specifies how often the laser is rotated. This equals the number
            of IRBIS measurements in the series minus the first measurement.
        motorPort : string
            COM port of the Arduino motor (check device manager in windows).
        """
        # do the first measurement
        motor = rotationMotor(motorPort)
        motor.rotate(startAngle)
        self.startMeasurement()
        self.fileCheck()

        # do the subsequent measurements
        for i in range(numberOfSteps):
            motor.rotate(angleStep)
            self.startMeasurement()
            self.fileCheck()

        self.Sentinel.info('Measurement series finished!')
        motor.reset()

    def heatingMeasure(self, startTemperature, temperatureStep, numberOfSteps, measurementsPerTemp=1, equilTime=600):
        """
        Starts a series of IRBIS measurements at different temperatures using
        the heat stage. The heat stage must be connected via USB and the heat
        stage control program WinTemp must be running.

        Time dependent temperature during the equilibration will be saved
        at the path fileName_measureCount_Temp_target_EquilData.xlsx.
        Time dependent temperature during the measurement(s) will be saved
        at the path fileName_measureCount_Temp_target_MeasData.xlsx.
        In the position of 'measureCount' and 'target' will be the number of
        the measurment and the target temperature as 4 digit number.

        Parameters
        ----------
        startTemperature : float
            Start temperature in degree Celsius.
        temperatureStep : float
            Temperature step between two measurements in degree Celsius.
        numberOfSteps : int
            Specifies how many heating steps are made. This equals the number
            of IRBIS measurements in the series minus the first measurement.
        measurementsPerTemp : int, optional
            Number of measurements per temperature level. Default is 1.
        equilTime : int, optional
            Waiting time in seconds for reaching the target temperature.
            Default is 600 s.
        """
        heater = AutoHeatStageProgram()
        target = startTemperature
        self.fileName += '_XXXX_Temp_XXXX'
        for i in range(numberOfSteps + 1):

            # write temperature level and measureCount into fileName
            self.fileName = self.fileName[:-14] + '{:04d}'.format(
                self.measureCount) + '_Temp_{:04d}'.format(target)

            # start heating, wait for equilibration and record the temperature
            heater.EquilToTemp(target, equilTime, self.fileName)
            self.Sentinel.info('Temperature equilibration data saved.')

            # start recording the temperature during the measurement(s)
            fileTime = heater.recordData(self.fileName + '_MeasData')

            # do the measurement(s)
            for i in range(measurementsPerTemp):
                self.startMeasurement()
                self.fileCheck()
            target += temperatureStep

            # save the measurement temperature data
            heater.saveData()
            heater.xls_to_xlsx(self.fileName + '_MeasData', fileTime)
            self.Sentinel.info('Measurement temperature data saved.')

        self.Sentinel.info('Measurement series finished!')
        heater.stop()
