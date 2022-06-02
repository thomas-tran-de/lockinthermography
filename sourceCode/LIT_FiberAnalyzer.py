import numpy as np
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sourceCode.LIT_SweepData import LIT_SweepData
from sourceCode.LIT_Analyzer import LIT_Analyzer


class LIT_FiberAnalyzer(LIT_Analyzer):
    """ Class to analyze LIT measurements of a single fiber """

    def SelectFiberRegion(self, Meas):
        """
        Shows the raw amplitude image.
        The user can select the approximate position of the fiber with
        a line. This will be used to determine the evaluation area later on.

        Parameters
        ----------
        Meas : LIT_SingleFiberMeas object
        """
        def onMouseClick(event):
            if event.xdata is None or event.ydata is None:
                return False
            else:
                xStart, yStart = event.xdata, event.ydata
            self.Sentinel.info('You selected ({}, {}) as your start point.'
                               ''.format(xStart, yStart))
            # Change the start point
            Meas.FiberLine[0][0] = xStart
            Meas.FiberLine[1][0] = yStart
            selection.set_data(Meas.FiberLine[0],
                               Meas.FiberLine[1])
            # Redraw canvas
            plt.draw()

        def onMouseRelease(event):
            if event.xdata is None or event.ydata is None:
                return False
            else:
                xStop, yStop = event.xdata, event.ydata
            self.Sentinel.info('You selected ({}, {}) as your end point.'
                               ''.format(xStop, yStop))
            # Change the data of the end point
            Meas.FiberLine[0][1] = xStop
            Meas.FiberLine[1][1] = yStop
            selection.set_data(Meas.FiberLine[0],
                               Meas.FiberLine[1])
            # Redraw canvas
            plt.draw()

        # Create the image
        fig = Meas.getRawAmplCanvas()

        # Connect the events with the canvas
        fig.canvas.mpl_connect('button_press_event', onMouseClick)
        fig.canvas.mpl_connect('button_release_event', onMouseRelease)

        # Create an initial line
        ax1 = fig.gca()
        XCenter = np.mean(ax1.get_xlim())
        YCenter = np.mean(ax1.get_ylim())
        Meas.FiberLine = [2 * [XCenter],
                          [YCenter + YCenter / 2, YCenter - YCenter / 2]]
        selection, = ax1.plot(Meas.FiberLine[0],
                              Meas.FiberLine[1], 'ro-')

        self.Sentinel.info(
            'Drag a line to select the fiber you want to analyze')
        plt.show()

    def FocusOnFiber(self, Meas, Width, Ref=None, Threshold=0.7):
        """
        Rotates and zooms the data to the region of interest (ROI).
        The ROI is selected with self.SelectFiberRegion.

        Parameters
        ----------
        Meas : LIT_SingleFiberMeas object
            The measurement to be handled. The raw data of this measurement
            will be rotated and then we will zoom in on the fiber
        Width : int
            The width to zoom in on either side of the fiber. The final
            image will have a width of (width + 1 + width)
        Ref: LIT_SingleFiberMeas object
            A reference measurement. The rotation angle of Meas will be set
            to Ref.FibRotAngle. This will be used if self.AutoRun is True
            and one measurement has been examined.
        Threshold : float, optional
            Used to determine which pixels to use for the tilt detection.
            For each row, the pixels with the maxIntensity * Threshold will
            be used. 0 < Threshold <= 1
        """
        # Add some buffer to Width
        origWidth = Width
        Width = max(Width * 6, Width + 20)

        if Ref is None:
            # Get angle from the user defined line in the amplitude image
            dx, dy = np.diff(Meas.FiberLine)
            angle = np.rad2deg(float(np.arctan(dy / dx))) + 90
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            self.Sentinel.debug('User chose angle {:.2f}°'.format(angle))

            # Rotate the amplitude image
            newAmpl = ndimage.rotate(Meas.RawAmplData[2], angle,
                                     cval=np.nan, order=1)

            # Store the image centers of the rotated and original image
            orgCenter = (np.array(Meas.RawAmplData[2].shape[::-1]) - 1) / 2
            newCenter = (np.array(newAmpl.shape[::-1]) - 1) / 2

            # Get starting point of the FiberLine to create a preliminary img
            point = np.array([Meas.FiberLine[0][0] / Meas.RScale,
                              Meas.FiberLine[1][0] / Meas.RScale])
            xCenter = self.RotatePoint(point, orgCenter, newCenter, angle)[0]

            # Cut image in x direction (might be asymmetrical)
            newAmpl = newAmpl[:, max(0, xCenter - Width):xCenter + Width + 1]

            # Cut image in y direction = delete all rows containing only NaNs
            newAmpl = newAmpl[~np.isnan(newAmpl).all(axis=1)]

            # Find the maximum and average of each row
            maxValues = np.nanmax(newAmpl, axis=1)
            avgValues = np.nanmean(newAmpl, axis=1)

            # For each row, get the average of the indices of the values that
            # - are larger than our threshold and
            # - are larger than 2 times the average (prevent using noise)
            x = [np.argwhere((newAmpl[i] >= Threshold * maxValues[i]) &
                             (newAmpl[i] > 1.5 * avgValues[i])).mean()
                 for i in range(newAmpl.shape[0])]
            x = np.array(x)
            y = np.arange(newAmpl.shape[0])
            y = y[~np.isnan(x)]
            x = x[~np.isnan(x)]

            # Create linear fit with x and y swapped to prevent infinite slope
            fitValues = stats.linregress(y, x)

            # Calculate a more accurate rotation angle from the linear fit
            angle = angle - np.degrees(np.arctan(fitValues[0]))
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            Meas.FibRotAngle = angle
            self.Sentinel.debug(
                'The rotation angle for {} was determined '
                'to be {:.2f}°'.format(Meas.AmplPath.name, angle))
        else:
            Meas.FibRotAngle = Ref.FibRotAngle

        # Create the final amplitude image
        newAmpl = ndimage.rotate(Meas.RawAmplData[2], Meas.FibRotAngle,
                                 cval=np.nan, order=1)
        orgCenter = (np.array(Meas.RawAmplData[2].shape[::-1]) - 1) / 2
        newCenter = (np.array(newAmpl.shape[::-1]) - 1) / 2
        excitation = self.FindCenterFromAmp(newAmpl)
        xCenter = excitation[0]

        # Limit Width to the width of the image
        Width = min(xCenter, Width)
        Width = min(newAmpl.shape[-1] - xCenter, Width)

        # Save the excitation center
        Meas.Center = np.array([Width, excitation[1]])

        # Cut the image in x direction symmetrically
        newAmpl = newAmpl[:, xCenter - Width:xCenter + Width + 1]
        Meas.AmplData = newAmpl

        # Create the final phase image
        newPhase = ndimage.rotate(Meas.RawPhaseData[2], Meas.FibRotAngle,
                                  cval=np.nan, order=1)
        newPhase = newPhase[:, xCenter - Width:xCenter + Width + 1]
        Meas.PhaseData = newPhase

        # Save kinked rectangle for the zoomed 2D plot
        points = np.array(
            [(xCenter - origWidth, 0), (xCenter + origWidth + 1, 0),
             (xCenter + origWidth + 1, Meas.AmplData.shape[0]),
             (xCenter - origWidth, Meas.AmplData.shape[0])])
        for i in range(len(points)):
            points[i] = self.RotatePoint(points[i], newCenter,
                                         orgCenter, -Meas.FibRotAngle)
        # Define a function to use the patch multiple times
        Meas.ROIBorder = lambda: Polygon(points * Meas.RScale,
                                         ec='red', fill=False)

    def RotatePoint(self, Point, OrgCenter, NewCenter, Angle):
        """
        Rotates the Point around the OrgCenter

        Rotation is done around the origin. Therefore, we need to translate
        the Point, rotate it, and translate it to the new image

        Parameters
        ----------
        Point : (int, int)
            Coordinates of the point to rotate
        OrgCenter : (int, int)
            Coordinates of the center of the original image
        NewCenter : (int, int)
            Coordinates of the center of the rotated image
        Angle : float
            Angle of rotation

        Returns
        -------
        (int, int)
            Coordinates of the rotated point
        """
        Point = Point - OrgCenter
        a = np.deg2rad(Angle)
        newPoint = (Point[0] * np.cos(a) + Point[1] * np.sin(a),
                    -Point[0] * np.sin(a) + Point[1] * np.cos(a))
        return np.int_(newPoint + NewCenter)

    def SelectBoundaries(self, Meas):
        """
        Shows the zoomed amplitude image with the center.
        The user can then select an appropriate maximal distance
        along the fiber.

        Parameters
        ----------
        Meas : LIT_SingleFiberMeas object
            contains all data from a single measurement
        """
        def onMouseClick(event):
            if event.xdata is not None:
                # Compute the distance from the mouse position to the center
                Meas.MaxDist = abs(Meas.Center[1] * self.RScale - event.ydata)
                self.Sentinel.info('New Max Dist selected is {:.3f} mm'
                                   ''.format(Meas.MaxDist))
            else:
                return True

            # display the recieved data to the user
            self.Sentinel.debug('User pressed at {} in data coords'
                                ''.format([event.xdata, event.ydata]))
            # change the position of the two horizontal boundary lines
            TopLine.set_ydata(
                [Meas.Center[1] * self.RScale + Meas.MaxDist] * 2)
            BottomLine.set_ydata(
                [Meas.Center[1] * self.RScale - Meas.MaxDist] * 2)
            plt.draw()  # redraw everything
            return True

        fig = Meas.getAmplCanvas()

        # connect the MouseClickEvent with the Canvas
        fig.canvas.mpl_connect('button_press_event', onMouseClick)
        ax = fig.gca()
        ax.plot([Meas.Center[0] * self.RScale],
                [Meas.Center[1] * self.RScale], 'ro')
        ax.axhline(Meas.Center[1] * self.RScale, c='w', ls='-')

        # plot the boundary lines
        TopLine = ax.axhline((Meas.Center[1] * self.RScale + Meas.MaxDist),
                             c='w', ls='--')
        BottomLine = ax.axhline((Meas.Center[1] * self.RScale - Meas.MaxDist),
                                c='w', ls='--')
        self.Sentinel.info(
            'Click on the image to change the vertical distance')
        plt.show()

    def FindCenterFromAmp(self, AmplData, nPoints=100):
        """
        Determines the center of the fiber excitation.
        The center is the average of the first nPoints pixels with the
        largest amplitude. This will only work if there is only one fiber
        measured. If the image consists of multiple fibers with rather large
        distances, the center might be inbetween the fibers.

        Parameters
        ----------
        AmplData : 2D np.array
            Contains the 2D amplitude image
        nPoints : int
            Number of pixels with the largest amplitude to use

        Returns
        -------
        int, int
            Average coordinates of the maximum amplitude
        """
        # Create a 1D array of the 2D image
        FlatImage = AmplData.ravel()
        # Check how many NaNs are in the data
        nanCount = np.isnan(FlatImage).sum()
        # Partition the array, i.e., put all indices of values larger than the
        # k-th highest value behind it, everything else before it
        partitionedImg = np.argpartition(FlatImage, -nPoints - nanCount)
        # Get the indices of the highest amplitude pixels and all NaNs
        maxIndices = partitionedImg[-nPoints - nanCount:]
        # Get rid of the NaNs
        maxIndices = maxIndices[~np.isnan(FlatImage[maxIndices])]
        # Get original indices (= coordinates)
        maxIndices = np.unravel_index(maxIndices, AmplData.shape)
        maxIndices = np.column_stack(maxIndices)
        Center = np.mean(np.column_stack(maxIndices), axis=1)
        Center = np.round(Center).astype(int)[::-1]
        self.Sentinel.debug('Center detected at {} px'.format(Center))
        return Center

    def getVerticalLines(self, Img, Xmin, Xmax, YCenter, MaxDistPx):
        """
        Extracts the data between Xmin and Xmax around YCenter +/- MaxDist
        from the image in vertical lines.

        Parameters
        ----------
        Img : 2D np.array
            The amplitude or phase as 2D np.array. Indices infer pixel
            coordinates
        Xmin : int
            Horizontal pixel index of the boundary on the left
        Xmax : int
            Horizontal pixel index of the boundary on the right
        YCenter : int
            Y position of the center pixel
        MaxDistPx : float, positive
            The distance in pixels from the center for data collection

        Returns
        -------
        xData : 1D np.array of float
            List with the distance data for the values in yData. Plotting
            xData vs yData is the phase/linearized amplitude vs distance.
            This is in mm from the center
        yDataTop : 1D np.array of float
            Values of the corresponding pixels on the top half of the image,
            including the center
        yDataBot : 1D np.array of float
            Values of the corresponding pixels on the bottom half of the image,
            including the center
        """
        # Get an integer to use as index
        MaxDistPx = int(round(MaxDistPx, 0))
        # Check if MaxDistPx is too big for evaluation
        # Keep in mind that MaxDistPx = 2 * MaxDist
        MaxDistPxTop = MaxDistPx
        MaxDistPxBot = MaxDistPx
        if YCenter + 0.5 * MaxDistPx > Img.shape[0]:
            MaxDistPxTop = Img.shape[0] - YCenter - 1
        if YCenter - 0.5 * MaxDistPx < 0:
            MaxDistPxBot = YCenter - 1

        if MaxDistPxTop != MaxDistPx:
            Msg = 'The distance from the center {} px is too big for the '\
                  'topside evaluation. It was changed to {}'\
                  ''.format(MaxDistPx, MaxDistPxTop)
            self.Sentinel.warning(Msg)
        if MaxDistPxBot != MaxDistPx:
            Msg = 'The distance from the center {} px is too big for the '\
                  'bottomside evaluation. It was changed to {}'\
                  ''.format(MaxDistPx, MaxDistPxBot)
            self.Sentinel.warning(Msg)

        # Take all data if MaxDistPx is rather big
        if (YCenter + MaxDistPxTop) > Img.shape[0]:
            MaxDistPxTop = Img.shape[0] - YCenter - 1
        if (YCenter - MaxDistPxBot) < 0:
            MaxDistPxBot = YCenter

        # Get all relevant data
        upperHalf = Img[YCenter + 1:YCenter + MaxDistPxTop + 1, Xmin:Xmax + 1]
        center = Img[YCenter, Xmin:Xmax + 1]
        lowerHalf = Img[YCenter - MaxDistPxBot:YCenter, Xmin:Xmax + 1][::-1]

        # Concatenate all the data
        yDataTop = np.hstack([center, np.ravel(upperHalf)])
        yDataBot = np.hstack([center, np.ravel(lowerHalf)])

        # Add NaNs if evaluation is longer in one direction
        diff = MaxDistPxTop - MaxDistPxBot

        if diff > 0:
            yDataBot = np.hstack(
                [yDataBot, np.full((center.size * diff), np.nan)])
        elif diff < 0:
            yDataTop = np.hstack(
                [yDataTop, np.full((center.size * -diff), np.nan)])

        # Create the distance data
        if diff > 0:
            xData = np.ravel([[row + 1] * upperHalf.shape[1]
                              for row in range(upperHalf.shape[0])])
        else:
            xData = np.ravel([[row + 1] * lowerHalf.shape[1]
                              for row in range(lowerHalf.shape[0])])
        xData = np.hstack([center.size * [0], xData])
        xData = xData * self.RScale

        return xData, (yDataTop, yDataBot)

    def extractFiberPixels(self, Meas, Width, Ref=None):
        """
        Identifies the vertical pixel line with the highest
        amplitude.

        Parameters
        ----------
        Meas : LIT_SingleFiberMeas object
            Contains all raw and processed data from a
            single fiber measurement
        Width : int
           Number of vertical pixel lines to extract
           on both sides of the center
        Ref : LIT_SingleFiberMeas object
            A reference measurement. This will be used if self.AutoRun is True.
            Only one measurement will be shown to the user. For all other
            measurements, the same angle and area will be evaluated
        """
        # compute the measurement ID
        newID = str(Meas.LockInFreq).replace('.', '_') + '_Hz'

        # Check if the name already exists
        allFreqs = self.SweepData.getFreqList()
        allIDs = self.SweepData.getMeasIDs()
        if allFreqs.count(Meas.LockInFreq) > 1:
            duplicate = True
            i = 1
            while duplicate:
                numberedID = newID + '_{:03d}'.format(i)
                if numberedID in allIDs:
                    i += 1
                else:
                    duplicate = False
                    newID = numberedID
        Meas.ID = newID

        # let the user select a line showing the approximate fiber position
        if (self.AutoRun is False) or (Ref is None):
            self.SelectFiberRegion(Meas)

        self.FocusOnFiber(Meas, Width, Ref)

        # let the user select the vertical boundaries
        if self.AutoRun is False or Ref is None:
            self.SelectBoundaries(Meas)
        else:
            Meas.MinDist = Ref.MinDist
            Meas.MaxDist = Ref.MaxDist

        Xmin = Meas.Center[0] - Width
        Xmax = Meas.Center[0] + Width

        # extract the amplitude lines
        Meas.LinDist, LinAmplData = self.getVerticalLines(
            Meas.AmplData, Xmin, Xmax, Meas.Center[1],
            2 * Meas.MaxDist / Meas.RScale)
        Meas.LinAmplDataTop = np.log(LinAmplData[0])
        Meas.LinAmplDataBot = np.log(LinAmplData[1])
        # extract the phase lines
        LinDist, LinearPhase = self.getVerticalLines(
            Meas.PhaseData, Xmin, Xmax, Meas.Center[1],
            2 * Meas.MaxDist / Meas.RScale)
        Meas.LinPhaseDataTop = LinearPhase[0]
        Meas.LinPhaseDataBot = LinearPhase[1]

    def AnalyzeFiberLITMeas(self, RawDataFolder, MinDist, MaxDist, Width=0,
                            SaveRawData=True, SaveNumpyData=False):
        """
        Analyzes the LIT measurement of a thin fiber
        excited with a point source (line or point laser)
        at one or multiple frequencies.

        Parameters
        ----------
        RawDataFolder : Path object
            path to the folder containing all the ASCII raw data files
        MinDist : float, >0 in [mm]
            default minimal distance used for the linear fit
        MaxDist : float, >0 in [mm]
            default maximal distance used for the linear fit
        Width : int, default 0
            Number of vertical pixel lines to extract left and right of center
            default is 0 (only central line)
        SaveRawData : bool, default True
            If True, save the raw data
        SaveNumpyData : str or bool, default False
            Can be one of ['lin', '2D', 'both'] to save data as numpy files
        """
        self.RawDataFolderPath = RawDataFolder
        self.SweepData = LIT_SweepData(self.RawDataFolderPath,
                                       self.RScale,
                                       self.ExpName, 'Fiber')
        self.SweepData.Width = Width

        Ref = None
        for Meas in self.SweepData.SMeasList:
            self.Sentinel.info(
                'Now processing File: \'{}\''.format(Meas.AmplPath))

            if Ref is None:
                userInteraction = True
                # save the default fit distances
                Meas.MinDist = MinDist
                Meas.MaxDist = MaxDist
            else:
                userInteraction = False
                Meas.MinDist = Ref.MinDist
                Meas.MaxDist = Ref.MaxDist

            # extract the pixel lines with the fiber
            self.extractFiberPixels(Meas, Width, Ref)

            # Check if the center is very close to the top/bot
            if np.isnan(Meas.LinAmplDataTop).sum() / len(Meas.LinAmplDataTop) > 0.9:
                evalTop = False
                self.Sentinel.warning(
                    'Top evaluation will be ignored, because more than '
                    '90 % of values are NaN.')
            else:
                evalTop = True
            if np.isnan(Meas.LinAmplDataBot).sum() / len(Meas.LinAmplDataBot) > 0.9:
                evalBot = False
                self.Sentinel.warning(
                    'Bottom evaluation will be ignored, because more than '
                    '90 % of values are NaN.')
            else:
                evalBot = True

            # fit the linearized amplitude
            botInteraction = userInteraction
            if evalTop:
                Meas.AmplFitDataTop = self.LinFitRegion(
                    Meas.LinDist, Meas.LinAmplDataTop, 'Amplitude',
                    Meas.MinDist, Meas.MaxDist,
                    Meas.LockInFreq, botInteraction)
                Meas.MinDist = Meas.AmplFitDataTop[2][0][0]
                Meas.MaxDist = Meas.AmplFitDataTop[2][1][0]
                botInteraction = False
            if evalBot:
                Meas.AmplFitDataBot = self.LinFitRegion(
                    Meas.LinDist, Meas.LinAmplDataBot, 'Amplitude',
                    Meas.MinDist, Meas.MaxDist,
                    Meas.LockInFreq, botInteraction)
                if botInteraction:
                    Meas.MinDist = Meas.AmplFitDataBot[2][0][0]
                    Meas.MaxDist = Meas.AmplFitDataBot[2][1][0]
            Meas.plotAmplFit(self.SweepData.PlotPath1D)
            # fit the linearized phase
            if evalTop:
                Meas.PhaseFitDataTop = self.LinFitRegion(
                    Meas.LinDist, Meas.LinPhaseDataTop, 'Phase',
                    Meas.MinDist, Meas.MaxDist,
                    Meas.LockInFreq, userInteraction)
                Meas.MinDist = Meas.PhaseFitDataTop[2][0][0]
                Meas.MaxDist = Meas.PhaseFitDataTop[2][1][0]
            if evalBot:
                Meas.PhaseFitDataBot = self.LinFitRegion(
                    Meas.LinDist, Meas.LinPhaseDataBot, 'Phase',
                    Meas.MinDist, Meas.MaxDist,
                    Meas.LockInFreq, botInteraction)
                if botInteraction:
                    Meas.MinDist = Meas.PhaseFitDataBot[2][0][0]
                    Meas.MaxDist = Meas.PhaseFitDataBot[2][1][0]
            Meas.plotPhaseFit(self.SweepData.PlotPath1D)

            if self.AutoRun is True and Ref is None:
                self.Sentinel.info('From this point on the software runs '
                                   'automatically if no errors occur.\n'
                                   '- - - - - - - - - - - - - - - - - - -')
            if np.isnan(Meas.LinAmplDataTop).any() is False:
                Meas.MinDist = max(Meas.AmplFitDataTop[2][0][0], Meas.MinDist)
                Meas.MaxDist = min(Meas.PhaseFitDataTop[2][1][0], Meas.MaxDist)
            MinDist = Meas.MinDist
            MaxDist = Meas.MaxDist

            # Create plots
            Meas.plotFiberAmplZoom(self.SweepData.PlotPathZoom2D, Width)
            Meas.plotFiberPhaseZoom(self.SweepData.PlotPathZoom2D, Width)

            # compute the thermal diffusivity
            if evalTop:
                ThermalDiffTop = abs((np.pi * Meas.LockInFreq) /
                                     (Meas.AmplFitDataTop[0][0] *
                                      Meas.PhaseFitDataTop[0][0]))
            else:
                ThermalDiffTop = np.nan
            if evalBot:
                ThermalDiffBot = abs((np.pi * Meas.LockInFreq) /
                                     (Meas.AmplFitDataBot[0][0] *
                                      Meas.PhaseFitDataBot[0][0]))
            else:
                ThermalDiffBot = np.nan
            Meas.ThermalDiff = [ThermalDiffTop, ThermalDiffBot]

            StdThermalDiffTop = self.computeSAlpha(Meas.LockInFreq,
                                                   Meas.AmplFitDataTop,
                                                   Meas.PhaseFitDataTop)
            StdThermalDiffBot = self.computeSAlpha(Meas.LockInFreq,
                                                   Meas.AmplFitDataBot,
                                                   Meas.PhaseFitDataBot)
            Meas.StdThermalDiff = [StdThermalDiffTop, StdThermalDiffBot]

            self.Sentinel.info('Thermal diffusivity is {} mm^2/s for lock-in '
                               'frequency {} Hz'.format(
                                   Meas.ThermalDiff, Meas.LockInFreq))

            if Ref is None and self.AutoRun is True:
                Ref = Meas

        if self.SweepData.MultiFreq is True:
            self.SweepData.AlphaSlope = self.calcAlphaLinReg(self.SweepData)
            self.SweepData.plotAlphaLinReg()
            self.Sentinel.info('Thermal diffusivity from the linear '
                               'regression is  {} +/- {} mm^2/s'.format(
                                   self.SweepData.AlphaSlope[0],
                                   self.SweepData.AlphaSlope[1]))
            self.SweepData.plotTDiffvsFreq()

        # save the data
        self.SweepData.saveData(SaveRawData, SaveNumpyData)
