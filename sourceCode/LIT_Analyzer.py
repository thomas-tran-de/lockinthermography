import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.stats import linregress


class LIT_Analyzer():
    """Class to analyze LIT measurements of a single sample"""
    Sentinel = logging.getLogger("Watchtower")

    def __init__(self, ExpName, ExpFolder, AutoRun):
        """
        Initialize LIT_Analyzer class

        Parameters
        ----------
        ExpName : str
           Name part used for all saved data files
        ExpFolder : Path object
            Path to the experiment folder. Files will be saved to this
            directory
        AutoRun : bool
            If False, the linear fit region with the corresponding fit is
            shown. This is useful to determine the boundaries for the fit
            region.
        """
        self.ExpName = ExpName
        self.ExpFolder = ExpFolder
        self.AutoRun = AutoRun

        # initialize all variables
        self.RawDataFolderPath = None
        self.RScale = None
        self.SweepData = None

    def setRScale(self, NewRScale):
        """
        Set a pixel scaling for all samples.

        Parameters
        ----------
        NewRScale : float, positive
            Pixel size of the IR camera in mm / pixel

        Raises
        ------
        ValueError : If NewRScale is <= 0
        """
        if NewRScale <= 0:
            ErrorMsg = f"The new RScale {NewRScale} is less then 0. "\
                       "Use a positive number."
            self.Sentinel.error(ErrorMsg)
            raise ValueError(ErrorMsg)
        self.RScale = NewRScale

    def setRawDataFolderPath(self, NewRawDataPath):
        self.RawDataFolderPath = NewRawDataPath

    def computeSAlpha(self, LockInFreq, AmplFitList, PhaseFitList):
        """
        Computes the standard deviation of the thermal diffusivity
        by Gaussian error propagation

        Parameters
        ----------
        LockInFreq : float, positive
            Frequency of the lock in measurements
        AmplFitList, PhaseFitList : list
            contains: [[slope, y-intersect], std_slope, EndPosList]

        Returns
        -------
        Std : float
            The standard deviation in mm^2/s
        """
        if None in [AmplFitList, PhaseFitList]:
            return np.nan
        Std = (np.pi * LockInFreq /
               (AmplFitList[0][0] * PhaseFitList[0][0]**2) *
               PhaseFitList[1])**2
        Std += (np.pi * LockInFreq /
                (AmplFitList[0][0]**2 * PhaseFitList[0][0]) *
                AmplFitList[1])**2
        return Std

    def calcAlphaLinReg(self, Sweep):
        """
        Computes the overall thermal diffusivity from a range of different
        frequencies. The product of the slopes is fitted with a line that
        passes through the origin.

        Parameters
        ----------
        Sweep : LIT_SweepData
            The sweep data containing all relevant information, i.e.,
            the frequencies, the amplitude slope, the phase slope, and
            the standard deviation of both slopes

        Returns
        -------
        [AlphaLinReg, SAlphaLinReg,
         SlopeProd, SProdError, LinearSlope] : list of float
            Thermal diffusivity and standard deviation in mm^2/s
            Product of the amplitude and phase slopes with error
            Slope of the linear fit
        """
        FreqList = Sweep.getFreqList()
        [m_Phase, SPFitList] = Sweep.getPhaseSlopeList()
        [m_Ampl, SAFitList] = Sweep.getAmplSlopeList()

        # multiply the slopes of amplitude and phase
        SlopeProd = np.array(m_Phase) * np.array(m_Ampl)
        SProdError = np.sqrt((np.array(m_Phase) * np.array(SAFitList))**2 +
                             (np.array(m_Ampl) * np.array(SPFitList))**2)

        # initial guess for the slope of the slope product
        LinearSlopeInit = sum(abs(SlopeProd) * np.array(FreqList)) /\
            sum(np.array(FreqList)**2)

        def fitFunc(f, m):
            return f * m

        Fit = curve_fit(fitFunc, FreqList, SlopeProd,
                        p0=LinearSlopeInit)

        LinearSlope = abs(Fit[0][0])
        SLinSlope = np.sqrt(Fit[1][0][0])

        # linear fit of the product versus frequency
        AlphaLinReg = abs(np.pi / LinearSlope)
        SAlphaLinReg = abs(np.pi / LinearSlope**2 * SLinSlope)

        return [AlphaLinReg, SAlphaLinReg, SlopeProd, SProdError, LinearSlope]

    def calcAlphaLinRegAniso(self, Sweep):
        """
        Computes the overall thermal diffusivity from a range of different
        frequencies. The product of the slopes is fitted with a line that
        passes through the origin. Comparable to self.calcAlphaLinReg(),
        but distinguishes between two slopes for anisotropic samples

        Parameters
        ----------
        Sweep : LIT_SweepData
            The sweep data containing all relevant information, i.e.,
            the frequencies, the amplitude slope, the phase slope, and
            the standard deviation of both slopes

        Returns
        -------
        [AlphaLinReg, SAlphaLinReg,
         SlopeProd, SProdError, LinearSlope] : lits of tuple of float
            Thermal diffusivity and standard deviation in mm^2/s
            Product of the amplitude and phase slopes with error
            Slope of the linear fit
            Each variable is a tuple for the long and short axis
        """
        FreqList = Sweep.getFreqList()
        [m_Phase, SPFitList] = Sweep.getPhaseSlopeList()
        [m_Ampl, SAFitList] = Sweep.getAmplSlopeList()

        # multiply the slopes of amplitude and phase
        SlopeProd = np.array(m_Phase) * np.array(m_Ampl)
        SProdError = np.sqrt((np.array(m_Phase) * np.array(SAFitList))**2 +
                             (np.array(m_Ampl) * np.array(SPFitList))**2)

        # split the values for long and short axis
        SlopeProd = SlopeProd.reshape((2, len(FreqList)), order='F')
        SProdError = SProdError.reshape((2, len(FreqList)), order='F')

        # initial guess for the slope of the slope product
        LinearSlopeInit = (sum(abs(SlopeProd[0]) * np.array(FreqList)) /
                           sum(np.array(FreqList)**2),
                           sum(abs(SlopeProd[1]) * np.array(FreqList)) /
                           sum(np.array(FreqList)**2))

        def fitFunc(f, m):
            return f * m

        FitLong = curve_fit(fitFunc, FreqList, SlopeProd[0],
                            p0=LinearSlopeInit[0])
        FitShort = curve_fit(fitFunc, FreqList, SlopeProd[1],
                             p0=LinearSlopeInit[1])

        LinearSlope = (abs(FitLong[0][0]), abs(FitShort[0][0]))
        SLinSlope = (np.sqrt(FitLong[1][0][0]), np.sqrt(FitShort[1][0][0]))

        # linear fit of the product versus frequency
        AlphaLinReg = [abs(np.pi / ls) for ls in LinearSlope]
        SAlphaLinReg = [abs(np.pi / LinearSlope[i]**2 * SLinSlope[i])
                        for i in range(len(LinearSlope))]

        return [AlphaLinReg, SAlphaLinReg, SlopeProd, SProdError, LinearSlope]

    def LinearFit(self, X, Y):
        """
        Computes a linear fit on the input (X, Y) data

        Returns
        -------
        values : list of float
            slope and y intersection of the linear fit
        error : float
            standard deviation for the slope
        """
        # Ignore NaNs
        idx = np.isfinite(X) & np.isfinite(Y)
        try:
            Lin_Fit = linregress(X[idx], Y[idx])
            values = Lin_Fit[:2]
            error = Lin_Fit[4]
        except ValueError:
            self.Sentinel.warning("No values found for linear fit")
            values = [np.nan, np.nan]
            error = np.nan
        return [values, error]

    def LinFitRegion(self, LinDist, LinY, Type, MinDist,
                     MaxDist, Frequency, ManualBounds=None):
        """
        Performs a linear fit on the linearized (Dist, Y) data.
        The predefined distance boundaries are used by default.
        The user can also select the boundaries by clicking on the plot.

        Parameters
        ----------
        LinDist : np.array
            Array with the linearized distance from the laser heating
        LinY : np.array
            Array with the linearized amplitude data or the phase data
        Type : str
            Either "Amplitude" or "Phase"
        MinDist : float, positive
            Minimal distance from the heating origin to use in mm
        MaxDist : float, positive
            Maximal distance from the heating origin to use in mm
        Frequency : float, positive
            Lock-in frequency of the measurement
        ManualBounds : bool, optional
            If True, the user has to choose the boundaries for the linear fit.
            This is useful for the first measurement of a series or frequency
            sweeps when boundaries must be adjusted.
            ManualBounds overwrites self.AutoRun. Default is None

        Raises
        ------
        ValueError if Type is not recognized

        Returns
        -------
        Y_Fit : list of float
            slope and y intersection of the linear fit
        S_Y_Fit : float
            standard deviation for the slope
        EndPosList : list of list
            Index and value of the minimal and maximal distance used
        """
        def onselect(xmin, xmax):
            self.Sentinel.debug(
                f"User selected region between {xmin:.3f} and {xmax:.3f}")

            # get the indices of the new borders
            Left, Right = np.searchsorted(LinDist, (xmin, xmax))
            if Right == LinDist.size:
                Right -= 1
            EndPosList[:] = [[xmin, Left], [xmax, Right]]

            # redo the linear fit
            Y_Fit, S_Y_Fit = self.LinearFit(LinDist[Left:Right],
                                            LinY[Left:Right])
            # plot the linear fit
            X_Line = [LinDist[Left], LinDist[Right]]
            Y_Line = Y_Fit[0] * np.array(X_Line) + Y_Fit[1]
            FitGraph.set_data(X_Line, Y_Line)
            fig.canvas.draw_idle()  # redraw everything
            return True

        # list for the boundaries on the [left, right]
        Left, Right = np.searchsorted(LinDist, (MinDist, MaxDist))
        if Right == LinDist.size:
            Right -= 1
        EndPosList = [[MinDist, Left], [MaxDist, Right]]

        fig = plt.figure()
        ax = plt.subplot()

        # plot the measured amplitude data
        ax.plot(LinDist, LinY, ".", markersize=1)

        # do the linear fit
        Y_Fit, S_Y_Fit = self.LinearFit(LinDist[Left:Right],
                                        LinY[Left:Right])

        if (ManualBounds is True) or\
                (self.AutoRun is False and ManualBounds is None):
            # plot the linear fit
            X_Line = [LinDist[Left], LinDist[Right]]
            Y_Line = Y_Fit[0] * np.array(X_Line) + Y_Fit[1]

            # create a plot for the linear fit
            FitGraph, = ax.plot(X_Line, Y_Line, "r-", lw=2, label="Linear Fit")
            ax.set_xlabel("Distance / mm")

            # create a SpanSelector widget for the graph
            span = SpanSelector(
                ax,
                onselect,
                "horizontal",
                useblit=True,
                span_stays=True,
                interactive=True,
                grab_range=20,
                props=dict(alpha=0.3, facecolor='tab:gray'),
            )
            span.extents = (MinDist, MaxDist)
            span.onselect(MinDist, MaxDist)

            if Type == "Amplitude":
                ax.set_ylabel("Linearized Amplitude")

            elif Type == "Phase":
                ax.set_ylabel("Phase / rad")
                NewYMin = min(
                    [P for P in LinY if -2 * np.pi < P < 2 * np.pi])
                NewYMax = max(
                    [P for P in LinY if -2 * np.pi < P < 2 * np.pi])

                # Set the base value for tick locators based on the limits
                phaseTicker = MaxNLocator(
                    'auto', min_n_ticks=3,
                    steps=[0.5 * np.pi, np.pi, 2 * np.pi,
                           2.5 * np.pi, 3 * np.pi])
                ax.yaxis.set_major_locator(phaseTicker)
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda x, pos: f'{x / np.pi} $\\pi$'))
                ax.set_ylim(NewYMin, NewYMax)

            else:
                ErrorMsg = f"The Type {Type} for the linearized plot is not "\
                           "recognized. Use either 'Amplitude' or 'Phase'"
                self.Sentinel.error(ErrorMsg)
                raise ValueError(ErrorMsg)

            ax.set_title(f"Frequency: {Frequency:.3f} Hz")
            fig.tight_layout()
            self.Sentinel.info('Select the area for the linear fit.')
            plt.show()

            # Update the fit params
            Left = EndPosList[0][1]
            Right = EndPosList[1][1]
            Y_Fit, S_Y_Fit = self.LinearFit(LinDist[Left:Right],
                                            LinY[Left:Right])
        plt.close('all')
        self.Sentinel.info(f"Linear fit for {Type} completed.")
        return [Y_Fit, S_Y_Fit, EndPosList]
