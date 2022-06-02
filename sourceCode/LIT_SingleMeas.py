import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter


class LIT_SingleMeas():
    """ Class to handle all the data from a single measurement """
    Sentinel = logging.getLogger('Watchtower')

    def __init__(self, AmplPath, PhasePath, RScale, ExpName):
        """
        Initialize the SingleFreqData class

        Parameters
        ----------
        AmplPath : Path
            Path to the file containing the 2D amplitude data for the sample
        PhasePath : Path
            Path to the file containing the 2D phase data for the sample
        RScale : float
            Conversion factor from pixel to mm in mm / Pixel
        ExpName : str
            Name of the experiment, not the single measurement
        """
        self.AmplPath = AmplPath
        self.PhasePath = PhasePath
        self.ExpName = ExpName
        self.RScale = RScale
        self.MinDist = None    # minimal
        self.MaxDist = None    # and maximal distance for linear fits
        self.ID = None
        self.Orientation = None

        # read in the raw 2D data
        self.RawAmplData = self.getRawDataLIT(self.AmplPath, 'Amplitude')
        self.RawPhaseData = self.getRawDataLIT(self.PhasePath, 'Phase')
        self.LockInFreq = self.RawAmplData[3]

        # Function to create matplotlib patch showing the area of reduced data
        # in raw data. This has to be a function, because otherwise it cannot
        # be reused!
        self.ROIBorder = None

        # Data to be analyzed
        # CAUTION: While the Raw...Data consists of 3 np.arrays, ...Data is
        # a 2D np.array. It consists only of measurement values. Coordinates
        # are inferred by the index
        self.AmplData = None
        self.PhaseData = None

        # data for the single frequency linear fits
        self.AmplFitData = None
        self.PhaseFitData = None
        self.LinDist = None
        self.LinAmplData = None
        self.LinPhaseData = None
        self.ThermalDiff = None
        self.StdThermalDiff = None

        # paths to the different plots
        self.AmplFitPlotPath = None
        self.PhaseFitPlotPath = None
        self.AmplZoomPlotPath = None
        self.PhaseZoomPlotPath = None

    def getAmplCanvas(self, Type='reduced'):
        """
        Creates the figure and axis with the default amplitude
        plot. Can be extended by different other functions later.

        Parameters
        ----------
        Type : str, optional
            'reduced' (default) to get a canvas showing only the data to be
            analyzed. This is useful for zoomed plots.
            'raw' to get a canvas of the raw data.

        Returns
        -------
        fig : matplotlib.figure object
            Contains the figure with the plot
        """
        fig = plt.figure()
        ax = plt.subplot(aspect='equal')

        if Type == 'raw':
            plot = ax.imshow(
                self.RawAmplData[2][::-1],
                extent=[0, self.RawAmplData[0][-1][-1] * self.RScale,
                        0, self.RawAmplData[1][-1][-1] * self.RScale],
                norm=LogNorm(vmin=np.nanmin(self.RawAmplData[2]),
                             vmax=np.nanmax(self.RawAmplData[2])),
                cmap='viridis')
        elif Type == 'reduced':
            x = np.arange(self.AmplData.shape[1]) * self.RScale
            y = np.arange(self.AmplData.shape[0]) * self.RScale
            plot = ax.pcolormesh(x, y, self.AmplData,
                                 norm=LogNorm(vmin=np.nanmin(self.AmplData),
                                              vmax=np.nanmax(self.AmplData)),
                                 cmap='viridis', shading='nearest')
        else:
            ErrMsg = 'The type \'{}\' has not been recognized. Please use '\
                     'either \'raw\' or \'reduced\'.'.format(Type)
            self.Sentinel.error(ErrMsg)
            raise ValueError(ErrMsg)
        ax.set_title('Amplitude / mK, f = {} Hz'.format(self.LockInFreq))
        ax.set_xlabel('x-position / mm')
        ax.set_ylabel('y-position / mm')
        ax.yaxis.set_major_locator(MaxNLocator())
        ax.xaxis.set_major_locator(MaxNLocator(
            'auto', steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
        fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        return fig

    def getRawAmplCanvas(self):
        return self.getAmplCanvas('raw')

    def getPhaseCanvas(self, Type='reduced'):
        """
        Creates the figure and axis with the default phase
        plot. Can be extended by different other functions later.

        Parameters
        ----------
        Type : str, optional
            'reduced' (default) to get a canvas showing only the data to be
            analyzed. This is useful for zoomed plots.
            'raw' to get a canvas of the raw data.

        Returns
        -------
        fig : matplotlib.figure object
            Contains the figure with the plot
        """
        fig = plt.figure()
        ax = plt.subplot(aspect='equal')

        if Type == 'raw':
            plot = ax.imshow(
                self.RawPhaseData[2][::-1],
                extent=[0, self.RawPhaseData[0][-1][-1] * self.RScale,
                        0, self.RawPhaseData[1][-1][-1] * self.RScale],
                cmap='bwr')
        elif Type == 'reduced':
            x = np.arange(self.PhaseData.shape[1]) * self.RScale
            y = np.arange(self.PhaseData.shape[0]) * self.RScale
            plot = ax.pcolormesh(x, y, self.PhaseData,
                                 cmap='bwr', shading='nearest')
        else:
            ErrMsg = 'The type \'{}\' has not been recognized. Please use '\
                     'either \'raw\' or \'reduced\'.'.format(Type)
            self.Sentinel.error(ErrMsg)
            raise ValueError(ErrMsg)

        ax.set_title('Phase / DEG, f = ' + str(self.LockInFreq) + ' Hz')
        ax.set_xlabel('x-position / mm')
        ax.set_ylabel('y-position / mm')
        ax.yaxis.set_major_locator(MaxNLocator())
        ax.xaxis.set_major_locator(MaxNLocator(
            'auto', steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
        cbar = fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        yax = cbar.ax.yaxis
        yax.set_major_locator(
            MaxNLocator('auto', min_n_ticks=3,
                        steps=[0.5 * np.pi, np.pi, 2 * np.pi,
                               2.5 * np.pi, 3 * np.pi]))
        yax.set_major_formatter(
            FuncFormatter(lambda x, pos: f'{x / np.pi} $\\pi$'))
        return fig

    def getRawPhaseCanvas(self):
        return self.getPhaseCanvas('raw')

    def getRawDataLIT(self, Path, Type):
        """
        Reads the 2D ASCII images from the files

        Parameters
        ----------
        Path : Path
            Path to the file containing the 2D data
        Type : str
            Either 'Amplitude' or 'Phase'
        """
        DataFile = open(str(Path.absolute()), 'r', encoding='latin-1')
        datalines = DataFile.readlines()
        DataFile.close()

        # look for the measurement parameters
        lockInFreq = False
        width = False
        height = False
        dataStart = False
        for i, line in enumerate(datalines):
            if line.startswith('LIT.LockInFrequency'):
                lockInFreq = float(line.split('=')[-1])
            elif line.startswith('ImageWidth'):
                width = int(line.split('=')[-1])
            elif line.startswith('ImageHeight'):
                height = int(line.split('=')[-1])
            elif line.startswith('[Data]'):
                dataStart = i + 1
            if lockInFreq and width and height and dataStart:
                break

        # Load data
        data = np.loadtxt(
            Path, delimiter=';', skiprows=dataStart,
            max_rows=height, usecols=range(width))

        if Type == 'Amplitude':
            # Prevent negative values, because log(-x) is illegal
            data = np.maximum(data, 0.001)
        elif Type == 'Phase':
            data = data * np.pi / 180

        xPxPos, yPxPos = np.meshgrid(range(width), range(height))
        # np.flipud() to flip the data upside down
        return [xPxPos, yPxPos, data, lockInFreq]

    def plotRawAmplitude(self, PlotPathRaw2D):
        """
        Plots the 2D amplitude data into the PlotPathRaw2D folder

        Parameters
        ----------
        PlotPathRaw2D : Path object
            path to the folder for the raw 2D plots
        """
        self.getRawAmplCanvas()
        PlotPath = PlotPathRaw2D / (str(self.AmplPath.stem[:-4]) +
                                    '_Raw2DAmpl.png')
        plt.savefig(PlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info(str(self.AmplPath.stem[:-4]) +
                           ' 2D Raw Amplitude plotted')

    def plotRawPhase(self, PlotPathRaw2D):
        """
        Plots the 2D phase data into the PlotPathRaw2D folder

        Parameters
        ----------
        PlotPathRaw2D : Path object
            path to the folder for the raw 2D plots
        """
        self.getRawPhaseCanvas()
        PlotPath = PlotPathRaw2D / (str(self.PhasePath.stem[:-6]) +
                                    '_Raw2DPhase.png')
        plt.savefig(PlotPath.absolute(), dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info(str(self.PhasePath.stem[:-6]) +
                           ' 2D Raw Phase plotted')

    def plotAmplFit(self, PlotPath1D, Anisotropic=False):
        """
        Plots the linear amplitude fit and save it in PlotPath1D

        Parameters
        ----------
        PlotPath1D : Path object
            Path to the folder for the 1D plots
        Anisotropic : bool, optional
            If True, separately save data for long and short axis
        """
        # Check if this is an anisotropic measurement
        if Anisotropic:
            origID = self.ID
            self.ID = origID + '_Long'
            self.AmplFitData = self.AmplFitDataLong
            self.LinDist = self.LinDistLong
            self.LinAmplData = self.LinAmplDataLong
            self.plotAmplFit(PlotPath1D)
            self.ID = origID + '_Short'
            self.AmplFitData = self.AmplFitDataShort
            self.LinDist = self.LinDistShort
            self.LinAmplData = self.LinAmplDataShort
            self.plotAmplFit(PlotPath1D)
            self.ID = origID
            return

        # unpack the data
        EndPosList = self.AmplFitData[2]
        Left = EndPosList[0][1]
        Right = EndPosList[1][1]

        plt.figure()
        ax = plt.subplot()

        # create a rectangle showing where data was evaluated
        ax.axvspan(self.LinDist[Left], self.LinDist[Right], fc='lightgray')

        # Plot the data
        ax.plot(self.LinDist, self.LinAmplData, '.', markersize=1)

        # plot the linear fit
        X_Line = [self.LinDist[Left], self.LinDist[Right]]
        Y_Line = self.AmplFitData[0][0] * np.array(X_Line) +\
            self.AmplFitData[0][1]
        FitGraph, = ax.plot(X_Line, Y_Line, 'r-', lw=2, label='Linear Fit')

        # finalize the graph
        ax.set_xlabel('Distance / mm')
        ax.set_ylabel('Linearized Amplitude')
        ax.set_title('Amplitude, f = ' + str(self.LockInFreq) + ' Hz')
        ax.set_ylim(max([-2, min(self.LinAmplData)]))
        self.AmplFitPlotPath = PlotPath1D /\
            (self.ExpName + '_' + self.ID + '_AmplFit.png')
        plt.savefig(self.AmplFitPlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info('Linear amplitude fit for ' + self.ExpName +
                           '_' + self.ID + ' plotted.')

    def plotPhaseFit(self, PlotPath1D, Anisotropic=False):
        """
        Plots the linear phase fit and save it in PlotPath1D

        Parameters
        ----------
        PlotPath1D : Path object
            path to the folder for the 1D plots
        Anisotropic : bool, optional
            If True, separately save data for long and short axis
        """
        # Check if this is an anisotropic measurement
        if Anisotropic:
            origID = self.ID
            self.ID = origID + '_Long'
            self.PhaseFitData = self.PhaseFitDataLong
            self.LinDist = self.LinDistLong
            self.LinPhaseData = self.LinPhaseDataLong
            self.plotPhaseFit(PlotPath1D)
            self.ID = origID + '_Short'
            self.PhaseFitData = self.PhaseFitDataShort
            self.LinDist = self.LinDistShort
            self.LinPhaseData = self.LinPhaseDataShort
            self.plotPhaseFit(PlotPath1D)
            self.ID = origID
            return

        # unpack the data
        EndPosList = self.PhaseFitData[2]
        Left = EndPosList[0][1]
        Right = EndPosList[1][1]

        plt.figure()
        ax = plt.subplot()

        # create a rectangle showing where data was evaluated
        ax.axvspan(self.LinDist[Left], self.LinDist[Right], fc='lightgray')

        # Save the raw data
        ax.plot(self.LinDist, self.LinPhaseData, '.', markersize=1)

        # find tight y-limits for points between -2pi and 2pi
        NewYMin = min([P for P in self.LinPhaseData if -2 * np.pi < P <
                       2 * np.pi])
        NewYMax = max([P for P in self.LinPhaseData if -2 * np.pi < P <
                       2 * np.pi])

        # plot the linear fit
        X_Line = [self.LinDist[Left], self.LinDist[Right]]
        Y_Line = self.PhaseFitData[0][0] * np.array(X_Line) +\
            self.PhaseFitData[0][1]
        FitGraph, = ax.plot(X_Line, Y_Line, 'r-', lw=2, label='Linear Fit')

        # finalize the graph
        ax.set_xlabel('Distance / mm')
        ax.set_ylabel('Phase / rad')
        ax.set_ylim(NewYMin, NewYMax)
        ax.yaxis.set_major_locator(
            MaxNLocator('auto', min_n_ticks=3,
                        steps=[0.5 * np.pi, np.pi, 2 * np.pi,
                               2.5 * np.pi, 3 * np.pi]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f'{x / np.pi} $\\pi$'))
        ax.set_title('Phase, f = ' + str(self.LockInFreq) + ' Hz')
        self.PhaseFitPlotPath = PlotPath1D /\
            (self.ExpName + '_' + self.ID + '_PhaseFit.png')
        plt.savefig(self.PhaseFitPlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info('Linear phase fit for ' + self.ExpName +
                           '_' + self.ID + ' plotted.')
