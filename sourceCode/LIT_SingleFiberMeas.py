import logging
from sourceCode.LIT_SingleMeas import LIT_SingleMeas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle


class LIT_SingleFiberMeas(LIT_SingleMeas):
    """ Class to handle all the data from a single fiber measurement """
    Sentinel = logging.getLogger('Watchtower')

    def __init__(self, AmplPath, PhasePath, RScale, ExpName):
        """
        Initialize the SingleFiberMeas class

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
        LIT_SingleMeas.__init__(self, AmplPath, PhasePath, RScale, ExpName)
        self.Type = 'Fiber'
        self.Center = []
        self.MaxDist = None
        self.FiberLine = [[], []]   # Two points, set in LIT_FiberAnalyzer
        self.FibRotAngle = None     # Rotation angle of the fiber

        # data for the single frequency linear fits
        # This is different from the LIT_SingleMeas attributes, because data
        # must be split in the upper and lower part of the fiber
        self.AmplFitDataTop = None
        self.PhaseFitDataTop = None
        self.LinAmplDataTop = None
        self.LinPhaseDataBot = None
        self.AmplFitDataBot = None
        self.PhaseFitDataBot = None
        self.LinAmplDataBot = None
        self.LinPhaseDataBot = None

    def getRawDataLIT(self, Path, Type):
        """
        Reads the 2D ASCII images from the files

        Extends the method of the parent class. The image will be rotated 90°
        automatically if there are more x pixels than y pixels

        Parameters
        ----------
        Path : Path
            Path to the file containing the 2D data
        Type : str
            Either 'Amplitude' or 'Phase'
        """
        x, y, data, freq = super().getRawDataLIT(Path, Type)
        if x.shape[1] > y.shape[0]:
            x, y, data = y, x, np.rot90(data)
        return [x, y, data, freq]

    def saveNumpyData(self, Path, Type):
        """
        Save the raw data as numpy arrays

        Parameters
        ----------
        Path : Path
            Path to a directory containing the raw data in numpy format
        Type : str
            Choose which data to save, either
                '2D':   the raw 2D data,
                'lin':  the linearized 1D data,
                'both': raw 2D and linearized 1D data
        """
        if Type in ['2D', 'both']:
            name = self.ExpName + '_' + self.ID
            np.savetxt(Path / (name + '_2dAmpl.gz'), self.AmplData)
            np.savetxt(Path / (name + '_2dPhase.gz'), self.PhaseData)
            np.savetxt(Path / (name + '_RawAmpl.gz'), self.RawAmplData[2])
            np.savetxt(Path / (name + '_RawPhase.gz'), self.RawPhaseData[2])
            np.savetxt(Path / (name + '_Center.gz'), self.Center)
            self.Sentinel.info('Saved raw data in numpy format for '
                               '{}_{}'.format(self.ExpName, self.ID))

        if Type in ['lin', 'both']:
            name = self.ExpName + '_' + self.ID
            savedAmpl = np.hstack(
                [self.LinAmplDataBot[::-1], self.LinAmplDataTop])
            savedPhase = np.hstack(
                [self.LinPhaseDataBot[::-1], self.LinPhaseDataTop])
            savedLinDist = np.hstack([-self.LinDist[::-1], self.LinDist])
            np.savetxt(Path / (name + '_Ampl.gz'), savedAmpl)
            np.savetxt(Path / (name + '_Phase.gz'), savedPhase)
            np.savetxt(Path / (name + '_LinDist.gz'), savedLinDist)

    def plotFiberAmplZoom(self, PlotPathZoom2D, Width):
        """
        Plots a zoom of the selected pixel line amplitude region.
        The location of the laser is overlayed.

        Parameters
        ----------
        PlotPathZoom2D : Path object
            Path to the folder for the zoomed 2D plots
        Width : int
            Width of the ROI in pixels
        """
        fig = self.getAmplCanvas()
        ax1 = fig.axes[0]
        cbar = fig.axes[1]

        # Add an additional axis showing an image of the analyzed data
        # instead of the whole fiber
        ax2 = fig.add_subplot(aspect='equal', label='super_zoom')
        x = np.arange(self.AmplData.shape[1]) * self.RScale
        y = np.arange(self.AmplData.shape[0]) * self.RScale
        ax2.pcolormesh(x, y, self.AmplData,
                       norm=LogNorm(vmin=np.nanmin(self.AmplData),
                                    vmax=np.nanmax(self.AmplData)),
                       cmap='viridis', shading='nearest')
        ax2.set_ylim(self.Center[1] * self.RScale - self.MaxDist * 1.2,
                     self.Center[1] * self.RScale + self.MaxDist * 1.2)

        # Move titles from the original axis to the figure/new axis
        fig.suptitle(ax1.title.get_text())
        ax1.set_title('')
        ax1.set_xlabel('')
        ax2.set_xlabel('x-position / mm')
        ax1.set_ylabel('')
        ax2.set_ylabel('y-position / mm')
        ax2.xaxis.set_major_locator(MaxNLocator(
            'auto', steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))

        # Add the center and evaluation area to the axes
        for ax in [ax1, ax2]:
            width = (2 * Width + 1) * self.RScale
            height = 2 * self.MaxDist
            rect = Rectangle(
                self.Center * self.RScale - np.array([width, height]) / 2,
                width, height,
                edgecolor='gray', facecolor='none')
            ax.add_patch(rect)
            ax.hlines(self.Center[1] * self.RScale,
                      self.Center[0] * self.RScale - width / 2,
                      self.Center[0] * self.RScale + width / 2,
                      colors='gray')

        # Create 3 columns with dynamic width
        gs = GridSpec(1, 3, top=0.9, width_ratios=[
            ax2.get_tightbbox(fig.canvas.get_renderer()).width,
            ax1.get_tightbbox(fig.canvas.get_renderer()).width,
            cbar.get_tightbbox(fig.canvas.get_renderer()).width])
        ax2.set_position(gs[0].get_position(fig))
        ax2.set_subplotspec(gs[0])
        ax1.set_position(gs[1].get_position(fig))
        ax1.set_subplotspec(gs[1])
        cbar.set_position(gs[2].get_position(fig))
        cbar.set_subplotspec(gs[2])

        self.AmplZoomPlotPath = PlotPathZoom2D /\
            (self.ExpName + '_' + self.ID + '_AmplZoom.png')
        plt.savefig(self.AmplZoomPlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info('Zoomed 2D Amplitude for {} at f = {} Hz plotted'
                           ''.format(self.ExpName, round(self.LockInFreq, 3)))

    def plotFiberPhaseZoom(self, PlotPathZoom2D, Width):
        """
        Plots a zoom of the selected pixel line phase region.
        The location of the line laser is overlayed.

        Parameters
        ----------
        PlotPathZoom2D : Path object
            Path to the folder for the zoomed 2D plots
        Width : int
            Width of the ROI in pixels
        """
        fig = self.getPhaseCanvas()
        ax1 = fig.axes[0]
        cbar = fig.axes[1]

        # Add an additional axis showing an image of the analyzed data
        # instead of the whole fiber
        ax2 = fig.add_subplot(aspect='equal', label='super_zoom')
        x = np.arange(self.PhaseData.shape[1]) * self.RScale
        y = np.arange(self.PhaseData.shape[0]) * self.RScale
        ax2.pcolormesh(x, y, self.PhaseData, cmap='bwr', shading='nearest')
        ax2.set_ylim(self.Center[1] * self.RScale - self.MaxDist * 1.2,
                     self.Center[1] * self.RScale + self.MaxDist * 1.2)

        # Move titles from the original axis to the figure/new axis
        fig.suptitle(ax1.title.get_text())
        ax1.set_title('')
        ax1.set_xlabel('')
        ax2.set_xlabel('x-position / mm')
        ax1.set_ylabel('')
        ax2.set_ylabel('y-position / mm')
        ax2.xaxis.set_major_locator(MaxNLocator(
            'auto', steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))

        # Add the center and evaluation area to the axes
        for ax in [ax1, ax2]:
            width = (2 * Width + 1) * self.RScale
            height = 2 * self.MaxDist
            rect = Rectangle(
                self.Center * self.RScale - np.array([width, height]) / 2,
                width, height,
                edgecolor='gray', facecolor='none')
            ax.add_patch(rect)
            ax.hlines(self.Center[1] * self.RScale,
                      self.Center[0] * self.RScale - width / 2,
                      self.Center[0] * self.RScale + width / 2,
                      colors='gray')

        # Create 3 columns with dynamic width
        gs = GridSpec(1, 3, top=0.9, width_ratios=[
            ax2.get_tightbbox(fig.canvas.get_renderer()).width,
            ax1.get_tightbbox(fig.canvas.get_renderer()).width,
            cbar.get_tightbbox(fig.canvas.get_renderer()).width])
        ax2.set_position(gs[0].get_position(fig))
        ax2.set_subplotspec(gs[0])
        ax1.set_position(gs[1].get_position(fig))
        ax1.set_subplotspec(gs[1])
        cbar.set_position(gs[2].get_position(fig))
        cbar.set_subplotspec(gs[2])

        self.PhaseZoomPlotPath = PlotPathZoom2D /\
            (self.ExpName + '_' + self.ID + '_PhaseZoom.png')
        plt.savefig(self.PhaseZoomPlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info('Zoomed 2D Phase for {} at f = {} Hz plotted'
                           ''.format(self.ExpName, round(self.LockInFreq, 3)))

    def plotAmplFit(self, PlotPath1D):
        """
        Plots the linear amplitude fit into the PlotPath1D folder

        Parameters
        ----------
        PlotPath1D : Path object
            path to the folder for the 1D plots
        """
        # unpack the data
        # This will happen if the evaluation to the top was not done
        if self.AmplFitDataTop is not None:
            EndPosList = self.AmplFitDataTop[2]
        else:
            EndPosList = self.AmplFitDataBot[2]
        Left = EndPosList[0][1]
        Right = EndPosList[1][1]

        plt.figure()
        ax = plt.subplot()

        # create two rectangles showing where data was evaluated
        ax.axvspan(self.LinDist[Left], self.LinDist[Right], fc='lightgray')
        ax.axvspan(-self.LinDist[Left], -self.LinDist[Right], fc='lightgray')

        # plot the raw data
        ax.plot(self.LinDist, self.LinAmplDataTop, '.', markersize=1,
                label='Upper Half')
        ax.plot(-self.LinDist, self.LinAmplDataBot, '.', markersize=1,
                label='Lower Half')

        # plot the linear fit
        X_Line = np.array([self.LinDist[Left], self.LinDist[Right]])
        if self.AmplFitDataTop is not None:
            Y_LineTop = self.AmplFitDataTop[0][0] * X_Line +\
                self.AmplFitDataTop[0][1]
        else:
            Y_LineTop = X_Line.size * [np.nan]
        if self.AmplFitDataBot is not None:
            Y_LineBot = self.AmplFitDataBot[0][0] * X_Line +\
                self.AmplFitDataBot[0][1]
        else:
            Y_LineBot = X_Line.size * [np.nan]

        FitGraph, = ax.plot(X_Line, Y_LineTop, 'r-', lw=2, label='Linear Fit')
        FitGraph, = ax.plot(-X_Line, Y_LineBot, 'r-', lw=2)

        # finalize the graph
        ax.set_xlabel('Distance / mm')
        ax.set_ylabel('Linearized Amplitude')
        ax.set_title('Amplitude, f = ' + str(self.LockInFreq) + ' Hz')
        ax.set_ylim(max([-2, np.nanmin([self.LinAmplDataTop,
                                        self.LinAmplDataBot])]))
        self.AmplFitPlotPath = PlotPath1D /\
            (self.ExpName + '_' + self.ID + '_AmplFit.png')
        plt.savefig(self.AmplFitPlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info('Linear amplitude fit for {}_{} plotted.'
                           ''.format(self.ExpName, self.ID))

    def plotPhaseFit(self, PlotPath1D):
        """
        Plots the linear phase fit into the PlotPath1D folder

        Parameters
        ----------
        PlotPath1D : Path object
            path to the folder for the 1D plots
        """
        # unpack the data
        # This will happen if the evaluation to the top was not done
        if self.PhaseFitDataTop is not None:
            EndPosList = self.PhaseFitDataTop[2]
        else:
            EndPosList = self.PhaseFitDataBot[2]
        Left = EndPosList[0][1]
        Right = EndPosList[1][1]

        plt.figure()
        ax = plt.subplot()

        # create two rectangles showing where data was evaluated
        ax.axvspan(self.LinDist[Left], self.LinDist[Right], fc='lightgray')
        ax.axvspan(-self.LinDist[Left], -self.LinDist[Right], fc='lightgray')

        # plot the raw data
        ax.plot(self.LinDist, self.LinPhaseDataTop, '.', markersize=1,
                label='Upper Half')
        ax.plot(-self.LinDist, self.LinPhaseDataBot, '.', markersize=1,
                label='Lower Half')

        # find tight y-limits for points between -2pi and 2pi
        YMinTop = min([P for P in self.LinPhaseDataTop if -2 * np.pi < P <
                       2 * np.pi])
        YMaxTop = max([P for P in self.LinPhaseDataTop if -2 * np.pi < P <
                       2 * np.pi])
        YMinBot = min([P for P in self.LinPhaseDataBot if -2 * np.pi < P <
                       2 * np.pi])
        YMaxBot = max([P for P in self.LinPhaseDataBot if -2 * np.pi < P <
                       2 * np.pi])
        NewYMin = min(YMinTop, YMinBot)
        NewYMax = max(YMaxTop, YMaxBot)

        # plot the linear fit
        X_Line = np.array([self.LinDist[Left], self.LinDist[Right]])
        if self.PhaseFitDataTop is not None:
            Y_LineTop = self.PhaseFitDataTop[0][0] * X_Line +\
                self.PhaseFitDataTop[0][1]
        else:
            Y_LineTop = X_Line.size * [np.nan]
        if self.PhaseFitDataBot is not None:
            Y_LineBot = self.PhaseFitDataBot[0][0] * X_Line +\
                self.PhaseFitDataBot[0][1]
        else:
            Y_LineBot = X_Line.size * [np.nan]

        FitGraph, = ax.plot(X_Line, Y_LineTop, 'r-', lw=2, label='Linear Fit')
        FitGraph, = ax.plot(-X_Line, Y_LineBot, 'r-', lw=2)

        # finalize the graph
        ax.set_xlabel('Distance / mm')
        ax.set_ylabel('Phase / rad')
        ax.yaxis.set_major_locator(
            MaxNLocator('auto', min_n_ticks=3,
                        steps=[0.5 * np.pi, np.pi, 2 * np.pi,
                               2.5 * np.pi, 3 * np.pi]))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f'{x / np.pi} $\\pi$'))
        ax.set_ylim(NewYMin, NewYMax)
        ax.set_title('Phase, f = ' + str(self.LockInFreq) + ' Hz')
        self.PhaseFitPlotPath = PlotPath1D /\
            (self.ExpName + '_' + self.ID + '_PhaseFit.png')
        plt.savefig(self.PhaseFitPlotPath, dpi=600, bbox_inches='tight')
        plt.close()
        self.Sentinel.info('Linear phase fit for: {}_{} plotted.'
                           ''.format(self.ExpName, self.ID))
