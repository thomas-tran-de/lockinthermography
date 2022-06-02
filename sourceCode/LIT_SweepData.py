import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sourceCode.LIT_SingleFiberMeas import LIT_SingleFiberMeas


class LIT_SweepData():
    """ Class to handle all the data from a LIT sweep """
    Sentinel = logging.getLogger("Watchtower")

    def __init__(self, RawDataFolder, RScale, ExpName, Type):
        """
        Initialize the LIT sweep data class

        Parameters
        ----------
        RawDataFolder : Path object
            Path to the folder with the amplitude and phase images
        RScale : float
            Pixel size of the IR camera in mm / pixel
        ExpName : str
            Name of the experiment, not the single measurement
        Type : str
            Type of single measurement used: "Point", "Line", "Fiber"
        """
        self.RawDataFolder = RawDataFolder
        self.ExpName = ExpName
        self.Type = Type
        self.RScale = RScale
        self.MultiFreq = False
        self.MultiAngle = False
        self.Anisotropic = False

        # lists for the sweep data
        self.FreqList = []
        self.AmplSlopeList = []
        self.PhaseSlopeList = []
        self.SMeasList = []
        self.AlphaSlope = None
        self.FitEllipse = None

        # find all amplitude and phase file pairs in the raw data folder
        self.FilePairs = self.getAmplPhaseFilePairs(self.RawDataFolder)

        # read in all the raw data files
        self.readRawData()

        # determine multi angle and/or frequency
        self.determineMultiType()

        # paths for all the plots
        self.MainFolder = RawDataFolder.parent
        self.MainPlotFolder = self.MainFolder / "Plots"
        self.MainPlotFolder.mkdir(parents=True, exist_ok=True)

        # create a folder for the 1D plots
        self.PlotPath1D = self.MainPlotFolder / "1DPlots"
        self.PlotPath1D.mkdir(parents=True, exist_ok=True)

        # path for the zoomed in 2D amplitude and phase
        self.PlotPathZoom2D = self.MainPlotFolder / "Zoom2DPlots"
        self.PlotPathZoom2D.mkdir(parents=True, exist_ok=True)

        # path for the raw 2D data
        self.PlotPathRaw2D = None

        # path for the numpy data
        self.PlotPathRaw2D = None

        # sweep plot paths
        self.AlphaLinRegPath = None
        self.AlphaFreqPath = None
        self.PolarPlotPath = None
        self.CLinePlotPath = None
        self.AngleTDiffPlotPath = None

    def determineMultiType(self):
        """
        Determines if the raw data set contains multiple
        frequencies and/or multiple angle (line) measurements.

        Raises
        ------
        NotImplementedError
            if both multifreq and multiangle data is
            detected for a line measurement
        """
        # get the frequency list
        FreqList = self.getFreqList()
        # if there is more than one unique frequency
        # dangerous because of implicit float comparison in set()
        if len(set(FreqList)) > 1:
            self.MultiFreq = True
            self.Sentinel.info("This is a multi frequency measurement.")

        # assuming there are no identical measurements (freq & dir)
        elif self.Type == "Line" and len(FreqList) > len(set(FreqList)):
            self.MultiAngle = True
            self.Sentinel.info("This is a multi angle measurment.")
            if self.MultiFreq is True:
                ErrorMsg = "Multi frequency and multi angle is"\
                           "currently not supported for line measurements."
                self.Sentinel.error(ErrorMsg)
                raise NotImplementedError(ErrorMsg)
        else:
            self.MultiFreq = False
            self.Sentinel.info("This is a single frequency measurement.")

    def getFreqList(self):
        """
        Returns a list of the frequencies for each measurement
        """
        return self.FreqList[:]

    def getCenterLineList(self):
        """
        Returns a list with all the center lines
        """
        self.CLineList = []
        for M in self.SMeasList:
            self.CLineList.append(M.CenterLine)
        return self.CLineList

    def getPhaseSlopeList(self):
        """
        Returns two lists (value, std) with the phase slope data
        """
        Slopes = []
        Stds = []
        for M in self.SMeasList:
            if M.Type == "Fiber":
                Slopes.append(M.PhaseFitDataTop[0][0])
                Slopes.append(M.PhaseFitDataBot[0][0])
                Stds.append(M.PhaseFitDataTop[1])
                Stds.append(M.PhaseFitDataBot[1])
            elif self.Anisotropic:
                Slopes.append(M.PhaseFitDataLong[0][0])
                Slopes.append(M.PhaseFitDataShort[0][0])
                Stds.append(M.PhaseFitDataLong[1])
                Stds.append(M.PhaseFitDataShort[1])
            else:
                Slopes.append(M.PhaseFitData[0][0])
                Stds.append(M.PhaseFitData[1])
        return [Slopes, Stds]

    def getAmplSlopeList(self):
        """
        Returns two lists (value, std) with the amplitude slope data
        """
        Slopes = []
        Stds = []
        for M in self.SMeasList:
            if M.Type == "Fiber":
                Slopes.append(M.AmplFitDataTop[0][0])
                Slopes.append(M.AmplFitDataBot[0][0])
                Stds.append(M.AmplFitDataTop[1])
                Stds.append(M.AmplFitDataBot[1])
            elif self.Anisotropic:
                Slopes.append(M.AmplFitDataLong[0][0])
                Slopes.append(M.AmplFitDataShort[0][0])
                Stds.append(M.AmplFitDataLong[1])
                Stds.append(M.AmplFitDataShort[1])
            else:
                Slopes.append(M.AmplFitData[0][0])
                Stds.append(M.AmplFitData[1])
        return [Slopes, Stds]

    def getTDiffList(self):
        """
        Returns two lists with the thermal diffusivity and its error
        """
        TDiff = []
        StdTDiff = []
        for M in self.SMeasList:
            if M.Orientation is None:
                TDiff.append(M.ThermalDiff)
                StdTDiff.append(M.StdThermalDiff)
            else:
                TDiff.append(M.ThermalDiffLong)
                TDiff.append(M.ThermalDiffShort)
                StdTDiff.append(M.StdThermalDiffLong)
                StdTDiff.append(M.StdThermalDiffShort)
        return [np.array(TDiff).flatten(), np.array(StdTDiff).flatten()]

    def getMeasIDs(self):
        """
        Returns a list of all the measurement IDs
        """
        return [Meas.ID for Meas in self.SMeasList]

    def plotAlphaLinReg(self):
        """
        Plots the linear fit for a multi frequency measurement
        """
        [AlphaLinReg, SAlphaLinReg, SlopeProd,
            SProdError, LinearSlope] = self.AlphaSlope

        plt.figure()
        plt.errorbar(self.FreqList, abs(SlopeProd), fmt="o", c="C0",
                     yerr=SProdError, markersize=5, label="Measured")
        plt.plot([0, max(self.FreqList)],
                 [0, max(self.FreqList) * LinearSlope],
                 "k-", label="Linear Fit")
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Slope Product / mm$^{-2}$")
        plt.title("$\\alpha$ = {:.3f} ± {:.3f} mm$^2$ s$^{{-1}}$ "
                  "({:.2f} % rel. std)".format(
                      AlphaLinReg, SAlphaLinReg,
                      SAlphaLinReg / AlphaLinReg * 100))
        plt.legend(loc="upper left", numpoints=1, frameon=False)
        self.AlphaLinRegPath = self.MainFolder / \
            (self.ExpName + "_AlphaSlopeFit.pdf")
        try:
            plt.savefig(self.AlphaLinRegPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the diffusivity fit. Please close file "
                  "{} and press enter.".format(self.AlphaLinRegPath))
            plt.savefig(self.AlphaLinRegPath, bbox_inches="tight")
        plt.close()

    def plotAlphaLinRegAniso(self):
        """
        Plots the linear fit for an anisotropic, multi frequency measurement
        """
        [AlphaLinReg, SAlphaLinReg, SlopeProd,
            SProdError, LinearSlope] = self.AlphaSlope

        plt.figure()
        plt.errorbar(self.FreqList, abs(SlopeProd[0]), fmt="o", c="C0",
                     yerr=SProdError[0], markersize=5, label="Measured")
        plt.errorbar(self.FreqList, abs(SlopeProd[1]), fmt="o", c="C1",
                     yerr=SProdError[1], markersize=5)
        plt.semilogx([0, max(self.FreqList)],
                     [0, max(self.FreqList) * LinearSlope[0]],
                     "--", c="C0", label="Linear Fit")
        plt.semilogx([0, max(self.FreqList)],
                     [0, max(self.FreqList) * LinearSlope[1]],
                     "--", c="C1")
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Slope Product / 1/mm$^2$")
        plt.title("$\\alpha_1$ = {:.3f} ± {:.3f} mm$^2$ s$^{{-1}}$ "
                  "({:.2f} % rel. std)\n"
                  "$\\alpha_2$ = {:.3f} ± {:.3f} mm$^2$ s$^{{-1}}$ "
                  "({:.2f} % rel. std)"
                  "".format(AlphaLinReg[0], SAlphaLinReg[0],
                            SAlphaLinReg[0] / AlphaLinReg[0] * 100,
                            AlphaLinReg[1], SAlphaLinReg[1],
                            SAlphaLinReg[1] / AlphaLinReg[1] * 100))
        plt.legend(loc="upper left", numpoints=1, frameon=False)
        self.AlphaLinRegPath = self.MainFolder / \
            (self.ExpName + "_AlphaSlopeFit.pdf")
        try:
            plt.savefig(self.AlphaLinRegPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the diffusivity fit. Please close file "
                  "{} and press enter.".format(self.AlphaLinRegPath))
            plt.savefig(self.AlphaLinRegPath, bbox_inches="tight")
        plt.close()

    def plotTDiffvsFreq(self):
        """ Plot all thermal diffusivities against the frequencies """
        FreqList = self.getFreqList()
        [TDiff, StdTDiff] = self.getTDiffList()

        # plot the data and the fit
        plt.figure()
        plt.errorbar(FreqList, TDiff, fmt="o", c="C0",
                     yerr=StdTDiff, markersize=5, label="Measured")
        plt.axhline(np.mean(TDiff), c="k", ls="-", label="Mean")
        plt.axhline(np.mean(TDiff) + np.std(TDiff),
                    c="k", ls="--", label="$\\pm \\sigma$")
        plt.axhline(np.mean(TDiff) - np.std(TDiff),
                    c="k", ls="--")
        plt.xscale("log")
        plt.ylim(np.mean(TDiff) - 3 * np.std(TDiff),
                 np.mean(TDiff) + 3 * np.std(TDiff))
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Thermal Diffusivity / mm$^2$/s")
        plt.title("$\\alpha$ = {:.3f} ± {:.3f} mm$^2$ s$^{{-1}}$ "
                  "({:.2f} % rel. std)"
                  "".format(np.mean(TDiff), np.std(TDiff),
                            np.std(TDiff) / np.mean(TDiff) * 100))
        plt.legend(loc="upper left", numpoints=1, frameon=False)
        self.AlphaFreqPath = self.MainFolder / \
            (self.ExpName + "_AlphavsFreq.pdf")
        try:
            plt.savefig(self.AlphaFreqPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the diffusivity plot. Please close file "
                  "{} and press enter.".format(self.AlphaFreqPath))
            plt.savefig(self.AlphaFreqPath, bbox_inches="tight")
        plt.close()

    def plotTDiffvsFreqAniso(self):
        """
        Plot all thermal diffusivities against the frequencies
        for an anisotropic measurement
        """
        FreqList = self.getFreqList()
        [TDiff, StdTDiff] = self.getTDiffList()

        # plot the data and the fit
        plt.figure(figsize=(4.5 * 1.5, 4.5))
        plt.errorbar(FreqList, TDiff[::2], fmt="o", c="C0",
                     yerr=StdTDiff[::2], markersize=5, label="Long")
        plt.errorbar(FreqList, TDiff[1::2], fmt="o", c="C1",
                     yerr=StdTDiff[1::2], markersize=5, label="Short")
        plt.axhline(np.mean(TDiff[::2]), c="C0", ls="-", label="Mean")
        plt.axhline(np.mean(TDiff[::2]) + np.std(TDiff[::2]),
                    c="C0", ls="--", label="± Sigma")
        plt.axhline(np.mean(TDiff[::2]) - np.std(TDiff[::2]),
                    c="C0", ls="--")
        plt.axhline(np.mean(TDiff[1::2]), c="C1", ls="-")
        plt.axhline(np.mean(TDiff[1::2]) + np.std(TDiff[1::2]),
                    c="C1", ls="--")
        plt.axhline(np.mean(TDiff[1::2]) - np.std(TDiff[1::2]),
                    c="C1", ls="--")
        plt.set_xscale("log")
        plt.ylim(np.mean(TDiff[1::2]) - 3 * np.std(TDiff[1::2]),
                 np.mean(TDiff[::2]) + 3 * np.std(TDiff[::2]))
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Thermal Diffusivity / mm$^2$/s")
        plt.title("$\\alpha_1$ = {:.3f} ± {:.3f} mm$^2$ s$^{{-1}}$ "
                  "({:.2f} % rel. std)\n"
                  "$\\alpha_2$ = {:.3f} ± {:.3f} mm$^2$ s$^{{-1}}$ "
                  "({:.2f} % rel. std)"
                  "".format(np.mean(TDiff[1::2]), np.std(TDiff[1::2]),
                            np.std(TDiff[1::2]) / np.mean(TDiff[1::2]) * 100,
                            np.mean(TDiff[::2]), np.std(TDiff[::2]),
                            np.std(TDiff[::2]) / np.mean(TDiff[::2]) * 100))
        plt.legend(loc="upper left", numpoints=1, frameon=False,
                   bbox_to_anchor=(1.02, 1))
        self.AlphaFreqPath = self.MainFolder / \
            (self.ExpName + "_AlphavsFreq.pdf")
        try:
            plt.savefig(self.AlphaFreqPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the diffusivity plot. Please close file "
                  "{} and press enter.".format(self.AlphaFreqPath))
            plt.savefig(self.AlphaFreqPath, bbox_inches="tight")
        plt.close()

    def plotPolarTDiff(self):
        """
        Plot the thermal diffusivity with error bars direction dependent
        in a polar coordinate system.
        """
        [TDiff, StdTDiff] = self.getTDiffList()
        CLineList = self.getCenterLineList()
        ForwardDir = [CL.HeatFlowAngles[0] for CL in CLineList]
        BackwardDir = [CL.HeatFlowAngles[1] for CL in CLineList]

        theta = np.linspace(-np.pi, np.pi, 1000)
        EllipsePlot = self.FitEllipse[0] * self.FitEllipse[1] /\
            (np.sqrt((self.FitEllipse[0] *
                      np.cos(theta - self.FitEllipse[2]))**2 +
                     (self.FitEllipse[1] *
                      np.sin(theta - self.FitEllipse[2]))**2))

        # plot in both directions (Angle, Angle + 180Deg)
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, EllipsePlot, "k--")
        ax.errorbar(ForwardDir, TDiff, yerr=StdTDiff,
                    fmt="o", c="C0", capsize=0, markersize=10)
        ax.errorbar(BackwardDir, TDiff, yerr=StdTDiff,
                    fmt="o", c="C1", capsize=0, markersize=10)
        ax.set_rmax(max(TDiff) * 1.2)
        ax.set_rmin(0)
        ax.set_xticks(np.deg2rad(np.arange(-180, 180, 45)))
        ax.set_thetalim(-np.pi, np.pi)
        ax.grid(True)
        ax.set_title(self.ExpName + " TDiff / mm$^2$ s$^{-1}$", pad=20)
        self.PolarPlotPath = self.MainFolder / \
            (self.ExpName + "_PolarPlot.pdf")
        try:
            plt.savefig(self.PolarPlotPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the polar plot. Please close file "
                  "{} and press enter.".format(self.PolarPlotPath))
            plt.savefig(self.PolarPlotPath, bbox_inches="tight")
        plt.close()

    def plotCenterLines(self):
        """
        Plots all center points and lines together
        to visualize the rotation offset.
        """
        CLineList = self.getCenterLineList()
        CP_X = [CL.CenterPoint[0] * self.RScale for CL in CLineList]
        CP_Y = [CL.CenterPoint[1] * self.RScale for CL in CLineList]
        ArrowLength = 10

        plt.figure()
        plt.plot(CP_X, CP_Y, "ro", label="Centers")
        plt.plot([np.mean(CP_X)], [np.mean(CP_Y)], "bo",
                 label="Rotation Center")
        ax = plt.gca()
        ax.set_aspect("equal")

        # plot all the laser lines with 60 pixel length
        for CL in CLineList:
            LMin = np.array(CL.CenterPoint) + 30 * np.array(CL.DirVector)
            LMax = np.array(CL.CenterPoint) - 30 * np.array(CL.DirVector)
            plt.plot([LMin[0] * self.RScale, LMax[0] * self.RScale],
                     [LMin[1] * self.RScale, LMax[1] * self.RScale],
                     "k-", lw=1)

            EndX = ArrowLength * np.cos(CL.HeatFlowAngles[0])
            EndY = ArrowLength * np.sin(CL.HeatFlowAngles[0])
            ax.arrow(CL.CenterPoint[0] * self.RScale,
                     CL.CenterPoint[1] * self.RScale,
                     EndX * self.RScale, EndY * self.RScale,
                     head_width=2 * self.RScale, head_length=4 * self.RScale,
                     fc="k", ec="k")

            EndX = ArrowLength * np.cos(CL.HeatFlowAngles[1])
            EndY = ArrowLength * np.sin(CL.HeatFlowAngles[1])
            ax.arrow(CL.CenterPoint[0] * self.RScale,
                     CL.CenterPoint[1] * self.RScale,
                     EndX * self.RScale, EndY * self.RScale,
                     head_width=2 * self.RScale, head_length=4 * self.RScale,
                     fc="k", ec="k")

        ax.set_xlabel("x-position / mm")
        ax.set_ylabel("y-position / mm")

        # create a square image around the center point
        YLimits = ax.get_ylim()
        XLimits = ax.get_xlim()
        Extent = max([abs(XLimits[0] - np.mean(CP_X)),
                      abs(XLimits[1] - np.mean(CP_X)),
                      abs(YLimits[0] - np.mean(CP_Y)),
                      abs(YLimits[1] - np.mean(CP_Y))]) * 1.1
        ax.set_xlim([np.mean(CP_X) - Extent,
                     np.mean(CP_X) + Extent])
        ax.set_ylim([np.mean(CP_Y) - Extent,
                     np.mean(CP_Y) + Extent])

        self.CLinePlotPath = self.MainFolder / \
            (self.ExpName + "_CenterLines.pdf")
        try:
            plt.savefig(self.CLinePlotPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the center line plot. Please close file "
                  "{} and press enter.".format(self.CLinePlotPath))
            plt.savefig(self.CLinePlotPath, bbox_inches="tight")
        plt.close()
        return True

    def plotAngleTDiff(self):
        """
        Plots the thermal diffusiviy vs. the angle
        in a cartesian coordinate system
        """
        [TDiff, StdTDiff] = self.getTDiffList()
        CLineList = self.getCenterLineList()
        ForwardDir = [CL.HeatFlowAngles[0] * 180 / np.pi for CL in CLineList]
        BackwardDir = [CL.HeatFlowAngles[1] * 180 / np.pi for CL in CLineList]
        theta = np.linspace(-np.pi, np.pi, 1000)
        EllipsePlot = self.FitEllipse[0] * self.FitEllipse[1] /\
            (np.sqrt((self.FitEllipse[0] *
                      np.cos(theta - self.FitEllipse[2]))**2 +
                     (self.FitEllipse[1] *
                      np.sin(theta - self.FitEllipse[2]))**2))

        plt.plot(theta * 180 / np.pi, EllipsePlot, "k--", label="Ellipse Fit")
        plt.errorbar(ForwardDir, TDiff, yerr=StdTDiff,
                     fmt="o", c="C0", capsize=0,
                     markersize=10, label="Forward")
        plt.errorbar(BackwardDir, TDiff, yerr=StdTDiff,
                     fmt="o", c="C1", capsize=0,
                     markersize=10, label="Backward")
        plt.xlabel("Direction / Deg")
        plt.ylabel("Thermal Diffusivity / mm$^2$ s$^{-1}$")
        plt.ylim(0, max(TDiff) * 1.1)
        plt.xlim(-180, 180)
        plt.legend(loc="lower right")
        self.AngleTDiffPlotPath = self.MainFolder / \
            (self.ExpName + "_AngleTDiff.pdf")
        try:
            plt.savefig(self.AngleTDiffPlotPath, bbox_inches="tight")
        except PermissionError:
            input("I cannot save the diffusivity vs angle plot. Please close "
                  "file {} and press enter.".format(self.AngleTDiffPlotPath))
            plt.savefig(self.AngleTDiffPlotPath, bbox_inches="tight")
        plt.close()

    def plotRawData(self):
        """
        Plots all the single measurements into the
        PlotPathRaw2D folder
        """
        for RawData in self.SMeasList:
            RawData.plotRawAmplitude(self.PlotPathRaw2D)
            RawData.plotRawPhase(self.PlotPathRaw2D)

    def getAmplPhaseFilePairs(self, Folder):
        """
        Extracts all amplitude and phase file pairs from the specified folder.
        Amplitude files have _Amp.txt at the end. Phase files have _Phase.txt
        at the end. Pairs share an identical file name before the _Amp.txt
        or _Phase.txt part.

        Parameters
        ----------
        Folder : Path
            Path to the folder with all raw data files (.txt)

        Raises
        ------
        ValueError if Folder is not a directory

        Returns
        -------
        FilePairs : list
            List of all amplitude and phase file pairs
        """
        if Folder.is_dir() is False:
            ErrorMsg = "The path {} is not a directory."\
                       "".format(Folder.absolute())
            self.Sentinel.error(ErrorMsg)
            raise FileNotFoundError(ErrorMsg)

        FilePairs = []
        # List all files and directories
        for File in Folder.iterdir():
            # if there is an amplitude file
            if File.is_file() and File.stem[-4:] == "_Amp":
                PhaseFile = Folder / (File.stem[:-4] + "_Phase.txt")
                # check if the corresponding phase file exists
                if PhaseFile.is_file() is True:
                    FilePairs.append([File, PhaseFile])

        if FilePairs == []:
            ErrorMsg = "I could not find any raw data in {}. Make sure you "\
                       "used the correct path.".format(Folder)
            self.Sentinel.error(ErrorMsg)
            raise ValueError(ErrorMsg)
        else:
            self.Sentinel.info("Found {} amplitude & phase file pairs."
                               "".format(len(FilePairs)))
        return sorted(FilePairs)

    def readRawData(self):
        """
        Reads in all the amplitude and phase file pairs
        Differentiates between the different measurement types:
        "Point", "Line", "Fiber"

        Raises
        ------
        ValueError if Type is not in ["Point", "Line", "Fiber"]
        """
        # determine the right type of single measurement class
        if self.Type == "Point":
            DataClass = LIT_SinglePointMeas
        elif self.Type == "Line":
            DataClass = LIT_SingleLineMeas
        elif self.Type == "Fiber":
            DataClass = LIT_SingleFiberMeas
        else:
            ErrorMsg = "The Type: {} is not recognized."\
                       "Use one of [Point, Line, Fiber]".format(self.Type)
            self.Sentinel.Error(ErrorMsg)
            raise ValueError(ErrorMsg)

        # read in all file pairs with this measurement class
        self.SMeasList = []   # empty the single measurement list
        for FPair in self.FilePairs:
            self.SMeasList.append(DataClass(FPair[0], FPair[1],
                                            self.RScale, self.ExpName))
        self.Sentinel.info("All raw data files for type {} are loaded."
                           "".format(self.Type))

        self.FreqList = []
        for M in self.SMeasList:
            self.FreqList.append(M.LockInFreq)
            # For fibers, add the frequency twice, because analysis is split
            # in top and bottom. The array lengths of frequency and analysis
            # data must be the same
            if M.Type == "Fiber":
                self.FreqList.append(M.LockInFreq)

    def saveData(self, Plot=True, Numpy=False):
        """
        Saves the evaluation values as a txt file and the raw data as images

        Parameters
        ----------
        Plot : bool, optional
            If True (Default), plot the raw data
        SaveNumpyData : str or bool, default False
            Can be one of ['lin', '2D', 'both'] to save data as numpy files
        """
        self.saveTxtFile()
        if Plot:
            # path for the raw 2D data
            self.PlotPathRaw2D = self.MainPlotFolder / "Raw2DPlots"
            self.PlotPathRaw2D.mkdir(exist_ok=True)
            self.plotRawData()
        if Numpy in ['lin', '2D', 'both']:
            # path for the numpy data
            self.SavePathNumpy = self.MainPlotFolder / "NumpyData"
            self.SavePathNumpy.mkdir(exist_ok=True)
            self.saveNumpyData(Numpy)

    def saveTxtFile(self):
        """
        Saves all the results from the analysis into one txt file.
        """
        FilePath = self.MainFolder / (self.ExpName + "_Results.txt")
        DataFile = open(str(FilePath.absolute()), "w")

        # write the header
        DataFile.write("Experiment Name: " + self.ExpName + "\n")
        DataFile.write("Processed on: {:%d-%m-%Y %H:%M:%S}\n"
                       "".format(datetime.datetime.now()))
        DataFile.write("Measurement type: {}\n".format(self.Type))

        # save the ellipse fit data for anisotropic measurements
        if self.FitEllipse is not None:
            DataFile.write("Major Axis thermal diffusivity (mm^2/s): {:.5f}\n"
                           "".format(self.FitEllipse[0]))
            DataFile.write("Minor Axis thermal diffusivity (mm^2/s): {:.5f}\n"
                           "".format(self.FitEllipse[1]))
            DataFile.write("Direction of the major axis (degree): {:.5f}\n"
                           "".format(np.degrees(self.FitEllipse[2])))

        # if multiple frequencies were measured
        if (self.MultiFreq is True) and (self.Anisotropic is False):
            DataFile.write("Thermal diffusivity from the linear regression: "
                           "{:.5f} +/- {:.5f} mm^2/s\n"
                           "".format(self.AlphaSlope[0], self.AlphaSlope[1]))
        elif (self.MultiFreq is True) and (self.Anisotropic is True):
            DataFile.write("Thermal diffusivity from the linear regression "
                           "for the long axis: {:.5f} +/- {:.5f} mm^2/s\n"
                           "".format(self.AlphaSlope[0][0],
                                     self.AlphaSlope[1][0]))
            DataFile.write("Thermal diffusivity from the linear regression "
                           "for the short axis: {:.5f} +/- {:.5f} mm^2/s\n"
                           "".format(self.AlphaSlope[0][1],
                                     self.AlphaSlope[1][1]))
        else:
            DataFile.write("This is a single frequency measurement.\n")

        # if fibers were measured
        if self.Type == "Fiber":
            DataFile.write("The line width was set to {}, i.e., {} px"
                           " were evaluated.\n".format(self.Width,
                                                       2 * self.Width + 1))

        # write the header
        DataFile.write("#Result Data:\n")

        LabelStr = "FileName\tLockInFrequency / Hz\t"
        if self.Type == "Line" and self.MultiAngle is True:
            LabelStr += "Direction 1 / DEG\tDirection 2 / DEG\t"
        elif self.Type == "Fiber":
            LabelStr += "Analysis Part\t"
        elif self.Type == "Point" and self.Anisotropic:
            LabelStr += "Evaluation Axis\t"
        LabelStr += "TDiff / mm^2/s\tSigma TDiff / mm^2/s\t"\
                    "LeftPhaseBound / mm\tRightPhaseBound / mm\t"\
                    "Phase Slope / rad/mm\tSigma Phase Slope rad/mm\t"\
                    "Phase y-axis / rad\tLeftAmplBound / mm\tRightAmplBound "\
                    "/ mm\tAmpl Slope\tSigma Ampl Slope\tAmpl y-axis\t"\
                    "Max Ampl\n"
        DataFile.write(LabelStr)

        for M in self.SMeasList:
            RString = M.AmplPath.stem[:-4] + "\t"
            RString += str(M.LockInFreq) + "\t"

            if self.Type == "Line" and self.MultiAngle is True:
                RString += str(np.degrees(M.CenterLine.RotAngle) - 90) + "\t"
                RString += str(np.degrees(M.CenterLine.RotAngle) + 90) + "\t"

            if self.Type == "Fiber":
                if not np.isnan(M.ThermalDiff[0]):
                    RString += "Top\t"
                    RString += str(M.ThermalDiff[0]) + "\t"
                    RString += str(M.StdThermalDiff[0]) + "\t"
                    RString += str(M.PhaseFitDataTop[2][0][0]) + "\t"
                    RString += str(M.PhaseFitDataTop[2][1][0]) + "\t"
                    RString += str(M.PhaseFitDataTop[0][0]) + "\t"
                    RString += str(M.PhaseFitDataTop[1]) + "\t"
                    RString += str(M.PhaseFitDataTop[0][1]) + "\t"
                    RString += str(M.AmplFitDataTop[2][0][0]) + "\t"
                    RString += str(M.AmplFitDataTop[2][1][0]) + "\t"
                    RString += str(M.AmplFitDataTop[0][0]) + "\t"
                    RString += str(M.AmplFitDataTop[1]) + "\t"
                    RString += str(M.AmplFitDataTop[0][1]) + "\t"
                    RString += str(np.nanmax(M.LinAmplDataTop)) + "\n"
                    if not np.isnan(M.ThermalDiff[1]):
                        RString += M.AmplPath.stem[:-4] + "\t"
                        RString += str(M.LockInFreq) + "\t"
                if not np.isnan(M.ThermalDiff[1]):
                    RString += "Bottom\t"
                    RString += str(M.ThermalDiff[1]) + "\t"
                    RString += str(M.StdThermalDiff[1]) + "\t"
                    RString += str(M.PhaseFitDataBot[2][0][0]) + "\t"
                    RString += str(M.PhaseFitDataBot[2][1][0]) + "\t"
                    RString += str(M.PhaseFitDataBot[0][0]) + "\t"
                    RString += str(M.PhaseFitDataBot[1]) + "\t"
                    RString += str(M.PhaseFitDataBot[0][1]) + "\t"
                    RString += str(M.AmplFitDataBot[2][0][0]) + "\t"
                    RString += str(M.AmplFitDataBot[2][1][0]) + "\t"
                    RString += str(M.AmplFitDataBot[0][0]) + "\t"
                    RString += str(M.AmplFitDataBot[1]) + "\t"
                    RString += str(M.AmplFitDataBot[0][1]) + "\t"
                    RString += str(np.nanmax(M.LinAmplDataTop)) + "\n"
            elif (self.Type == "Point") and (M.ThermalDiff is None):
                RString += "Long\t"
                RString += str(M.ThermalDiffLong) + "\t"
                RString += str(M.StdThermalDiffLong) + "\t"
                RString += str(M.PhaseFitDataLong[2][0][0]) + "\t"
                RString += str(M.PhaseFitDataLong[2][1][0]) + "\t"
                RString += str(M.PhaseFitDataLong[0][0]) + "\t"
                RString += str(M.PhaseFitDataLong[1]) + "\t"
                RString += str(M.PhaseFitDataLong[0][1]) + "\t"
                RString += str(M.AmplFitDataLong[2][0][0]) + "\t"
                RString += str(M.AmplFitDataLong[2][1][0]) + "\t"
                RString += str(M.AmplFitDataLong[0][0]) + "\t"
                RString += str(M.AmplFitDataLong[1]) + "\t"
                RString += str(M.AmplFitDataLong[0][1]) + "\n"
                RString += M.AmplPath.stem[:-4] + "\t"
                RString += str(M.LockInFreq) + "\t"
                RString += "Short\t"
                RString += str(M.ThermalDiffShort) + "\t"
                RString += str(M.StdThermalDiffShort) + "\t"
                RString += str(M.PhaseFitDataShort[2][0][0]) + "\t"
                RString += str(M.PhaseFitDataShort[2][1][0]) + "\t"
                RString += str(M.PhaseFitDataShort[0][0]) + "\t"
                RString += str(M.PhaseFitDataShort[1]) + "\t"
                RString += str(M.PhaseFitDataShort[0][1]) + "\t"
                RString += str(M.AmplFitDataShort[2][0][0]) + "\t"
                RString += str(M.AmplFitDataShort[2][1][0]) + "\t"
                RString += str(M.AmplFitDataShort[0][0]) + "\t"
                RString += str(M.AmplFitDataShort[1]) + "\t"
                RString += str(M.AmplFitDataShort[0][1]) + "\n"
            else:
                RString += str(M.ThermalDiff) + "\t"
                RString += str(M.StdThermalDiff) + "\t"
                RString += str(M.PhaseFitData[2][0][0]) + "\t"
                RString += str(M.PhaseFitData[2][1][0]) + "\t"
                RString += str(M.PhaseFitData[0][0]) + "\t"
                RString += str(M.PhaseFitData[1]) + "\t"
                RString += str(M.PhaseFitData[0][1]) + "\t"
                RString += str(M.AmplFitData[2][0][0]) + "\t"
                RString += str(M.AmplFitData[2][1][0]) + "\t"
                RString += str(M.AmplFitData[0][0]) + "\t"
                RString += str(M.AmplFitData[1]) + "\t"
                RString += str(M.AmplFitData[0][1]) + "\n"

            DataFile.write(RString)

        DataFile.close()
        self.Sentinel.info("Saved all results to " + str(FilePath))

    def saveNumpyData(self, Type):
        """
        Save linear numpy data for all measurements
        """
        for M in self.SMeasList:
            M.saveNumpyData(self.SavePathNumpy, Type)
