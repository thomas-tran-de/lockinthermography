__version__ = '2020-03-23'


def confMatplotlib(GlobFontSize, CanvasSize, Unit='inches'):
    """
    Configures the rcParameters in Matplotlib.

    Parameters
    ----------
    GlobFontSize : int
        Font size for all text in the plots
    CanvasSize : list of length 2
        Size of the drawing canvas in inches
    Unit : str, optional
        Either 'inches' (default) or 'cm'
    """
    import matplotlib.pyplot as plt

    if Unit == 'cm':
        CanvasSize = (CanvasSize[0] / 2.54, CanvasSize[1] / 2.54)
    elif Unit == 'inches':
        pass
    else:
        raise ValueError('{} is not a valid unit'.format(Unit))

    params = {
        'font.size': GlobFontSize,
        'font.family': 'arial',
        'figure.figsize': CanvasSize
    }
    plt.rcParams.update(params)


def getCustomLogger(ExpName, ExpFolder=None,
                    ConsoleLvl='INFO', FileLvl='DEBUG'):
    """
    Creates a custom logger for use in other modules.
    The logging data is saved to a .txt file related to ExpName

    Parameters
    ----------
    ExpName : str
        Name for the log file.
    ExpFolder : Path object, optional
        Path to the save location. If no path is given, the logfile
        will be saved in the same folder as the script that used
        'getCustomLogger' first.
    ConsoleLvl : str, optional
        The logging level shown in the shell. Case insensitive.
        One of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].
        Default is 'INFO'.
    FileLvl : str, optional
        The logging level shown in the logfile. Case insensitive.
        One of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].
        Default is 'DEBUG'.

    Returns
    -------
    logging.Logger object
        Custom logger
    """
    import logging
    import sys

    # create the logging object, use same ID name in all modules
    Sentinel = logging.getLogger("Watchtower")
    Sentinel.propagate = False  # do not report to root
    Sentinel.setLevel(logging.DEBUG)

    # Direct the logging from the Sentinel to the shell / console
    ShellLog = logging.StreamHandler(sys.stdout)
    ShellLog.setFormatter(logging.Formatter("%(message)s"))
    ShellLog.setLevel(getattr(logging, ConsoleLvl))
    Sentinel.addHandler(ShellLog)

    if ExpFolder is None:
        name = ExpName + '.log'
    else:
        name = ExpFolder / (ExpName + '.log')

    # Define a log file for the Sentinel output
    LogFile = logging.FileHandler(filename=name,
                                  mode="w", encoding="utf-8")
    LogFormatter = logging.Formatter("%(asctime)s : %(levelname)-8s : "
                                     "%(module)-25s : Line %(lineno)-4s : "
                                     "%(funcName)-25s : %(message)s",
                                     "%Y-%m-%d %H:%M:%S")
    LogFile.setFormatter(LogFormatter)
    LogFile.setLevel(getattr(logging, FileLvl))
    Sentinel.addHandler(LogFile)

    return Sentinel
