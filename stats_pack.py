#!/usr/bin/python
# vim: set expandtab tabstop=4 shiftwidth=4 autoindent:
#
# File:     stats_pack.py
# Author:   Mark Addinall - May 2018 
#
# Synopsis: This is a new application for Asset-IQ Dealer Network.
#           It replaces an existing PHP Web 1.0 type system that
#           has seved well enough but time for a replacement.
#
#           At the moment we are building a proof of concept for
#           the first month using the following technologies.
#
#           - HTML5
#           - CSS3
#           - REACT
#           - REDUX
#           - ES6
#           - jQuery
#           - Bootstrap
#           - Python
#           - Flask
#           - NPM
#           - Babel
#           - Webpack
#
#           This library will contain our statistical METHODs and possibly MODELS we
#           will use in our new application.  The requests for greater and more integrated
#           statistical processing and reporting has increased over the ALPHA development
#           of our work libraries.
#
#
#           I originall did this library in R, in retrospect, that wasn't a great idea
#           due to:
#               1. scarity of programmers
#               2. the introduction of yet another language into the shop
#
#           I settled on Python, Numpy and Scipy.
#
#           SciPy is a collection of mathematical algorithms and convenience functions 
#           built on the Numpy extension of Python. It adds significant power to the interactive 
#           Python session by providing the user with high-level commands and classes for 
#           manipulating and visualizing data. With SciPy an interactive Python session 
#           becomes a data-processing and system-prototyping environment rivaling systems such as 
#           MATLAB, IDL, Octave, R-Lab, and SciLab.
#
#           The additional benefit of basing SciPy on Python is that this also makes a powerful 
#           programming language available for use in developing sophisticated programs and 
#           specialized applications. Scientific applications using SciPy benefit from the development 
#           of additional modules in numerous niches of the software landscape by developers 
#           across the world. Everything from parallel programming to web and data-base 
#           subroutines and classes have been made available to the Python programmer. 
#           All of this power is available in addition to the mathematical libraries in SciPy.


# virtually ALL of the documentation on line as to how
# to import this stuff is WRONG.  Trial and error gets
# this list that works.... Open source, gotta love the 
# attention to detail....

import sys

import csv
import json
from   PIL                      import Image
import pandas                   as pd
import numpy                    as np
import matplotlib
import matplotlib.pyplot        as plot, mpld3
import statsmodels.tsa.api      as smt
import statsmodels.api          as sm

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn                  as sea
import datetime

from scipy      import constants, interpolate, linalg, odr, stats
from fbprophet  import Prophet


"""

The methods and classes in mpld3 are really very useful when returning
dirferent data shapes back to the caller, be that comman line, PHP
or a JS library.  Unfortunately the Encoder was broken.
This is the fix.
"""

#------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
from mpld3 import _display
_display.NumpyEncoder = NumpyEncoder


#--------------
def tempFile(filetype = '.png'):

    """
    return a filename that should be 99.9999% sure to be unique
    we added 'smartForms on the weekend that will do PDF printing.
    so this function need to know a format
    """

    return("/tmp/" + str(datetime.datetime.now().date()) + '_' + 
                        str(datetime.datetime.now().time()).replace(':', '_') + filetype)


#--------------
def Show(data):
    picture = Image.frombytes('1', data.shape[::-1], np.packbits(data, 1))
    picture.show()                      # temp, for testing
    filename = tempFile()
    picture.save(filename)              # temp, for testing

#----------------------
def readCSVDict(table):
    """
    Deciding where to do the reading and number crunching for this stuff was not easy.
    These functions will initially be used to look at data from the OLD system and present
    the data in a modern UI.  The charting functions on the React client have the
    ABILITY to consume CSV files, however I didn't want to chuck huge CSV file around
    the net when the PLOT functions might only require a small subset of data points.

    Also, most of these functions will be tied to the new database, if and when I 
    can get it finished.  So here is a good place to build functions.

    I initially was doing all this in R but decided that another introduced tool
    would be daft.
    """

    with open(table, 'rb') as csv_file:
        table_reader = csv.DictReader(csv_file)
        row = list(table_reader)
        for row in rows:
            print(json.dumps(row))

    return(rows)

#------------------
def csvToJson(rows):

    """
    The client software (React, ES6, expects JSON data.  The sample ouput from this function is:

        [{
        "username": "lanky",
        "user_id": "4",
        "firstname": "Joan",
        "middlename": "Agetha",
        "lastname": "Lanke",
        "age": "36",
        "usertype": "admin",
        "email": "lanke.joan.wind@gmail.com"
        }, {
        "username": "masp",
        "user_id": "56",
        "firstname": "Mark",
        "middlename": "Patrick",
        "lastname": "Aspir",
        "age": "25",
        "usertype": "member",
        "email": "aspirman@gmail.com"
        }]
    """
    json_array = json.dumps(rows)
    return(json_array)


"""
#----------------------------------------------------------------------------------
# Time series METHODS
#----------------------------------------------------------------------------------
#   A time series is a series of data points indexed (or listed or graphed) 
#   in time order. Most commonly, a time series is a sequence taken at successive 
#   equally spaced points in time. Thus it is a sequence of discrete-time data. 
#
#   A parametric modelling approach to time series analysis makes the fundamental 
#   assumption that the underlying data generating process is a stationary stochastic 
#   process. That is, the process has a certain statistically stable structure which 
#   can be described by using a number of parameters (e.g autoregressive or moving 
#   average model components).
#
"""

#----------------------------------------------
def tsSimple(   data            = None,  
                title           = "Title",
                rows            = 3, 
                columns         = 3, 
                data_window     = 12, 
                plot_me         = True,
                show_me         = True, 
                return_plot     = True, 
                return_html     = False, 
                return_mpld     = False, 
                return_data     = False,
                return_stats    = False):

    """
    The plots of the Autocorrelation function (ACF) and the Partial Autorrelation 
    Function (PACF) are the two main tools to examine the time series dependency 
    structure. The ACF is a function of the time displacement of the time series 
    itself. Informally, it is the similarity between observations as a function 
    of the time lag between them. The PACF is the conditional correlation between 
    two variables under the assumptions that the effects of all previous data_window on 
    the time series are known.

    The tsSimple function will be used to quickly evaluate statistical and 
    distributional phenomena of a given time series process. It includes the ACF, 
    PACF, QQ plot, and a histogram visualization.

    However, most financial time series, e.g. returns on assets, are assumed to stem 
    from a Students t-distribution. From a visual point of view, as the Degrees of Freedom (DoF) 
    of t-distribution increase it approaches the normal distribution (at around DoF=30). 
    The interpretation is, that for low DoF, extreme events are more likely to occur when 
    compared to the normal distribution. This is referred to as leptocurtic behavior, 
    informally fat tails, and highlight the fact that there is more distributional density 
    in the tail areas of a distribution when compared to a normal distribution.

    A more refined set of models will be built on top of these initial approximations.
    In this library.

    This function is used both externally and internally to this library.  That is,
    it can be and IS used by other methods in this library, but can and IS
    instantiated from within our application, and stand alone from the OS 
    command shell.
    """

    answer = []

    # if the time_series is not a Panda series object, coerce it

    if not isinstance(data, pd.Series):
        print("Coercing data")
        data = pd.Series(data)

    # sometimes a caller may only want the raw data
    if (plot_me):
        # initialise the figure and axes
        figure          = plot.figure(figsize=(columns, rows))
        layout          = (3, 2)
        ts_ax           = plot.subplot2grid(layout, (0, 0), colspan = 2)
        acf_ax          = plot.subplot2grid(layout, (1, 0))
        pacf_ax         = plot.subplot2grid(layout, (1, 1))
        qq_ax           = plot.subplot2grid(layout, (2, 0))
        histogram_ax    = plot.subplot2grid(layout, (2, 1))

        # time series first
        data.plot(ax=ts_ax)
        plot.legend(loc='best')
        ts_ax.set_title(title)

        # now the acf and pacf
        smt.graphics.plot_acf(data, lags=data_window, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(data, lags=data_window, ax=pacf_ax, alpha=0.5) 
    
        # qq plot
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('Normal QQ Plot')
    
        # hist plot
        data.plot(ax=histogram_ax, kind='hist', bins=25);
        histogram_ax.set_title('Histogram');
        plot.tight_layout();

        """
        # do we want to send a piccie back?
        # there is a trap for young players here.  If you call
        # plot.show, the plot library dumps to the screen and
        # CLEARS your plotting area to a blank state!
        """

        if (return_plot):
            temp = tempFile()
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])

        if (return_html):
            html = mpld3.fig_to_html(figure)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(figure)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()
    return(answer)


#------------------------------------------------------------------------------------------------------------
def scatterSimple(  data            = None,  
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):
    """
    These functions as a general rule accept some common parameters, being:
    data        - the data read from a CSV file to process.  This can be ASSET, Production or HTML
    title       - page title if required
    rows        - number of rows for the plot grid
    columns     - number of columns for the plot grid
    data_window        - amount of data lag between plots
    plot_me     - claculate the plot in the Python server instance
    show_me     - show the plot in the Python server instance
    return_plot - return the generated plot itself
    return_html - return a formatted block of HTML and Javascript to include into a DOM
    return_data - return the generated DATA only

    This gives these set of METHODS the ability to be viewd on a server, on a client
    with a plotting rendering engine, or on a mobile device with no plotting or
    rendering function.
    """

    answer = []

    # sometimes a caller may only want the raw data
    if (plot_me):
        fig, axes = plot.subplots(ncols=columns, nrows=rows, figsize=(4 * columns, 4 * rows))
        for ax, lag in zip(axes.flat, np.arange(1, data_window + 1, 1)):
            lag_str = 't-{}'.format(lag)
            chart = (pd.concat([data, data.shift(-lag)], axis=1, keys=['y']+[lag_str]).dropna())
    
            # plot data
            chart.plot(ax=ax, kind='scatter', y='y', x=lag_str, color='#65C3D3');
            corr = chart.corr().as_matrix()[0][1]
            ax.set_ylabel('Original');
            ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
            ax.set_aspect('equal');
    
            # top and right spine from plot
            sea.despine();

        # plot puts a LOT of white space around stuff.  Get rid of it. 
        fig.tight_layout()

        """
        # do we want to send a piccie back?
        # there is a trap for young players here.  If you call
        # plot.show, the plot library dumps to the screen and
        # CLEARS your plotting area to a blank state!
        """

        if (return_plot):
            temp = tempFile()
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])
        if (return_data):
            answer.append([{"data": data}])
        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)




#----------------------------------------------------------------------------------------------------------------
def scatterPlotCSV( filename        = None,
                    field           = None,
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):
    """
    this is ONE of the ways of involing the simple time series functions.  Drive it via
    a CSV file of data from SOMEWHERE.
    """

    if (filename):
        data = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)

        manipulated = scatterSimple(data            = data, 
                                    title           = title, 
                                    rows            = rows, 
                                    columns         = columns, 
                                    data_window     = data_window, 
                                    plot_me         = plot_me, 
                                    show_me         = show_me,    
                                    return_plot     = return_plot,    
                                    return_html     = return_html, 
                                    return_data     = return_data,
                                    return_stats    = return_stats)
    else:
        manipulated = "ERROR: No file name given."

    return(manipulated)


#-------------------------------------------
def linePlot(   data            = None,
                field           = "Subject Plotted", 
                title           = "Title of Line Plot",
                rows            = 3, 
                columns         = 3, 
                data_window     = 12, 
                plot_me         = True,
                show_me         = True, 
                return_plot     = True, 
                return_html     = False, 
                return_mpld     = False, 
                return_data     = False,
                return_stats    = False):
    """
    These functions as a general rule accept some common parameters, being:
    data        - the data read from a CSV file to process.  This can be ASSET, Production or HTML
    plot_me     - claculate the plot in the Python server instance
    show_me     - show the plot in the Python server instance
    return_plot - return the generated plot itself
    return_html - return a formatted block of HTML and Javascript to include into a DOM
    return_data - return the generated DATA only

    This gives these set of METHODS the ability to be viewd on a server, on a client
    with a plotting rendering engine, or on a mobile device with no plotting or
    rendering function.
    """

    answer = ['NO DATA SUPPLIED TO linePlot']

    
    if (not data.empty): 
        # sometimes a caller may only want the raw data
        if (plot_me):
    
            # plot data
   
            plot.plot(data)
            plot.title(title)
            plot.ylabel(field)
            plot.xlabel("Time") 

            """
            # the caller wants an IMAGE sent back.  This is good for mobile
            # devices that don't want to transmit large chunks of data or do
            # number crunching on a tichy device.

            # do we want to send a piccie back?
            # there is a trap for young players here.  If you call
            # plot.show, the plot library dumps to the screen and
            # CLEARS your plotting area to a blank state!
            """

            fig = plot.figure()

            if (return_plot):
                temp = tempFile()
                print(json.dumps(temp))
                plot.savefig(temp)
                answer.append([{"plot_url": temp}])

            if (return_html):
                html = mpld3.fig_to_html(fig)
                answer.append([{"html": html}])

            if (return_mpld):
                mpld = mpld3.fig_to_dict(fig)
                answer.append([{"mpld": mpld}])
        
            if (return_data):
                answer.append([{"data": data}])
        
            if (return_stats):
                answer.append([{"stats": data.describe(include='all')}])

            if (return_data):
                answer.append([{"data": data}])

            if (show_me):
                plot.show()

    return(answer)



#-------------------------------------------------------------------------------------------------
def linePlotCSV(filename        = None,
                field           = 1,
                title           = "Title",
                rows            = 3, 
                columns         = 3, 
                data_window     = 1, 
                plot_me         = True,
                show_me         = True, 
                return_plot     = True, 
                return_html     = False, 
                return_mpld     = False, 
                return_data     = False,
                return_stats    = False):
    """
    this is ONE of the ways of involing the simple time series functions.  Drive it via
    a CSV file of data from SOMEWHERE.
    """

    manipulated = 'NO DATA FILE PROVIDED'

    if (filename):
        data = pd.read_csv( filename, 
                            sep         = ',', 
                            header      = 0, 
                            parse_dates = True, 
                            index_col   = 0)

        manipulated = linePlot( data            = data,
                                field           = field,
                                title           = title,
                                rows            = rows, 
                                columns         = columns, 
                                data_window     = data_window, 
                                plot_me         = plot_me, 
                                show_me         = show_me,
                                return_plot     = return_plot,
                                return_html     = return_html,
                                return_mpld     = return_mpld,
                                return_data     = return_data,
                                return_stats    = return_stats )

    return(manipulated)



#-----------------------------------------------------------
def whiteNoiseReferenceProcess( data            = None,
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):

    """
    Errors are serially uncorrelated if they are independent and identically distributed (iid). 
    Serially uncorrelated errors implies that the joint distribution of, say 
    (epsilon_1, epsilon_2) must be a product of the distribution of the individual 
    components f(epsilon_1, epsilon_2) = f(epsilon_1) . f(epsilon_2).

    This is important because if a time series model is appropriate and successful at 
    capturing the underlying process, residuals of the model will be iid and resemble a so 
    called white noise process. Therefore part of time series analysis is simply trying to 
    fit a model to a time series such that the residual series is indistinguishable from white noise.

    A random process x_t is said to be a white noise process if its components each have a probability 
    distribution with result mean, finite variance and are statistically (serially) uncorrelated. 

    # seed the random number generator but we want to be able to generate
    # THE SAME SEQUENCE of psuedo-random number so we can validate results
    # of several runs.  So pick an arbitary see.
    """

    np.random.seed(42)
    size = data
    data = None
    data = np.random.normal(0.0, 1.0, size)

    """
    def tsSimple(   data            = None,  
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):
    """

    manipulated = tsSimple( data            = data, 
                            title           = title, 
                            rows            = rows, 
                            columns         = columns, 
                            data_window     = data_window, 
                            plot_me         = plot_me, 
                            show_me         = show_me, 
                            return_plot     = return_plot, 
                            return_html     = return_html, 
                            return_mpld     = return_mpld, 
                            return_data     = return_data,
                            return_stats    = return_stats)

    return(manipulated)


#------------------------------------------------------------
def randomWalkReferenceProcess (data            = 10,  
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):

    """
    A random walk process if it is of the form
        x_t = x_{t-1} + \epsilon_t

    where \epsilon_t is a white noise process. 
    An important characteristic of a random walk process is that is non-stationary. 
    This means that if a given time series is assumed to be governed by a random walk 
    process it is unpredictable. 


    # seed the random number generator but we want to be able to generate
    # THE SAME SEQUENCE of psuedo-random number so we can validate results
    # of several runs.  So pick an arbitary see.
    """

    np.random.seed(253)
    size    = data
    data    = None
    data    = np.random.normal(0.0, 1.0, size)
    result  = np.zeros_like(data)

    # and do the random walk

    for time in range(size):
        result[time] = result[time - 1] + data[time]

    """
    def tsSimple(   data            = None,  
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):
    """

    manipulated = tsSimple( data            = result, 
                            title           = title, 
                            rows            = rows, 
                            columns         = columns, 
                            data_window     = data_window,  
                            plot_me         = plot_me, 
                            show_me         = show_me, 
                            return_plot     = return_plot, 
                            return_html     = return_html,
                            return_mpld     = return_mpld, 
                            return_data     = return_data,
                            return_stats    = return_stats)

    return(manipulated)


#----------------------------------------------------------------
def autoRegressiveReferenceProcess( data            = 1000,  
                                    title           = "Title",
                                    rows            = 3, 
                                    columns         = 3, 
                                    data_window     = 12, 
                                    plot_me         = True,
                                    show_me         = True, 
                                    return_plot     = True, 
                                    return_html     = False, 
                                    return_mpld     = False, 
                                    return_data     = False,
                                    return_stats    = False):

    """
    As mentioned above the random walk process belongs to a more general 
    group of processes, called autoregressive process of the form

        x_t = mu + sum_{i=1}^p\phi_p x_{t-p} + epsilon_t

    The current observation is a linear combination of past observations. 
    For example an AR(1) time series is one period lagged weighted 
    version of itself and is formulated as

        x_t = mu + phi x_{t-1} + epsilon_{t}

     seed the random number generator but we want to be able to generate
     THE SAME SEQUENCE of psuedo-random number so we can validate results
     of several runs. 
    """

    np.random.seed(4)
    size    = data
    data    = None
    data    = np.random.normal(0.0, 1.0, size)
    result  = np.zeros_like(data)
    phi     = 0.3

    # simulate AR(1)

    for time in range(size):
        result[time] = phi * result[time - 1] + data[time]

    """
    def tsSimple(   data            = None,  
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):
    """

    manipulated = tsSimple( result, 
                            title, 
                            12, 
                            14, 
                            data_window,  
                            plot_me=True, 
                            show_me=True, 
                            return_plot=True, 
                            return_html=False, 
                            return_mpld=False, 
                            return_data=True,
                            return_stats = return_stats)

    return(manipulated)


#---------------------------------------------------------------
def movingAverageReferenceProcess(  data            = None,  
                                    title           = "Title",
                                    rows            = 3, 
                                    columns         = 3, 
                                    data_window     = 12, 
                                    plot_me         = True,
                                    show_me         = True, 
                                    return_plot     = True, 
                                    return_html     = False, 
                                    return_mpld     = False, 
                                    return_data     = False,
                                    return_stats    = False):

    """
    # seed the random number generator but we want to be able to generate
    # THE SAME SEQUENCE of psuedo-random number so we can validate results
    # of several runs. 
    """

    np.random.seed(7)
    size    = data
    data    = None
    data    = np.random.normal(0.0, 1.0, size)
    result  = np.zeros_like(data)
    theta1  = 0.8
    theta2  = -1.4

    # simulate MA(2)

    for time in range(size):
        result[time] = data[time] + theta1 * data[time - 1] + theta2 * data[time - 2]

    """
    def tsSimple(   data            = None,  
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):
    """

    manipulated = tsSimple( result, 
                            title, 
                            12, 
                            14, 
                            data_window,  
                            plot_me=True, 
                            show_me=True, 
                            return_plot=True, 
                            return_html=False, 
                            return_mpld=False, 
                            return_data=True,
                            return_stats = return_stats)

    return(manipulated)



#--------------------------------------------------------------------------
def autoRegressiveProcessCSV(   filename        = None,
                                field           = 1, 
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):
    """
    As mentioned above the random walk process belongs to a more general 
    group of processes, called autoregressive process of the form

        x_t = mu + sum_{i=1}^p\phi_p x_{t-p} + epsilon_t

    The current observation is a linear combination of past observations. 
    For example an AR(1) time series is one period lagged weighted 
    version of itself and is formulated as

        x_t = mu + phi x_{t-1} + epsilon_{t}
    """
    
    manipulated = 'NO DATA FILE PROVIDED'
    if (filename):
        df      = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
        data    = df[field].values
        size    = len(data)
        result  = np.zeros_like(data)
        phi     = 0.3

        # calculate AR(1)

        for time in range(size):
            result[time] = phi * result[time - 1] + data[time]

        """
        def tsSimple(   data            = None,  
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):
        """

        manipulated = tsSimple( result, 
                                title, 
                                12, 
                                14, 
                                data_window,  
                                plot_me=True, 
                                show_me=True, 
                                return_plot=True,  
                                return_html=False,  
                                return_mpld=False, 
                                return_data=True,
                                return_stats=return_stats)

    return(manipulated)


#-------------------------------------------------------------------------
def movingAverageProcessCSV(filename        = None, 
                            title           = "Title",
                            rows            = 3, 
                            columns         = 3, 
                            data_window     = 12, 
                            plot_me         = True,
                            show_me         = True, 
                            return_plot     = True, 
                            return_html     = False, 
                            return_mpld     = False, 
                            return_data     = False,
                            return_stats    = False):

    manipulated = 'NO DATA FILE PROVIDED'

    if (filename):
        df      = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
        data    = df[field].values
        size    = len(data)
        result  = np.zeros_like(data)
        theta1  = 0.8
        theta2  = -1.4

        # calculate MA(2)

        for time in range(size):
            result[time] = data[time] + theta1 * data[time - 1] + theta2 * data[time - 2]

        """
        def tsSimple(   data            = None,  
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):
        """

        manipulated = tsSimple( data            = result, 
                                title           = title, 
                                rows            = 12, 
                                columns         = 14, 
                                data_window     = data_window,  
                                plot_me         = plot_me, 
                                show_me         = show_me, 
                                return_plot     = return_plot, 
                                return_html     = return_html, 
                                return_mpld     = return_mpld, 
                                return_data     = return_data,
                                return_stats    = return_stats)

    return(manipulated)

"""
#------------------------------------------------------------------------------
# now that the basics are working, we can add some more complex statistical
# and visualisation chores.  As usual, getting the simple stuff to work was
# a matter of following the documentation and ignoring the tather lare chunks 
# that were WRONG.
#
# The fundamental idea of time series decomposition (and ultimately time series 
# analysis) is to decompose the original time series (sales, stock market trends, etc.) 
# into several independent components. Typically, business time series are divided into 
# the following four components:
#
# - Trend  overall direction of the series i.e. upwards, downwards etc
# - Seasonality  monthly or yearly patterns
# - Cycle  long-term business cycles, they usually come after 5 or 7 years
# - Noise  irregular remainder left after extraction of all the components
#
# Why bother decomposing the original / actual time series into components? 
# It is much easier to forecast the individual regular patterns produced through 
# decomposition of time series than the actual series. Since stationarity is a vital 
# assumption we need to verify if our time series follows a stationary process or not. 
# We can do so by
#
# - Plotting: review the time series plot of our data and visually check if there are 
#   any obvious trends or seasonality
# - Statistical tests: use statistical tests to check if the expectations of stationarity 
#   are met or have been violated.
#
# There is no such algorithm in statistics where you can ask
#     predict_something(data)
#
# otherwise there would be no data scientists on the market!  We would all be filthy
# rich!  We use ALL of these tests to see if a dataset is a CANDIDATE for prediction.
# There is also the subject of data quality.  Not so much for the Other systems Milk and Transport,
# but the Fastrack Data has LOTS of GAPING holes that need to be cleansed.
#
"""


#----------------------------------------------------------
def plotMovingAveragesComplex(  data            = None, 
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):
    answer = []
    y = data[field]

    figure, axes = plot.subplots(2, 2, sharey=False, sharex=False)
    figure.set_figwidth(14)
    figure.set_figheight(8)

    # move the data to each ax

    for i in range(2):
        if (i == 0):
            axes[i][i].plot(y.index, y, label=label)
            axes[i][i].plot(y.index, y.rolling(window=i+window).mean(), label=title[i])
            axes[i][i].set_xlabel(xlabel)
            axes[i][i].set_ylabel(ylabel)
            axes[i][i].set_title(title[i])
            axes[i][i].legend(loc="best")
        
            axes[i][i+1].plot(y.index, y, label=label)
            axes[i][i+1].plot(y.index, y.rolling(window=i+window).mean(), label=title[i+1])
            axes[i][i+1].set_xlabel(xlabel)
            axes[i][i+1].set_ylabel(ylabel)
            axes[i][i+1].set_title(title[i+1])
            axes[i][i+1].legend(loc="best")
        else:
            axes[i][i-1].plot(y.index, y, label=label)
            axes[i][i-1].plot(y.index, y.rolling(window=i+window).mean(), label=title[i+1])
            axes[i][i-1].set_xlabel(xlabel)
            axes[i][i-1].set_ylabel(ylabel)
            axes[i][i-1].set_title(title[i+1])
            axes[i][i-1].legend(loc="best")
        
            axes[i][i].plot(y.index, y, label=label)
            axes[i][i].plot(y.index, y.rolling(window=i+window).mean(), label=title[i+2])
            axes[i][i].set_xlabel(xlabel)
            axes[i][i].set_ylabel(ylabel)
            axes[i][i].set_title(title[i+2])
            axes[i][i].legend(loc="best")

    plot.tight_layout()

    """
    # do we want to send a piccie back?
    # there is a trap for young players here.  If you call
    # plot.show, the plot library dumps to the screen and
    # CLEARS your plotting area to a blank state!
    """

    fig = plot.figure()

    if (return_plot):
        temp = tempFile()
        print(json.dumps(temp))
        plot.savefig(temp)
        answer.append([{"plot_url": temp}])

    if (return_html):
        html = mpld3.fig_to_html(fig)
        answer.append([{"html": html}])

    if (return_mpld):
        mpld = mpld3.fig_to_dict(plot)
        answer.append([{"mpld": mpld}])

    if (return_data):
        answer.append([{"data": data}])

    if (return_stats):
        answer.append([{"stats": data.describe(include='all')}])

    if (show_me):
        plot.show()

    return(answer)



#----------------------------------------------------------------------------------
def plotSingleRollingAverage(   data            = None, 
                                window          = 12,
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):

    answer = []
    y = data[field]

    rolling_mean        = pd.Series.rolling(y, window).mean()
    rolling_deviation   = pd.Series.rolling(y, window).std()

    plot.plot(y, label="Raw Data")

    # these two are not working as expected
    print(json.dumps(rolling_mean))
    print(json.dumps(rolling_deviation))

    plot.plot(rolling_mean, color='red', label=label + 'Mean.')
    plot.plot(rolling_deviation, color='darkblue', label = label + "Standard Deviation.")
    plot.legend(loc='best')
    plot.title(title)
    plot.tight_layout()

    data = y

    """
    # do we want to send a piccie back?
    # there is a trap for young players here.  If you call
    # plot.show, the plot library dumps to the screen and
    # CLEARS your plotting area to a blank state!
    """

    fig = plot.figure()

    if (return_plot):
        temp = tempFile()
        print(json.dumps(temp))
        plot.savefig(temp)
        answer.append([{"plot_url": temp}])

    if (return_html):
        html = mpld3.fig_to_html(fig)
        answer.append([{"html": html}])

    if (return_mpld):
        mpld = mpld3.fig_to_dict(fig)
        answer.append([{"mpld": mpld}])

    if (return_data):
        answer.append([{"data": data}])

    if (return_stats):
        answer.append([{"stats": data.describe(include='all')}])

    if (show_me):
        plot.show()

    return(answer)

# -------------------------------------
def plotRollingAverage(y, window=12):
    '''
    Plot rolling mean and rolling standard deviation for a given time series and window
    '''
    # calculate moving averages
    rolling_mean = pd.rolling_mean(y, window=window)
    rolling_std = pd.rolling_std(y, window=window)
 
    # plot statistics
    plt.plot(y, label='Original')
    plt.plot(rolling_mean, color='crimson', label='Moving average mean')
    plt.plot(rolling_std, color='darkslateblue', label='Moving average standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    return



#-----------------------------------------------------------
def movingAveragesComplexCSV(   filename        = None, 
                                field           = 1,   
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):


    data          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)

    manipulated = plotMovingAveragesComplex(data            = data, 
                                            field           = field,
                                            title           = title,
                                            rows            = rows, 
                                            columns         = columns, 
                                            data_window     = data_window, 
                                            plot_me         = plot_me,
                                            show_me         = show_me, 
                                            return_plot     = return_plot, 
                                            return_html     = return_html, 
                                            return_mpld     = return_mpld, 
                                            return_data     = return_data,
                                            return_stats    = return_stats)
    return(manipulated)



#--------------------------------------------------------------
def singleRollingAverageCSV(    filename        = None, 
                                field           = 'value',   
                                label           = 'label',
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):

    manipulated = 'NO DATA FILE PROVIDED'

    if (filename):
        data          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)

        manipulated = plotSingleRollingAverage( data            = data,  
                                                field           = field, 
                                                window          = window, 
                                                title           = title,
                                                rows            = rows, 
                                                columns         = columns, 
                                                data_window     = data_window, 
                                                plot_me         = plot_me,
                                                show_me         = show_me, 
                                                return_plot     = return_plot, 
                                                return_html     = return_html, 
                                                return_mpld     = return_mpld, 
                                                return_data     = return_data,
                                                return_stats    = return_stats)
    return(manipulated)



"""
# Stage two finished.  So now we have the simple models along with reference models.  We have more
# complex analysis of moving and rolling statistics.  Now the bit we have been waiting for,
# using all of them to attempt to make PREDICTIONS.  As I am not a "climate scientist",
# but a real one, I must add that predictive statistics is at best and informed guess with
# squiggly lines.
"""
"""
Seasonality
Does the season effect how much milk a herd is giving, how many trips a truck driver can
make or he amount of money a person will spend on a particular asset?

We can check this hypothesis of a seasonal effect by pivoting our data for years and months 
and then check for the distribution of events.
"""


#--------------------------------------------------------------------------------------------------------
def seasonalLinePlot(   data            = None, 
                        label           = 'label',
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):

    answer = []

    # create new columns to DataFrame by extracting a string representing 
    # the time under the control of an explicit format string
    # '%b' extracts the month in locale's abbreviated name from the index
    df['Month'] = df.index.strftime('%b')
    df['Year'] = df.index.year
 
    # create nice axes names
    month_names = pd.date_range(start=start, periods=periods, freq='MS').strftime('%b')
 
    # reshape data using 'Year' as index and 'Month' as column
    df_piv_line = df.pivot(index=index, columns=columns, values=values)
    df_piv_line = df_piv_line.reindex(index=month_names)
 
    # create line plot
    df_piv_line.plot(colormap='jet')
    plt.title('Seasonal Effect per Month', fontsize=24)
    plt.ylabel(label)
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))

    """
    # do we want to send a piccie back?
    # there is a trap for young players here.  If you call
    # plot.show, the plot library dumps to the screen and
    # CLEARS your plotting area to a blank state!
    """

    fig = plot.figure() 

    if (return_plot):
        temp = tempFile()
        print(json.dumps(temp))
        plot.savefig(temp)
        answer.append([{"plot_url": temp}])

    if (return_html):
        html = mpld3.fig_to_html(fig)
        answer.append([{"html": html}])

    if (return_mpld):
        mpld = mpld3.fig_to_dict(fig)
        answer.append([{"mpld": mpld}])

    if (return_data):
        answer.append([{"data": data}])

    if (return_stats):
        answer.append([{"stats": data.describe(include='all')}])

    if (show_me):
        plot.show()

    return(answer)


#---------------------------------------------------------
def seasonalLinePlotCSV(filename        = None, 
                        field           = 'value',   
                        label           = 'label',
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):


    df          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
    manipulated = seasonalLinePlot(df, field, label, title, show, return_plot, return_html, return_mpld, return_data)
    return(manipulated)



#------------------------------------------------------------------------------------------------------
def seasonalBoxPlot(data            = None, 
                    label           = 'label', 
                    title           = "Title",
                    rows            = 3, 
                    columns         = 3, 
                    data_window     = 12, 
                    plot_me         = True,
                    show_me         = True, 
                    return_plot     = True, 
                    return_html     = False, 
                    return_mpld     = False, 
                    return_data     = False,
                    return_stats    = False):

    answer = []

    # create new columns to DataFrame by extracting a string representing 
    # the time under the control of an explicit format string
    # '%b' extracts the month in locale's abbreviated name from the index
    df['Month'] = df.index.strftime('%b')
    df['Year']  = df.index.year
 

    # create new columns to DataFrame by extracting a string representing 
    # the time under the control of an explicit format string
    # '%b' extracts the month in locale's abbreviated name from the index
    df['Month'] = df.index.strftime('%b')
    df['Year']  = df.index.year
 
    # create nice axes names
    month_names = pd.date_range(start='1999-01-01', periods=12, freq='MS').strftime('%b')
 
    # reshape date
    df_piv_box = df.pivot(index='Year', columns='Month', values='n_amounts')
 
    # reindex pivot table with 'month_names'
    df_piv_box = df_piv_box.reindex(columns=month_names)
 
    # create a box plot
    fig, ax = plt.subplots();
    df_piv_box.plot(ax=ax, kind='box');
    ax.set_title('Seasonal Effect per Period', fontsize=24);
    ax.set_xlabel('Month');
    ax.set_ylabel('Count');
    ax.xaxis.set_ticks_position('bottom');
    fig.tight_layout();


    """
    # do we want to send a piccie back?
    # there is a trap for young players here.  If you call
    # plot.show, the plot library dumps to the screen and
    # CLEARS your plotting area to a blank state!
    """

    fig = plot.figure()

    if (return_plot):
        temp = tempFile()
        print(json.dumps(temp))
        plot.savefig(temp)
        answer.append([{"plot_url": temp}])
    
    if (return_html):
        html = mpld3.fig_to_html(fig)
        answer.append([{"html": html}])

    if (return_mpld):
        mpld = mpld3.fig_to_dict(fig)
        answer.append([{"mpld": mpld}])

    if (return_data):
        answer.append([{"data": data}])

    if (return_stats):
        answer.append([{"stats": data.describe(include='all')}])

    if (show_me):
        plot.show()

    return(answer)


#----------------------------------------------------
def seasonalBoxPlotCSV( filename        = None, 
                        field           = 'value',   
                        label           = 'label',
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):

    answer = [{ 'error':'NO DATA FILE PROVIDED'}]

    if (filename):
        df          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
    
        manipulated = seasonalLinePlot( df, 
                                        field, 
                                        label, 
                                        title, 
                                        show, 
                                        return_plot, 
                                        return_html, 
                                        return_mpld, 
                                        return_data)
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)



#------------------------------------------------
def seasonalDecomposeMultiplicative(data = None):
    # multiplicative seasonal decomposition
    if (data):
        decomp = seasonal_decompose(y, model='multiplicative')
        decomp.plot();
        plt.show()


#------------------------------------------
def seasonalDecomposeAdditive(data = None):
    # multiplicative seasonal decomposition
    if (data):
        decomp = seasonal_decompose(y, model='additive')
        decomp.plot();
        plt.show()





#--------------------------------------------------------------------------
def autoRegressiveProcessJSON(  data            = None,  
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):


    """
    As mentioned above the random walk process belongs to a more general 
    group of processes, called autoregressive process of the form

        x_t = mu + sum_{i=1}^p\phi_p x_{t-p} + epsilon_t

    The current observation is a linear combination of past observations. 
    For example an AR(1) time series is one period lagged weighted 
    version of itself and is formulated as

        x_t = mu + phi x_{t-1} + epsilon_{t}
    """

    if (data):
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)




"""
#------------------------------------------------------------------------------
# this area will contain very similar functions to the EXTERNAL definitions,
# however these are designed to be action DIRECTLY from the ASSET-IQ Python
# API as a result of a CLIENT REQUEST from REDUX/React
#
# They all got emptied as I refactored the CSV and test stuff.
# This is complex and having a mistake hiding in a typo is
# common.
#
"""

#--------------------------------------------------------------------------
def autoRegressiveProcessJSON(  data          = None,
                                field           = 1, 
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):
    """
    As mentioned above the random walk process belongs to a more general 
    group of processes, called autoregressive process of the form

        x_t = mu + sum_{i=1}^p\phi_p x_{t-p} + epsilon_t

    The current observation is a linear combination of past observations. 
    For example an AR(1) time series is one period lagged weighted 
    version of itself and is formulated as

        x_t = mu + phi x_{t-1} + epsilon_{t}
    """
    
    manipulated = 'NO DATA OBJECT PROVIDED'
    if (filename):
        df      = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
        data    = df[field].values
        size    = len(data)
        result  = np.zeros_like(data)
        phi     = 0.3

        # calculate AR(1)

        for time in range(size):
            result[time] = phi * result[time - 1] + data[time]


        manipulated = tsSimple( result, 
                                title, 
                                12, 
                                14, 
                                data_window,  
                                plot_me=True, 
                                show_me=True, 
                                return_plot=True,  
                                return_html=False,  
                                return_mpld=False, 
                                return_data=True,
                                return_stats=return_stats)

        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)


#-------------------------------------------------------------------------
def movingAverageProcessJSON(filename        = None, 
                            title           = "Title",
                            rows            = 3, 
                            columns         = 3, 
                            data_window     = 12, 
                            plot_me         = True,
                            show_me         = True, 
                            return_plot     = True, 
                            return_html     = False, 
                            return_mpld     = False, 
                            return_data     = False,
                            return_stats    = False):

    manipulated = 'NO DATA OBJECT PROVIDED'

    if (filename):
        df      = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
        data    = df[field].values
        size    = len(data)
        result  = np.zeros_like(data)
        theta1  = 0.8
        theta2  = -1.4

        # calculate MA(2)

        for time in range(size):
            result[time] = data[time] + theta1 * data[time - 1] + theta2 * data[time - 2]


        manipulated = tsSimple( data            = result, 
                                title           = title, 
                                rows            = 12, 
                                columns         = 14, 
                                data_window     = data_window,  
                                plot_me         = plot_me, 
                                show_me         = show_me, 
                                return_plot     = return_plot, 
                                return_html     = return_html, 
                                return_mpld     = return_mpld, 
                                return_data     = return_data,
                                return_stats    = return_stats)
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)




#-----------------------------------------------------------
def movingAveragesComplexJSON(  data          = None, 
                                field           = 1,   
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):


    data          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)

    if (data):
        manipulated = plotMovingAveragesComplex(data            = data, 
                                                field           = field,
                                                title           = title,
                                                rows            = rows, 
                                                columns         = columns, 
                                                data_window     = data_window, 
                                                plot_me         = plot_me,
                                                show_me         = show_me, 
                                                return_plot     = return_plot, 
                                                return_html     = return_html, 
                                                return_mpld     = return_mpld, 
                                                return_data     = return_data,
                                                return_stats    = return_stats)
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)



#--------------------------------------------------------------
def singleRollingAverageJSON(   data            = None, 
                                field           = 'value',   
                                label           = 'label',
                                title           = "Title",
                                rows            = 3, 
                                columns         = 3, 
                                data_window     = 12, 
                                plot_me         = True,
                                show_me         = True, 
                                return_plot     = True, 
                                return_html     = False, 
                                return_mpld     = False, 
                                return_data     = False,
                                return_stats    = False):

    manipulated = 'NO DATA OBJECT PROVIDED'

    if (filename):
        data          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)

        manipulated = plotSingleRollingAverage( data            = data,  
                                                field           = field, 
                                                window          = window, 
                                                title           = title,
                                                rows            = rows, 
                                                columns         = columns, 
                                                data_window     = data_window, 
                                                plot_me         = plot_me,
                                                show_me         = show_me, 
                                                return_plot     = return_plot, 
                                                return_html     = return_html, 
                                                return_mpld     = return_mpld, 
                                                return_data     = return_data,
                                                return_stats    = return_stats)
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)





#---------------------------------------------------------
def seasonalLinePlotJSON(data           = None, 
                        field           = 'value',   
                        label           = 'label',
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):


    df          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)
    
    if (data):
        manipulated = seasonalLinePlot( df, 
                                        field, 
                                        label, 
                                        title, 
                                        show, 
                                        return_plot, 
                                        return_html, 
                                        return_mpld, 
                                        return_data)
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)



#----------------------------------------------------
def seasonalBoxPlotJSON(data            = None, 
                        field           = 'value',   
                        label           = 'label',
                        title           = "Title",
                        rows            = 3, 
                        columns         = 3, 
                        data_window     = 12, 
                        plot_me         = True,
                        show_me         = True, 
                        return_plot     = True, 
                        return_html     = False, 
                        return_mpld     = False, 
                        return_data     = False,
                        return_stats    = False):

    manipulated = 'NO DATA OBJECT PROVIDED'

    if (filename):
        df          = pd.read_csv(filename, sep=',', header=0, parse_dates=True, index_col=0)

        if (plot_me):
            manipulated = seasonalLinePlot( df, 
                                            field, 
                                            label, 
                                            title, 
                                            show, 
                                            return_plot, 
                                            return_html, 
                                            return_mpld, 
                                            return_data)
        if (return_plot):
            temp = tempFile()
            print(json.dumps(temp))
            plot.savefig(temp)
            answer.append([{"plot_url": temp}])
    
        if (return_html):
            html = mpld3.fig_to_html(fig)
            answer.append([{"html": html}])

        if (return_mpld):
            mpld = mpld3.fig_to_dict(fig)
            answer.append([{"mpld": mpld}])

        if (return_data):
            answer.append([{"data": data}])

        if (return_stats):
            answer.append([{"stats": data.describe(include='all')}])

        if (show_me):
            plot.show()

    return(answer)




"""
ARIMA Forecasting

An autoregressive integrated moving average (ARIMA) model is an generalization of 
an autoregressive moving average (ARMA) model. ARIMA models are applied in some cases where data show evidence of 
non-stationarity, where an initial differencing step (corresponding to the integrated part of the model) 
can be applied one or more times to eliminate the non-stationarity.  Wikipedia

In the base model there are three parameters (p, d, q) that are used to parametrize ARIMA models

- autoregressive component p
- integration component d
- moveing average component q
- Hence, an ARIMA model is denoted as ARIMA(p, d, q) and is defined by

left(1 - sum_{i=1}^p phi_i L^i right) (1 - L)^d y_t = mu + left(1 + sum_{i=1}^q theta_i L^i right) epsilon_t

where L denotes the lag operator. The lag operator performs a lagged transformation of a variable with time index like 

y_t L^d = y_{t-d}. 

Each of these three parts is an effort to make the time series stationary, i. e. make the final residual a white noise pattern.
 define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
 
The grid search found that the best model is a seasonal ARIMA model with 
ARIMA(1, 1, 0)(1, 2, 1)12 for y_test. The SARIMAX model will be trained 
on the training data y_train under the optimal parameter setting obtained 
from the grid search. If the seasonal order is not given, the standard 
ARIMA model will be applied to the input data.


The summary attribute from the SARIMAX output returns a large amount of 
information, however,well focus our attention on the table of coefficients.

"""


#------------------------------
def jarqueBeraTest(res = None):

    """
    The Jarque-Bera test is a goodness-of-fit test of whether the data 
    has the skewness and kurtosis of a normal distribution. 
    The normal distribution has a skew of 0 and a kurtosis of 3.
    """

    if (res):
        res.plot_diagnostics(figsize = (16, 10))
        plt.tight_layout()
        plt.show()

    return(res)



#------------------------------------------------------------------
def getBestSARIMAX( pdq = None, seasonal_pdq = None, order = None):

    best_akalite        =   np.inf
    best_pdq            =   None
    best_seasonal_pdq   =   None
    best_model          =   None
    working_model       =   None
    result              =   None

    """
    To find the best fitting model we iteratively create a SARIMAX model with a given 
    parameter constellation and fit the data to it. For each of these models we compute 
    the Akaike Information Criterion (AIC) and eventually choose the model for which the 
    fitted data results in the lowest AIC.
    """

    bestSARIMAX = []

    if ((pdq) and (seasonal_pdq)):
        for index in pdq:
            for seasonal_index in seasonal_pdq:
                try:
                    working_model = sm.tsa.statespace.SARIMAX(  y_training,
                                                                order                   = index,
                                                                seasonal_order          = seasonal_index,
                                                                enforce_stationarity    = True,
                                                                enforce_invertability   = True)
                    result = working_model.fit()
                    # now this just a bubble sort on the model result!
                    if (reslt.aic < best_akalte):
                        best_akalite        = result.aic
                        best_pdq            = index
                        best_seasonal_pdq   = seasonal_index
                        best_model          = working_model
                except:
                    print('Unexpected error in SARIMAX space')
                    continue

    bestSARIMAX.append[{"bestPDQ": best_pdq}]
    bestSARIMAX.append[{"bestPDQ": best_seasonal_pdq}]
    bestSARIMAX.append[{"bestPDQ": best_akalite}]

    return(bestSARIMAX)

#---------------------------------------------------------
def ARMIMA(data, training_chunk, testing_chunk, ranged=4):

    result = {}
    if (data):

        # first spilt the data into training and testing sets

        y_training = data[:training_chunk]
        y_testing  = data[testing_chunk:]

        """
        To fit the time series data to a seasonal ARIMA model with parameters 
        ARIMA(p, d, q)(P, D, Q)s we need to find the optimal parameter setting 
        that strips away the systematic information and leaves us with white noise 
        residuals. Thus for selecting the appropriate model we will use grid search, 
        i.e. the iterative exploration of all possible parameters constellations.
        """

        # establish p, d, q as the parameters of our model matrix

        p = d = q = range(0, ranged)

        # now build the matrix containing the combinations of triplets
    
        pdq             = list(itertools.product(p, d, q))
        seasonal_pdq    = [(x[0], x[1], x[2], 12) 
            for x in list(itertools.product(p, d, q))]

        # now  generate the best SARIMAX fit 

        best_sarimax = getBestSARIMAX(pdq, seasonal_pdq)

    return(result)

    """
    Now we have contructed our SARIAX we examine its Akaike Information Criterion
    The Akaike information criterion (AIC) is a measure of the relative quality of
    statistical models for a given set of data. Given a collection of models for 
    the data, AIC estimates the quality of each model, relative to each of the other models. 
    Hence, AIC provides a means for model selection.  Wikipedia

    It measures the trade-off between the goodness of fit of the model and the 
    complexity of the model (number of included and estimated aprameters). 
    The AIC is calculated as follows

    AIC = 2k - 2ln(L)

    where k=1 corresponds to the number of estimated parameters in the model and 
    L refers to the maximum value of the likelihood function for the model. 
    It is important to note that the AIC only measures the in-sample model fit of 
    the data given for training.
    """

#----------------------------------------
def printSARIMAXStatistics(result = None):

    print(result.aic)
    print(result.summary())

    return True

#----------------------------------------
def plotSARIMAXDiagnostics(result = None):

    result.plot_diagnostics(figsize = (16, 10))
    plot.tight_layout()
    plot.show()

    return True



#----------------------------------------------
def plotSARIMAXOneStepPredictio(result = None):

    # in-sample-prediction and confidence bounds
    pred = result.get_prediction(start  = pd.to_datetime('1998-12-01'), 
                                 end=pd.to_datetime('2016-12-01'),
                                 dynamic=True)
    pred_ci = pred.conf_int()
 
    # plot in-sample-prediction
    ax = y['2009':].plot(label='Observed',color='#006699');
    pred.predicted_mean.plot(   ax      = ax, 
                                label   = 'One-step Ahead Prediction', 
                                alpha   = 0.7, 
                                color   = '#ff0066');
 
    # draw confidence bound (grey)
    ax.fill_between(pred_ci.index, 
                    pred_ci.iloc[:, 0], 
                    pred_ci.iloc[:, 1], 
                    color   = '#ff0066', 
                    alpha   = 0.25);
 
    # style the plot
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1958-12-01'), y.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    plt.legend(loc='upper left')
    plt.show()

    """
    To quantify the accuracy between model fit and true observations wie 
    will use the mean squared error (MSE). The MSE computes the squared 
    difference between the true and predicted value. The MSE is calculated as follows

    MSE = 1/T sum_{t=1}^T (yhat_t - y_t)^2
    """
    y_hat = pred.predicted_mean
    y_true = y['1958-12-01':]
 
    # compute the mean square error
    mse = ((y_hat - y_true) ** 2).mean()
    print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))

    
    return True



#----------------------------------------------------
def plotSARIMAXNStepsPredictio(result = None, N = 2):

    # in-sample-prediction and confidence bounds
    pred = result.get_prediction(start  = pd.to_datetime('1998-12-01'), 
                                 end=pd.to_datetime('2016-12-01'),
                                 dynamic=True)
    pred_ci = pred.conf_int()
 
    # plot in-sample-prediction
    ax = y['2009':].plot(label='Observed',color='#006699');
    pred.predicted_mean.plot(   ax      = ax, 
                                label   = 'One-step Ahead Prediction', 
                                alpha   = 0.7, 
                                color   = '#ff0066');
 
    # draw confidence bound (grey)
    ax.fill_between(pred_ci.index, 
                    pred_ci.iloc[:, 0], 
                    pred_ci.iloc[:, 1], 
                    color   = '#ff0066', 
                    alpha   = 0.25);
 
    # style the plot
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-12-01'), y.index[-1], alpha=.15, zorder=-1, color='grey');
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    plt.legend(loc='upper left')
    plt.show()

    return True


#--------------------------------------------------
def plotOutofSampleConfidenceBounds(result = None):

    """
    Now to put the model to the real test with a 24-month-head prediction. 
    This requires to pass the argument dynamic=False when using the 
    get_prediction method.
    """

    pred_out = result.get_prediction(start=pd.to_datetime('1958-12-01'), 
                                     end=pd.to_datetime('1960-12-01'), 
                                     dynamic=False, full_results=True)
    pred_out_ci = pred_out.conf_int()
 
    # plot time series and out of sample prediction
    ax = y['1999':].plot(label='Observed', color='#006699')
    pred_out.predicted_mean.plot(ax=ax, label='Out-of-Sample Forecast', color='#ff0066')
    ax.fill_between(pred_out_ci.index,
                    pred_out_ci.iloc[:, 0],
                    pred_out_ci.iloc[:, 1], color='#ff0066', alpha=.25)
    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1958-12-01'), y.index[-1], alpha=.15, zorder=-1, color='grey')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    plt.legend()
    plt.savefig('./tmp/out_of_sample_pred.png')
    plt.show()

    return True



#----------------------------------------------
def arimaJSON(data            = None,  
              title           = "Title",
              rows            = 3, 
              columns         = 3, 
              data_window     = 12, 
              plot_me         = True,
              show_me         = True, 
              return_plot     = True, 
              return_html     = False, 
              return_mpld     = False, 
              return_data     = False,
              return_stats    = False):

    result = {}

    return(result)



#----------------------------------------------
def arimaCSV( filename        = None,  
              title           = "Title",
              rows            = 3, 
              columns         = 3, 
              data_window     = 12, 
              plot_me         = True,
              show_me         = True, 
              return_plot     = True, 
              return_html     = False, 
              return_mpld     = False, 
              return_data     = False,
              return_stats    = False):

    result = {}

    return(result)

#----------------------------------------------------------------------------------------------
# End of the ARIMA stuff
#----------------------------------------------------------------------------------------------

"""
Time Series Forecasting with Prophet

Prophet is a procedure for forecasting time series data. 
It is based on an additive model where non-linear trends are fitted with yearly 
and weekly seasonality, plus holidays. It works best with daily periodicity data 
with at least one year of historical data. 

These features may make it easier to use with Milk runs, whist SARIMAS may be better
suited to ASSET valuation over time(s)

Prophet is robust to missing data, shifts in the trend, and large outliers.

The Model
Forecasting in the domain of Prophet is a curve-fitting task. 
The underlying model has an additive form

y(t) = d(t) + s(t) + h(t) + epsilon_t

where d(t) denotes a trend function modeling non-periodic changes, 
s(t) denotes seasonality modeling periodic changes and h(t) representing 
the effects of holidays. This model assumes time as its only regressor, 
however, linear and non-linear transformations are included if it increases the models fit.

Preparing Data input
The input to Prophet is always a pandas.DataFrame object with two columns and headers: ds and y. 
The ds (datestamp) column must contain a date or datetime object (either is fine). 
The y column must be numeric, and represents the time series to forecast.
"""

#-----------------
def plotProphet():

 
    # create new coumns, specific headers needed for Prophet
    df['ds'] = df['month']
    df['y'] = pd.DataFrame(df['n_values'])
    df.pop('month')
    df.pop('n_values')

    # Prophet assumes an additive model and thus we need to log transform our input data


    # log transform data
    df['y'] = pd.DataFrame(np.log(df['y']))

    # plot data
    ax = df.set_index('ds').plot(color='#006699');
    ax.set_ylabel('values');
    ax.set_xlabel('Date');
    plt.savefig('./img/log_transformed_value.png')
    plt.show()


    # log transform data
    df['y'] = pd.DataFrame(np.log(df['y']))
 
    # plot data
    ax = df.set_index('ds').plot(color='#006699');
    ax.set_ylabel('values');
    ax.set_xlabel('Date');
    plt.savefig('./img/log_transformed_value.png')
    plt.show()


    return True


#---------------------
def plotPhrophetCSV():

    df = pd.read_csv('./data/values.csv', sep=';', header=0, parse_dates=True)
    return True



#---------------------
def plotProphetJSON():

    return True


#---------------------------------------------------------------------------------
# End of Prophet stuff
#---------------------------------------------------------------------------------



"""
Now for a few utility methods.
"""


#----------------
def switch(func):

    """
    some emthod in my madness of last week's work making the argument lists
    the same and wiyth ALL name variables.  This means that for certain 
    functions you can leave out an argument and construct an architecture
    where a function that is called is allocated at runtime.
    To use this both from a Javascript API call, PHP eval and from the command
    line this implementation of swtch has been constructed.
    """

    return {
        'noisetest'         : whiteNoiseReferenceProcess,
        'walktest'          : randomWalkReferenceProcess,
        'regresstest'       : autoRegressiveReferenceProcess,
        'matest'            : movingAverageReferenceProcess,
        'scatcsv'           : scatterPlotCSV,
        'linecsv'           : linePlotCSV,
        'autoregresscsv'    : autoRegressiveProcessCSV,
        'mavcsv'            : movingAverageProcessCSV,
        'mavcompcsv'        : movingAveragesComplexCSV,
        'rollavcsv'         : singleRollingAverageCSV,
        'scatjson'          : scatterPlotJSON,
        'linejson'          : linePlotJSON,
        'autoregressjson'   : autoRegressiveProcessJSON,
        'mavjson'           : movingAverageProcessJSON,
        'mavcompjson'       : movingAveragesComplexJSON,
        'rollavjson'        : singleRollingAverageJSON,
        'arimacsv'          : arimaCSV,
        'arimajson'         : arimaJSON,
        'prophetcsv'        : prophetCSV,
        'prophetjson'       : prophetJSON
        }.get(func, usage)


#-----------------------------------------
def genericStatsCall(process, **kwargs):

    """
    some emthod in my madness of last week's work making the argument lists
    the same and wiyth ALL name variables.  This means that for certain 
    functions you can leave out an argument and construct an architecture
    where a function that is called is allocated at runtime.
    To use this both from a Javascript API call, PHP eval and from the command
    line this implementation of swtch has been constructed.
    
    functions in Python are first order objects so can be passed as arguments.
    in this instance the NAME of the process represented as a string is passed in,
    and switch returns the function OBJECT.

    This needs a LOT of testing!
    """

    func = switch(process)
    func(**kwargs)
    

#-----------
def usage():
    return("Lege textum")




"""
#------------------------------------------------------------------------------
# area for testing as stand alone before we use this lot in a library
#
# again, this is designed to be a library integrated into ASSET-IQ, it
# will also operate as a raw data server to a client plotter OR it can just
# return a generated graphic to a smartphone.  If we are going to do
# something we might as well do it right.
#-------------------------------------------------------------------------------


# This next bit is really slack.  I just need a quick and dirty method of
# testing this stand alone without breaking my regular API which DOES
# load it as a MODULE.


# the first argument to a Python program executed by the command line
# or some exec() type function in another language is ALWAYS the name
# of the running program.  That is argv[0].  So, argv[1] SHOULD contain
# the MODEL of choice to run.

# If you load a MODULE from the command line or the Python interpreter,
# it looks for __name__.  If it is set to __main__ Python assumes you
# want to execute some code on load.
"""

if (__name__ == "__main__"):
    arg = None

    if (len(sys.argv) > 1):
        arg = sys.argv[1]

    """
    # if the MODEL chosen is not one of the REFERENCE set, then we expect a
    # FILENAME as the next argument
    """

    file = None

    if (len(sys.argv) > 2):
        file = sys.argv[2]

    """
    # for the time series analysis, since this is able to be called from the
    # command line or from an exec() process in PHP or similar, we don't have
    """

    column = ''

    if (len(sys.argv) > 3):
        column = sys.argv[3]

    """
    # there is almost NO error trapping here in the main()
    # on purpose.  If you want to drive this from a command shell
    # or from PHP/Ruby/NodeJS/C#/COBOL then you need to be a little
    # careful and confident on HOW to invoke the thing.  REsults
    # are returned in stdout()
    #
    # it was designed this way as the routines are part of my
    # new smartReports suite (similar in construction to smartForms),
    # and the MODELs will be accessed by React/Redux API -> Python API
    # in the traditional manner.
    #
    # being able to run this as an executable covers the use of being
    # able to handle data from the olde Fastrack system, and also to
    # be of some use to the Other systemsXXXXX products.

                These are the functions that will be allowed to be called from the
                COMMAND line (for now).


    def tsSimple
    def scatterSimple
    def scatterPlotCSV
    def linePlot
    def linePlotCSV
    def whiteNoiseReferenceProcess
    def randomWalkReferenceProcess
    def autoRegressiveReferenceProcess
    def movingAverageReferenceProcess
    def autoRegressiveProcessCSV
    def movingAverageProcessCSV
    def plotMovingAveragesComplex
    def plotSingleRollingAverage
    def plotRollingAverage
    def movingAveragesComplexCSV
    def singleRollingAverageCSV
    def seasonalLinePlot
    def seasonalLinePlotCSV
    def seasonalBoxPlot
    def seasonalBoxPlotCSV
    def seasonalDecomposeMultiplicative
    def seasonalDecomposeAdditive
    def jarqueBeraTest
    def getBestSARIMAX
    def ARMIMA
    def printSARIMAXStatistics
    def plotSARIMAXDiagnostics
    def plotSARIMAXOneStepPrediction
    def arimaCSV
    def genericStatsCall


    Regarding: usage()

    I WILL write an instructive usage.  It is just I am changing the argument list ever few hours
    this week and updating the help is getting to be a pain in the arse.
    """

    if (arg == None or arg =='h' or arg == '-h' or arg == '--h' or arg == 'help' or arg == '-help' or arg == '--help'):
        print usage()
    else:
        if (file == None):
            if (arg == 'noisetest'):
                print(whiteNoiseReferenceProcess(   data            = 1000, 
                                                    title           = "White Noise Reference",
                                                    rows            = 12,
                                                    columns         = 14,
                                                    data_window     = 9, 
                                                    plot_me         = True,
                                                    show_me         = True,
                                                    return_plot     = True, 
                                                    return_html     = True, 
                                                    return_mpld     = True, 
                                                    return_data     = True,
                                                    return_stats    = True))
            elif (arg == 'walktest'):
                print(randomWalkReferenceProcess(   data            = 1000, 
                                                    title           = "Random Walk Reference",
                                                    rows            = 12,
                                                    columns         = 14,
                                                    data_window     = 100,  
                                                    plot_me         = True,
                                                    show_me         = True,
                                                    return_plot     = True, 
                                                    return_html     = True, 
                                                    return_mpld     = True, 
                                                    return_data     = True,
                                                    return_stats    = True))
            elif (arg == 'regresstest'):
                print(autoRegressiveReferenceProcess(   data            = 1000, 
                                                        title           = "Autoregression Reference AP(p)",
                                                        rows            = 12,
                                                        columns         = 14,
                                                        data_window     = 30, 
                                                        plot_me         = True,
                                                        show_me         = True,
                                                        return_plot     = True, 
                                                        return_html     = True, 
                                                        return_mpld     = True, 
                                                        return_data     = True,
                                                        return_stats    = True))
            elif (arg == 'matest'):
                print(movingAverageReferenceProcess(    data            = 1000, 
                                                        title           = "The Moving Average Reference MA(q)",
                                                        rows            = 12,
                                                        columns         = 14,
                                                        data_window     = 30, 
                                                        plot_me         = True,
                                                        show_me         = True,
                                                        return_plot     = True, 
                                                        return_html     = True, 
                                                        return_mpld     = True, 
                                                        return_data     = True,
                                                        return_stats    = True))
        else:
            """
            #  All of these CSV routines are for Other systemsInc and the old astrack data.
            # I will duplicate these little "lead-in" routines so data is fed directly
            # from the Python API from the new Asset-IQ database.
            #
            # I am going to sen data bask to PHP in a JSON ARRAY.
            #
            # All of the BOOLEAN arguments are set to TRUE for testing.  This reflects
            # the type and amount of data one gets back from the calling finctions.
            #
            # The ones that NEED a CSV REALLY NEED a CSV file.  Bugger all error checking
            """

            if (arg == 'scatcsv'):
                print(scatterPlotCSV(   filename        = file, 
                                        field           = column,
                                        title           = 'Scatter from CSV File', 
                                        rows            = 3, 
                                        columns         = 3, 
                                        data_window     = 9, 
                                        plot_me         = True, 
                                        show_me         = True, 
                                        return_plot     = True, 
                                        return_html     = True, 
                                        return_data     = True,
                                        return_stats    = True))
            elif (arg == 'lineplotcsv'):
                print(linePlotCSV(  filename        = file,
                                    field           = column, 
                                    title           = 'Line Plot from CSV File', 
                                    rows            = 3, 
                                    columns         = 3, 
                                    data_window     = 1, 
                                    plot_me         = True, 
                                    show_me         = True, 
                                    return_plot     = True, 
                                    return_html     = True, 
                                    return_data     = True,
                                    return_stats    = True))
            elif (arg == 'autoregresscsv'):
                print(autoRegressiveProcessCSV( filename        = file, 
                                                field           = column,
                                                title           = "Autoregression AP(p) CSV", 
                                                rows            = 3,
                                                columns         = 3,
                                                data_window     = 6,
                                                return_plot     = True, 
                                                return_html     = True, 
                                                return_mpld     = True, 
                                                return_data     = True, 
                                                return_stats    = True))
            elif (arg == 'mavcsv'):
                print(movingAverageProcessCSV(  filename        = file, 
                                                field           = column,
                                                title           = "Moving Average from CSV",
                                                rows            = 3, 
                                                columns         = 3, 
                                                data_window     = 12, 
                                                plot_me         = True,
                                                show_me         = True, 
                                                return_plot     = True, 
                                                return_html     = True, 
                                                return_mpld     = True, 
                                                return_data     = True,
                                                return_stats    = True))
            elif (arg == 'mavcompcsv'):
                print(movingAveragesComplexCSV( filename        - file, 
                                                field           = column,
                                                title           = "Complex Moving Average from CSV",
                                                rows            = 3, 
                                                columns         = 3, 
                                                data_window     = 12, 
                                                plot_me         = True,
                                                show_me         = True, 
                                                return_plot     = True, 
                                                return_html     = True, 
                                                return_mpld     = True, 
                                                return_data     = True,
                                                return_stats    = True))
            elif (arg == 'rollavcsv'):
                print(singleRollingAverageCSV(  filename        = file,
                                                field           = column,
                                                title           = "Rolling Average from CSV",
                                                rows            = 3, 
                                                columns         = 3, 
                                                data_window     = 12, 
                                                plot_me         = True,
                                                show_me         = True, 
                                                return_plot     = True, 
                                                return_html     = True, 
                                                return_mpld     = True, 
                                                return_data     = True,
                                                return_stats    = True))

                
                
                #--------------   EOF stat_pack  ------------------


