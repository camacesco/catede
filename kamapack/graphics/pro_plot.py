'''
pro_plot Library:
Francesco Camaglia, LPENS July 2021
'''

import pandas
import numpy as np
from kamapack.graphics import unihists as UH

import matplotlib.pyplot as plt

#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex = False) # Option 

from matplotlib.colors import is_color_like
import matplotlib.colors as colors

_greys_dic_ = {'signal': '#282828', 'pidgeon': '#606e8c', 'silver': '#c0c0c0' }

#################
#  BAR PLOTTER  #
#################

def barPlotter( dataframe, colors,
               columns=None, figsize=None, errorbars=None, rotation=90,
               hatch=None, grid=False, legend=False ) :
    '''
    '''
    
    # >>>>>>>>>>>>>>>>>>>
    #  OPTIONS LOADING  #
    # >>>>>>>>>>>>>>>>>>>
    
    #  dataframe
    df = dataframe.copy()
    
    #  columns
    if columns is not None :
        if np.any( [c not in df.columns for c in columns] ) :
            raise KeyError( "Some requested `columns` are not in `dataframe`." )
    else : # default
        columns = df.columns.values
        
    #  colors 
    # NOTE: a nice way to make this variable optional?
    if not np.all( [is_color_like(c) for c in colors] ) :
        raise IOError( "Some provided `colors` are not recognized." )
    if len(colors) == 1 :
        colors = list(colors) * len(columns)
    elif len(colors) < len(columns) :
        colors = list(colors) * int( 1+np.ceil( len(columns)/len(colors)-1 ) )
        colors = colors[:len(columns)]
                
    #  figsize
    if figsize is not None :
        if type(figsize) != tuple or len(figsize) != 2 :
            raise IOError( "Wrong choice for `figsize` format." )
    else : # default   
        figsize = ( int(0.75*len(df.index)), 4 )
        
    #  errorbars
    if errorbars is not None :
        if np.any( [c not in df.columns for c in errorbars] ) :
            raise KeyError( "Some requested `errorbars` are not in `dataframe`." )
            
    # rotation
    rotation = int(rotation)
    
    # hatch
    if hatch is not None :
        if len(hatch) == 1 :
            hatch = list(hatch) * len(columns)
        elif len(hatch) < len(columns) :
            hatch = list(hatch) * int( 1+np.ceil( len(columns)/len(hatch)-1 ) )
            hatch = hatch[:len(columns)]
    else :
        hatch = [''] * len(columns)
            
    # grid
    grid = bool(grid)
    
    # legend
    legend = bool(legend)
        
        
    # >>>>>>>>>>>>
    #  PLOTTING  #
    # >>>>>>>>>>>>    
    
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=figsize,
                           subplot_kw={'adjustable':'box'} )

    x = np.arange(0.5, len(df.index)+0.5)
    breaks = np.linspace(0, 1, len(columns)+2) - 0.5
    width = breaks[1] - breaks[0]

    ylim_min, ylim_max = [], []
    for i, col in enumerate(columns) :
        ax.bar( x + breaks[i+1], df[col], hatch=hatch[i], edgecolor='white', lw=0.5,
                width=width, color=colors[i], label=col, zorder=1 )  
        if errorbars is not None :
            err_col = errorbars[i]
            ax.errorbar( x + breaks[i+1], df[col], yerr=df[err_col], 
                        ls="", color="black", fmt='', label="")
            ylim_min.append( np.min( df[col] - df[err_col] ) )
            ylim_max.append( np.max( df[col] + df[err_col] ) )
        else :
            ylim_min.append( np.min( df[col] ) )
            ylim_max.append( np.max( df[col] ) )
   
    # lim
    ax.set_xlim( [x[0]-0.75, x[-1]+0.75] )
    ax.set_ylim( [ 0.99 * np.min( ylim_min ), 1.01* np.max( ylim_max ) ] )
    
    # ticks
    ax.set_xticks( x )
    ax.set_xticklabels( df.index, rotation=rotation )

    # grid
    if grid is True :
        ax.yaxis.grid(which="both", color=_greys_dic_["silver"], ls='--', zorder=-10)
        ax.set_axisbelow(True)

    # legend
    
    if legend is True :
        plt.legend(loc="center left", ncol=1, 
                   shadow=False, edgecolor='inherit', framealpha=1, bbox_to_anchor=(1,.5))

    return ax
###



###################
#  CURVE PLOTTER  #
###################

def curvePlotter( dataframe, colors,
               columns=None, figsize=None, errorbars=None, 
               grid=False, legend=False, linestyle="-" ) :
    '''
    '''
    
    # >>>>>>>>>>>>>>>>>>>
    #  OPTIONS LOADING  #
    # >>>>>>>>>>>>>>>>>>>
    
    #  dataframe
    df = dataframe.copy()
    
    #  columns
    if columns is not None :
        if np.any( [c not in df.columns for c in columns] ) :
            raise KeyError( "Some requested `columns` are not in `dataframe`." )
    else : # default
        columns = df.columns.values
        
    #  colors 
    # NOTE: a nice way to make this variable optional?
    if not np.all( [is_color_like(c) for c in colors] ) :
        raise IOError( "Some provided `colors` are not recognized." )
    if len(colors) == 1 :
        colors = list(colors) * len(columns)
    elif len(colors) < len(columns) :
        colors = list(colors) * int( 1+np.ceil( len(columns)/len(colors)-1 ) )
        colors = colors[:len(columns)]
                
    #  figsize
    if figsize is not None :
        if type(figsize) != tuple or len(figsize) != 2 :
            raise IOError( "Wrong choice for `figsize` format." )
    else : # default   
        figsize = ( 6, 4 )
        
    #  linestyle 
    if len(linestyle) == 1 :
        linestyle = list(linestyle) * len(columns)
    elif len(linestyle) < len(columns) :
        linestyle = list(linestyle) * int( 1+np.ceil( len(columns)/len(linestyle)-1 ) )
        linestyle = linestyle[:len(columns)]
        
    #  errorbars
    if errorbars is not None :
        if np.any( [c not in df.columns for c in errorbars] ) :
            raise KeyError( "Some requested `errorbars` are not in `dataframe`." )
                    
    # grid
    grid = bool(grid)
    
    # legend
    legend = bool(legend)
        
        
    # >>>>>>>>>>>>
    #  PLOTTING  #
    # >>>>>>>>>>>>    
    
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=figsize,
                           subplot_kw={'adjustable':'box'} )

    x = df.index.values

    for i, col in enumerate(columns) :
        ax.plot( x, df[col], linestyle=linestyle[i], lw=1,
                color=colors[i], label=col, zorder=1 )  
        if errorbars is not None :
            err_col = errorbars[i]
            ax.errorbar( x, df[col], yerr=df[err_col], 
                        ls="", color=colors[i], fmt='o', label="")

    # grid
    if grid is True :
        ax.yaxis.grid(which="both", color=_greys_dic_["silver"], ls='--', zorder=-10)
        ax.set_axisbelow(True)

    # legend
    
    if legend is True :
        plt.legend(loc="center left", ncol=1, 
                   shadow=False, edgecolor='inherit', framealpha=1, bbox_to_anchor=(1,.5))

    return ax
###