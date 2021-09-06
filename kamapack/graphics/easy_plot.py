'''
Easy_Plot Library:
Francesco Camaglia, LPENS June 2020
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
_kol_dic_ = {'cerulean': '#08457e',  'eggplant': '#991199', 'magenta':  '#FF0066', 
             'chartreuse': '#7fff00', 'flame': '#ff9900', 'mystic-pearl': '#32c6a6', 
             'azure': '#0088ff', 'sulfur': '#ffff66'}
_kol_ = [ c for c in _kol_dic_.values() ]


'''
What follows is kind old and should be checked.
################################################################################################################################
################################################################################################################################
'''

#############
#  DISPLAY  #
#############

def display(album_obj, n_cols=None, xlabel=None, ylabel=None, title=None, xlim=None, color=None, tags=False, fileout=None, **owargs):
       
    '''    
    It plots the n_sets histograms contained in the album_obj object.
    It can be used in overwrite mode (not recommended) if a ndarray of already defined axes is specified amongst the **owargs.

    Parameters:
    -----------
    
    album_obj: Album
            an object belonging to uniHist class.
    n_cols: integer, optional
            the number of multiplot columns.
    xlabel: string, optional
            the label to be put on the visible x axes.
    ylabel: string, optional
            the label to be put on the vsible y axes.
    title: string, optional
            the title to be put on top.
    xlim: list, optional
            a list of two floats fixing the extremes of x interval displayed.
    color: string, optional
            a string defining a color.
    tags: bool, optional
            if True each plot is tagged with a number (1),(2),...; 
            if False (default) the tag is the correspondent histogram header.
    fileout: str, optional
            the path/to/outfile where to save the plot

    Returns: 
    --------
    pandas.DataFrame
    ''' 

    # >>>>>>>>>>>>>>>>
    #  OPTIONS LOAD  #
    # >>>>>>>>>>>>>>>>

    n_sets = album_obj.n_sets    

    # Loading n_cols
    if n_cols == None:         # empirical choice of the number of columns
        if n_sets == 1:
            n_cols = 2
        else:
            n_cols =  2 + int( n_sets / 3 ) + ( n_sets < 4 ) - ( n_sets > 7 )
    else :
        try :
            n_cols = int( n_cols ) 
            if n_cols > (n_sets + 1):
                raise ValueError("n_cols cannot be greater than n_sets + 1.")
        except :
            raise TypeError("n_cols requires an integer.")

    n_rows = int( n_sets / n_cols ) + ( ( n_sets % n_cols ) > 0 )

    # tags assignment
    if tags == False:        
        _text_ = list(album_obj.headers)
    else:
        _text_ = [ "(%d)" % i for i in range(1, n_sets+1) ]

    # bin load and xlim check:
    x = list(album_obj.DataFrame["bins"])
    if xlim == None:
        xlim = [ x[0], x[-1] ]
    elif type(xlim) != list :
        raise TypeError("xlim has to be a list.")
    elif len(xlim) != 2 :
        raise IOError("xlim length must be equal to 2.")
    else:
        xlim = [ min(xlim) , max(xlim) ]    

    # load the plot color
    if color == None:
        _color_ = [ _kol_[ i % len(_kol_) ] for i in range(n_sets) ]
    elif type(color) == str:
        _color_ = [color] * n_sets
    elif type(color) == list:
        if len(color) < n_sets:
            _color_ = color * int( np.ceil( n_sets / len(color) ) )
        else:
            _color_ = color
    else:
        raise ValueError("color type is wrong, provide a single color or a list.")

    # WARNING OVERWRITE TO BE DEVELOPED
    # overwrite mode options (not checked because controlled by the wrapper)
    _par = {}
    for key, value in owargs.items():
        _par[key]=value
    if "_ow_axes_" in _par:
        ax = _par["_ow_axes_"]
        _overwrite = True       
    else:
        _overwrite = False

    # define the index of the subplot
    if (n_rows == 1) or (n_cols == 1):
        ax_index_ = [ i % n_cols for i in range( n_rows * n_cols ) ]
    else:
        ax_index_ = [ ( int( i / n_cols ) , i % n_cols ) for i in range( n_rows * n_cols ) ]
        
    if _overwrite == False:                                     
        fig, ax = plt.subplots(nrows = n_rows, ncols = n_cols, 
                                figsize = ( n_cols * 2.5, n_rows * 2.5 ),
                                gridspec_kw={'height_ratios': [1] * n_rows},
                                subplot_kw={'adjustable':'box'}, 
                                sharex = False, sharey = True
                                )
        fig.subplots_adjust(wspace = 0)                         # width space
        fig.subplots_adjust(hspace = 0)                         # height space
        if title != None:                                       # title creation
            fig.suptitle(title, y = 0.95)
        if xlabel == None:                                      # xlabel default
            xlabel = ""       
        if ylabel == None:                                      # ylabel default
            ylabel = ""         
        
    for i in range(n_sets):
        
        # plots
        y = album_obj.DataFrame[ album_obj.headers[i] ]
        if album_obj.headers[i] + "-err" in album_obj.DataFrame :
            yerr = album_obj.DataFrame[ album_obj.headers[i] + "-err" ]
            _this_Lgnd = ax[ ax_index_[ i ] ].errorbar(
                x, y, yerr=yerr, color = _color_[ i ], mec = _color_[ i ],
                label = "", ls = "-", marker = "s",
                ms = 3, lw = 1, zorder = 2
                )
        else :
            _this_Lgnd, = ax[ ax_index_[ i ] ].plot( x, y, color = _color_[ i ], mec = _color_[ i ], 
                                        label = "", ls = "-", marker = "s", 
                                        ms = 3, lw = 1, zorder = 2
                                        )
        # xlim, text, tags and else
        if _overwrite == False:
            ax[ ax_index_[ i ] ].set_xlim( xlim )


            ax[ ax_index_[ i ] ].text(0.05, 0.95, _text_[ i ], 
                                transform = ax[ ax_index_[ i ] ].transAxes, 
                                verticalalignment='top', 
                                bbox=dict(boxstyle="round", ec= "none", fc = "none", alpha = 0.5)
                                ) 
                                
            if ( (i + n_cols) >= n_sets ):
                ax[ ax_index_[ i ] ].set_xlabel(xlabel)   
            else:
                plt.setp(ax[ ax_index_[ i ] ].get_xticklabels(), visible=False)  
                ax[ ax_index_[ i ] ].tick_params(axis='x', which='both', length=0)

            if ( ( i % n_cols ) == 0 ):
                ax[ ax_index_[ i ]  ].set_ylabel(ylabel)
            else:
                plt.setp(ax[ ax_index_[ i ] ].get_yticklabels(), visible=False)
                ax[ ax_index_[ i ] ].tick_params(axis='y', which='both', length=0)
        
    if _overwrite == False:
        # Clean the empty squares if there are
        for i in range( n_sets, n_cols * n_rows ):    
            plt.setp(ax[ ax_index_[ i ] ].get_xticklabels(), visible=False)
            plt.setp(ax[ ax_index_[ i ] ].get_yticklabels(), visible=False)
            ax[ ax_index_[ i ]  ].tick_params(axis='both', which='both', length=0)
            ax[ ax_index_[ i ] ].set_frame_on(False)

    if ((fileout !=None) and (type(fileout)==str)):
        try:
            open(fileout,"w")
        except IOError:
            print("impossible to open file: ", fileout)
        plt.savefig(fileout, bbox_inches='tight', dpi=300) #WARNING!
        print("Plot saved with success at: ", fileout)
        plt.close()

    return ax, _this_Lgnd      # WARNING: this should be returned only in overwrite mode, but then how to distinguish the first?
###



################################
#  KULLBACK-LEIBER DIVERGENCE  #
################################

def kullback_leiber(album_obj, unit=None ):

    '''
    Parameters
    ----------
    album_obj: Album
            an object belonging to album class.
    unit: str, optional
            the entropy logbase unit:
            - "log": natural logarithm (default);
            - "log2": base 2 logairhtm;
            - "log10":base 10 logarithm.
    Returns:
            np.ndarray
            each element of the matrix i,j contains D_KL( prob_i || prob_j ) where i represents the i-th histogram of album_obj.
    '''

    n_sets = album_obj.n_sets

    # loading units
    unit_Dict = { None: 1., "log": 1., "log2": 1. / np.log(2), "log10": 1. / np.log(10) }
    if unit in unit_Dict.keys( ) :
        unit_conv = unit_Dict[ unit ]
    else:
        raise IOError("Error in function entropy: unknown unit, please choose amongst ", unit_Dict.keys( ) )
    
    # WARNING: protocol (_msk) to handle zeros must be revised
    diver = np.zeros( (n_sets, n_sets) )
    for i in range(n_sets):
        _prob_i = np.array(album_obj.DataFrame[album_obj.headers[i]])
        for j in range(n_sets):
            if i != j:
                _prob_j = np.array(album_obj.DataFrame[album_obj.headers[j]])
                _msk = np.logical_and( _prob_i > 0, _prob_j > 0 )
                diver[i,j] = np.dot( _prob_i[_msk], np.log( _prob_i[_msk] / _prob_j[_msk] ) )
    return diver * unit_conv
###



################
#  DIVERGENCE  #
################

def divergence(album_obj, unit=None, xlabel=None, ylabel=None, title=None, xlim=None, color=None, cmap_bounds=None, tags=False, fileout=None):
       
    '''    
    It plots the n_sets histograms contained in the album_obj object.

    Parameters    
    ----------
    album_obj: Album
            an object belonging to uniHist class.
    xlabel: string, optional
            the label to be put on the visible x axes.
    ylabel: string, optional
            the label to be put on the visible y axes.
    title: string, optional
            the title to be put on top.
    xlim: list, optional
            a list of two floats fixing the extremes of x interval displayed.
    color: string or list, optional
            a string or list defining colors.
    tags: bool, optional
            if True each plot is tagged with a number (1),(2),...; 
            if False (default) the tag is the correspondent histogram header.
    fileout: str, optional
            the path/to/outfile where to save the plot    
    ''' 

    n_sets = album_obj.n_sets

    # tags assignment
    if tags == False:        
        _text_ = list(album_obj.headers)
    else:
        _text_ = [ "(%d)" % i for i in range(1, n_sets+1) ]

    # bin load and xlim check:
    x = list(album_obj.DataFrame["bins"])
    if xlim == None:
        xlim = [ x[0], x[-1] ]
    elif type(xlim) != list :
        raise TypeError('Error in "divergence" function: "xlim" has to be a list.')
    elif len(xlim) != 2 :
        raise IOError('Error in "divergence" function: "xlim" length must be equal to 2.')
    else:
        xlim = [ min(xlim) , max(xlim) ]    

    # load the plot color
    if color == None:
        if 2 * (n_sets-1) < len(_kol_): 
            _color_ = [ _kol_[ 2 * i ] for i in range(n_sets) ]
        else:            
            _color_ = [ _kol_[ i % len(_kol_) ] for i in range(n_sets) ]
    elif type(color) == str:
        _color_ = [color] * n_sets
    elif type(color) == list:
        if len(color) < n_sets:
            _color_ = color * int( np.ceil( n_sets / len(color) ) )
        else:
            _color_ = color
    else:
        raise ValueError('Error in "divergence" function: color type is wrong, provide a single color or a list.')
                                   
    fig, ax = plt.subplots(nrows = 1, ncols = 3, 
                            figsize = ( 6.5, 2.5 ),
                            gridspec_kw={'width_ratios': [3, 0.9, 3.1] },
                            subplot_kw={'adjustable':'box'}, 
                            sharex = False, sharey = False
                            )
    fig.subplots_adjust(wspace = 0.3)                       # width space
    if title != None:                                       # title creation
        fig.suptitle(title, y = 0.95)
    if xlabel == None:                                      # xlabel default
        xlabel = ""       
    if ylabel == None:                                      # ylabel default
        ylabel = ""         
        
    # computing Kullaback Leiber divergence
    diver = kullback_leiber(album_obj, unit=unit)
    # define the cmap bins if not given
    try :
        cmap_bounds = list(cmap_bounds)
    except :
        print("Default colormap bins for Kullback-Leiber is set.")
        cmap_bounds = np.linspace( 0, np.max( diver ), 8 )

    _Lgnd = []
    for i in np.arange(n_sets):
        
        # plots
        y = album_obj.DataFrame[album_obj.headers[ i ]]
        if album_obj.headers[i] + "-err" in album_obj.DataFrame :
            yerr = album_obj.DataFrame[ album_obj.headers[i] + "-err" ]
            _this_Lgnd = ax[ 0 ].errorbar(
                x, y, yerr=yerr, color = _color_[ i ], mec = _color_[ i ],
                label = _text_[ i ], ls = "-", marker = "s",
                ms = 3, lw = 1, zorder = 2
                )
        else :
            _this_Lgnd, = ax[ 0 ].plot( x, y, color = _color_[ i ], mec = _color_[ i ], 
                                        label = _text_[ i ], ls = "-", marker = "s", 
                                        ms = 3, lw = 1, zorder = 2,
                                        )
        _Lgnd.append( _this_Lgnd )

    ax[ 0 ].set_xlim( xlim )                 
    ax[ 0 ].set_xlabel(xlabel)   
    ax[ 0 ].set_ylabel(ylabel)

    _Lgnd.reverse()
    _text_.reverse()
    leg=ax[ 1 ].legend( _Lgnd, _text_, ncol=1, loc="center", frameon=None)   
    leg.get_frame().set_linewidth(0.0)
    plt.setp(ax[ 1 ].get_xticklabels(), visible=False)
    plt.setp(ax[ 1 ].get_yticklabels(), visible=False)
    ax[ 1 ].tick_params(axis='both', which='both', length=0)
    ax[ 1 ].set_frame_on(False)
    _text_.reverse()

    cmap = plt.cm.gray_r  # define the colormap (#cmap = plt.cm.inferno_r)
    cmaplist = [cmap(i) for i in range(cmap.N)]                                                 # extract all colors from the map
    cmap = colors.LinearSegmentedColormap.from_list( 'Kamap', cmaplist, cmap.N )     # create the new map
    norm = colors.BoundaryNorm(cmap_bounds, cmap.N)
    im = ax[ 2 ].imshow( diver , cmap = cmap, norm = norm, 
                    interpolation='nearest', origin='lower')  
    cbar = fig.colorbar(im, spacing='proportional', shrink=0.85, ax= ax[ 2 ])
    cbar.set_label(r'$KL\;[bits]$')
    cbar.ax.set_yticklabels( [ "%.1E" % i for i in cmap_bounds ] )

    ax[ 2 ].set_xticks( range(n_sets),  )
    ax[ 2 ].set_xticklabels( _text_, rotation=90 )

    ax[ 2 ].set_yticks( range(n_sets) )
    plt.setp(ax[ 2 ].get_yticklabels(), visible=False)
    #ax[ 2 ].tick_params(axis='y', which='both', length=0)

    #plt.colorbar(pcm)
        
    if ((fileout !=None) and (type(fileout)==str)):
        try:
            open(fileout,"w")
        except IOError:
            print('Error in "divergence" function: impossible to open file: ', fileout)
        plt.savefig(fileout, bbox_inches='tight', dpi=300) #WARNING!
        print("Plot saved with success at: ", fileout)
        plt.close()

###


###########
#  USAGE  #
###########

def usage( rank_obj, xlabel=None, ylabel=None, title=None, cmap_bounds=None, tags=False, thresh=None, fileout=None ):
    '''
    '''

    # defining matrix of normalized values to be plotted
    diver = ( ( rank_obj.DataFrame.values[:,1:] ).astype(np.float) ).transpose()
    diver *= 100    # percentual conversion

    # adding a threshold on the specimens to display    
    if thresh != None :
        mask_diver = diver < thresh
        temp=[]
        specimen_list=[]
        for i in range( diver.shape[1] ):
            if not ( mask_diver[:,i] == [True] * diver.shape[0] ).all() :
                temp.append(list(diver[:,i]))
                specimen_list.append( rank_obj.specimens[i])
        diver = np.array(temp).transpose()
        cbar_label=r'$\%$'
        cbar_label+=' (thr= %.1E' % (thresh) + ')'
    else :
        specimen_list = list( rank_obj.specimens  )
        cbar_label=r'$\%$' 

    # tags assignment
    if tags == False:        
        _text_ = list(rank_obj.headers)
    else:
        _text_ = [ "(%d)" % i for i in range(1, rank_obj.n_sets+1) ]
                                   
    fig, ax = plt.subplots(figsize = ( len( specimen_list ) * 0.4, rank_obj.n_sets * 0.4 ),
                            subplot_kw={'adjustable':'box'}, 
                            )
    fig.subplots_adjust(wspace = 0.3)                       # width space

    if title != None:                                       # title creation 
        if rank_obj.n_sets < 4 :
            fig.suptitle(title, y = 1)     
        else :
            fig.suptitle(title, y = 0.95)     

    if xlabel == None:                                      # xlabel default
        xlabel = ""      
    ax.set_xlabel(xlabel)   

    if ylabel == None:                                      # ylabel default
        ylabel = ""         
    ax.set_ylabel(ylabel)

    try :                                                   # load cmap bins if not given
        cmap_bounds = list(cmap_bounds)
    except :                                                # default cmap
        print("Default colormap bins is set.")
        if rank_obj.n_sets < 4 :
            cmap_bounds = np.linspace( 0, np.max( diver ), 6 )
        else :
            cmap_bounds = np.linspace( 0, np.max( diver ), 8 )

    cmap = plt.cm.plasma                                                             # define the colormap (#cmap = plt.cm.gray_r)
    cmaplist = [cmap(i) for i in range(cmap.N)]                                      # extract all colors from the map
    cmap = colors.LinearSegmentedColormap.from_list( 'Kamap', cmaplist, cmap.N )     # create the new map
    norm = colors.BoundaryNorm(cmap_bounds, cmap.N)
    im = ax.imshow( diver , cmap = cmap, norm = norm, interpolation='nearest', origin='lower')  
    cbar = fig.colorbar(im, spacing='proportional', shrink=1., ax= ax)
    cbar.set_label( cbar_label )
    cbar.ax.set_yticklabels( [ "%.1E" % i for i in cmap_bounds ] )

    ax.set_yticks( range(rank_obj.n_sets),  )
    ax.set_yticklabels( _text_ )    
    ax.set_xticks( range( len(specimen_list ) ) )
    ax.set_xticklabels( specimen_list, rotation=90 )     
        
    if ((fileout !=None) and (type(fileout)==str)):
        try:
            open(fileout,"w")
        except IOError:
            print('Error in "divergence" function: impossible to open file: ', fileout)
        plt.savefig(fileout, bbox_inches='tight', dpi=300) #WARNING!
        print("Plot saved with success at: ", fileout)
        plt.close()

###


#######################
#  SAMPLES MULTIPLOT  #
#######################

def samples_multiplot( LoadFuncList, bin_size=None, n_sets = None, headers = None, samples = None, color=None,
                   n_cols=None, xlabel=None, ylabel=None, xlim=None, title=None, tags=False, fileout=None, **kwargs): 
    
    '''
    It plots together hisograms for different samples for each set.
    
    bin_size: scalar
            the width of the bin, starting from the minimum value up to the maximum of the all sets.
    LoadFuncList: list of functions
            a list of different functions associated each to a different sample;
            each function must return an array-like variable containing data for the set k;
            its optional arguments can be defined through additional **kwargs.
    n_sets: integer, optional
            the number of different sets tobe loaded.
    headers: list, optional
            a list with the names of the different data sets (default is ["Hist 1", "Hist 2", ...]);
            at least one between n_sets and headers must be specified.
    samples: list, optional
            a list with the names of the different samples (default is ["Sample 1", "Sample 2", ...]);
            samples list must have a length equal to LoadFuncList list.    
    color: string or list, optional
            a string or list defining color(s) of samples.
    n_cols: integer, optional
            the number of multiplot columns.
    xlabel: string, optional
            the label to be put on the visible x axes.
    ylabel: string, optional
            the label to be put on the vsible y axes.
    title: string, optional
            the title to be put on top.
    xlim: list, optional
            a list of two floats fixing the extremes of x interval displayed.
    tags: bool, optional
            if True each plot is tagged with a number (1),(2),...; 
            if False (default) the tag is the correspondent histogram header.
    fileout: str, optional
            the path/to/outfile where to save the plot

    '''       
    
    # Checking the validity of samples in input

    n_samples = len(LoadFuncList)
    if samples == None:
        _Lgnd_names = ["Sample %d" % i for i in range(1, n_samples+1)] 
    elif type(samples)!=list:
        raise TypeError("samples option requires a list.")
    else: 
        if ( len(samples) != n_samples ):
            raise ValueError("samples list must have a length equal to LoadFuncList list.")    
        else:
            _Lgnd_names = samples    

    # default choice of colors

    if color == None:
        if 2 * (n_samples-1) < len(_kol_): 
            _color_ = [ _kol_[ 2 * i ] for i in range(n_samples) ]
        else:            
            _color_ = [ _kol_[ i % len(_kol_) ] for i in range(n_samples) ]
    elif type(color) == str:
        _color_ = [color] * n_samples
    elif type(color) == list:
        if len(color) < n_samples:
            _color_ = color * int( np.ceil( n_samples / len(color) ) )
        else:
            _color_ = color
    else:
        raise ValueError('Color type is wrong, provide a single color or a list.')


    
    _Lgnd_list = []    


    # WARNING!: here to modify for generalization!
    # load the first album and use it as background
    obj = UH.Hists(LoadFuncList[0], bin_size=bin_size, n_sets=n_sets, headers=headers, **kwargs ) 

    axes0, this_Lgnd = obj.display(
        n_cols=n_cols, xlabel=xlabel, ylabel=ylabel, title=title, 
        xlim=xlim, color=_color_[0], tags=tags 
        )
    _Lgnd_list.append(this_Lgnd)
    

    for k in range( 1, n_samples ):
        # WARNING!: here to modify for generalization!
        objk = UH.Hists( LoadFuncList[k], bin_size = bin_size, n_sets=n_sets, headers=headers, **kwargs )
        _trash, _this_Lgnd = objk.display(n_cols=n_cols, color=_color_[k], _ow_axes_=axes0 )
        _Lgnd_list.append( _this_Lgnd )
    
    try:
        axes0[ -1 , -1 ].legend( _Lgnd_list, _Lgnd_names, ncol = 1, loc = "lower right")
    except:
        axes0[ -1 ].legend( _Lgnd_list, _Lgnd_names, ncol = 1, loc = "lower right")

    if ((fileout !=None) and (type(fileout)==str)):
        try:
            open(fileout,"w")
        except IOError:
            print("impossible to open file: ", fileout)

        plt.savefig(fileout, bbox_inches='tight', dpi=300) #WARNING!
        print("Plot saved with success at: ", fileout)
        plt.close()
###


