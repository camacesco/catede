'''
Francesco Camaglia, LPENS July 2021
'''

# coding: utf-8

import pandas
import numpy as np

from kamapack.utils import dict_generator

# LATEX GRAPH DEFAULT: use latex format and fonts
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex = False) # Option 
from matplotlib.colors import is_color_like
_greys_dic_ = {'signal': '#282828', 'pidgeon': '#606e8c', 'silver': '#c0c0c0' }
_kol_dic_ = {'cerulean': '#08457e',  'eggplant': '#991199', 'magenta':  '#FF0066', 
            'mystic-pearl': '#32c6a6', 'flame': '#ff9900', 'azure': '#0064FF', 'chartreuse': '#7fff00'}
_kol_ = [ c for c in _kol_dic_.values() ]

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colors = ['#08457e','#0088ff',"#ef5675",'#ffff66']
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

#########################
#  COMPUTELOCALDENSITY  #
#########################

def computeLocalDensity( x, y, xlim=[0,0], ylim=[0,0], n_bins_x=10, n_bins_y=10):
    '''
    Square Local Density Matrices.
    
    Parameters
    ----------
    
    x: list
    y: list
    xlim: list
    ylim: list
    bins: int, optional
    '''
    
    # >>>>>>>>>>>>>>>>>>
    #  Load parameters
    # >>>>>>>>>>>>>>>>>>
    
    #  x, y
    if len(x) != len(y) :
        raise IOError("The vectors x and y must have the same length.")
    
    #  limits
    for this_lim, this_par_name in zip([xlim,ylim],['xlim','ylim']) :
        if np.all( this_lim == [0,0] ) :
            this_lim = np.array([np.min(x),np.max(x)])
        else :
            try :
                this_lim = list( this_lim )
            except :
                print(f"The provided {this_par_name} type is unrecognized.")
            if len( this_lim ) != 2 :
                raise IOError(f"Parameter {this_par_name} required a list with 2 elements.")
                
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  computing local counts grid
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # get bin size
    Res_x = ( xlim[-1] - xlim[0] ) / n_bins_x
    Res_y = ( ylim[-1] - ylim[0] ) / n_bins_y
    
    # digitize coordinates
    ind_x = np.floor( ( x - xlim[0] ) / Res_x ).astype(int)
    ind_y = np.floor( ( y - ylim[0] ) / Res_y ).astype(int)
    
    # counting the number of cells in each square of the grid
    coordinates = dict_generator( zip( ind_y, ind_x ) )
    
    # loop on all the grid cells
    LocalCounts = np.zeros( ( n_bins_y, n_bins_x ) )
    for c in coordinates:
        i , j = c
        # WARNING!:
        if i < n_bins_x and j < n_bins_y and i > 0 and j > 0 :
            LocalCounts[ i,j ] = coordinates[ c ]
            
    # normalizing
    LocalDensity = LocalCounts / ( len(x) * Res_x * Res_y )

    return LocalDensity
###



#########################
#  DENSITY_SCATTERPLOT  #
#########################

def density_scatterplot( data, bins=None, names=None,  density=True, orientation=None,
                        hist=True, my_cmap=cmap1, bisector=True, density_logscale=True,
                        figsize=None, show_matrix='both' ):
    """
    It produces a scatterplot matrix from a pandas data frame <data>.
    
    Parameters
    ----------
    
    data: pandas.DataFrame
    names: list 
            list of columns of data to be plotted
    """
    
    # >>>>>>>>>>>>>>>>>>
    #  Load parameters
    # >>>>>>>>>>>>>>>>>>
    
    #  names 
    if names == None :
        names = list( data.columns )
    else :
        try :
            names = list( names )
        except :
            print("The provided names type is unrecognized : list is required.")
            
        # check if the names are in the column names
        check = [ n in list( data.columns ) for n in names ]
        if False in check :
            raise KeyError("The provided names do not match with data frame column names: ", names[check])
   
    ncols = len( names ) 
    
    #  figsize
    if figsize is not None :
        if type(figsize) != tuple or len(figsize) != 2 :
            raise IOError( "Wrong choice for `figsize` format." )
    else : # default   
        figsize = ( 2 * ncols, 2 * ncols )
                
    #  bins
    if bins is None :
        # purely empirical choice of the bins
        bins = np.power( 2.8, np.log10( len( data ) ) ).astype(int)
    else : 
        try :
            bins = int( bins )
        except :
            print("The provided bins type is unrecognized : int is required.")
            
    if orientation is not None :
        if not is_color_like(orientation) :
            if orientation != True :
                print('Warning: selected color for `orientation` is not recognized.')
            orientation = _greys_dic_['pidgeon']
            
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  Define subplot dimensions  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
    
    fig, axes = plt.subplots( nrows=ncols, ncols=ncols, 
                             figsize=figsize, sharex=False, sharey=False )
    fig.subplots_adjust( hspace=0.05, wspace=0.05 )

    # adjust xlim for binning
    xlim = np.array([min(np.min(data)), max(np.max(data))])
    xlim[0] *= 1. - 0.00001 * np.sign( xlim[0] ) 
    xlim[1] *= 1. + 0.00001 * np.sign( xlim[1] )     
    ylim = xlim                
    Dxlim = xlim[1] - xlim[0]
    Dylim = ylim[1] - ylim[0]
    
    # smart definition of indices off diagonal for plotting
    if show_matrix == 'lower' :
        smart_indices = list(zip( *np.tril_indices_from(axes, k=-1) ))        
        nega_indices = list(zip( *np.triu_indices_from(axes, k=1) ))
    elif show_matrix == 'upper' :
        smart_indices = list(zip( *np.triu_indices_from(axes, k=1) ))       
        nega_indices = list(zip( *np.tril_indices_from(axes, k=-1) ))
    elif show_matrix == 'both' :
        smart_indices = list(zip( *np.tril_indices_from(axes, k=-1) )) 
        smart_indices += list(zip( *np.triu_indices_from(axes, k=1) ))
        nega_indices = []
    else :
        raise IOError('Unrecognized option for show_matrix.')
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #  pre-process for density scatterplot  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if density is True :
        v_clim = np.zeros(2)
        store_loc_density = {}
        
        for i, j in smart_indices:
            x = data[ names[j] ].values
            y = data[ names[i] ].values
            Loc_density = computeLocalDensity( x, y, xlim=xlim, ylim=ylim, n_bins_x=bins, n_bins_y=bins )  
            if density_logscale is True : Loc_density = safe_log10( Loc_density )
            # update the limits for the values in the colorbar
            v_clim = np.array([min(v_clim[0],np.nanmin(Loc_density)), max(v_clim[1],np.nanmax(Loc_density))])
            store_loc_density[(i,j)] = Loc_density
            
    # >>>>>>>>>>>>>>>>>
    #  Plot the data  #
    # >>>>>>>>>>>>>>>>>
    
    for i, j in smart_indices:
        
        x = data[ names[j] ].values
        y = data[ names[i] ].values
                
        #  SCATTER PLOT  # 
        if density is False :    
            
            axes[i,j].scatter( x, y, color=_kol_dic_['azure'], ec=None, marker='.', alpha=0.2, zorder=0 )                    
            axes[i,j].set_xlim(xlim), axes[i,j].set_ylim(ylim)           
            if bisector is True :
                axes[i,j].plot( xlim, ylim, lw=1, ls="--", color=_greys_dic_['silver'] )
            if orientation is not None :
                lreg = compute_gyration_ax(x, y)
                axes[i,j].plot( xlim, lreg.intercept + lreg.slope * xlim, lw=1, ls="-", color=orientation )
                
        #  DENSITY MAP  #
        if density is True : 
            Loc_density = store_loc_density[(i,j)]
            this_im = axes[i,j].imshow( Loc_density, cmap=my_cmap, vmin=v_clim[0], vmax=v_clim[1],
                                       interpolation='nearest', origin='lower' )
                       
            xImshowLim = np.array([0, Loc_density.shape[0]])
            yImshowLim = np.array([0, Loc_density.shape[1]])
            axes[i,j].set_xlim(xImshowLim), axes[i,j].set_ylim(yImshowLim)
            
            if bisector == True :
                axes[i,j].plot( xImshowLim, yImshowLim, lw=1, ls="--", color=_greys_dic_['silver'] )
                
            if orientation is not None :
                lreg = compute_gyration_ax(x, y)
                lreg_Imshow_slope = lreg.slope * ( yImshowLim[1] * Dxlim ) / ( xImshowLim[1] * Dylim )
                lreg_Imshow_intercept = ( lreg.intercept + lreg.slope * xlim[0] - ylim[0]) * yImshowLim[1] / Dylim
                axes[i,j].plot( xImshowLim, lreg_Imshow_intercept + lreg_Imshow_slope * xImshowLim,
                          lw=1, ls = "-", color=orientation )                         

    # >>>>>>>>>>>>>
    #  x/y ticks  #
    # >>>>>>>>>>>>>
    
    ticks_pos = np.linspace( 0.1, 0.9, 3 )
    
    for ax in axes.flat:            
        
        # modify frame appearence
        [ ax.spines[p].set_linewidth(1) for p in ['right','left', 'top','bottom'] ]

        # y ticks and labels position
        if ax.is_first_col() or ax.is_last_col() :  

            if ax.is_first_col() : ax.yaxis.set_ticks_position('left')
            else : ax.yaxis.set_ticks_position('right')

            ticks_y = ticks_pos * Dylim + ylim[0]

            if density == True : ax.set_yticks( ticks_pos * bins )
            else : ax.set_yticks( ticks_y )

            ax.set_yticklabels( np.round(ticks_y,1) )

        else :      
            plt.setp(ax.get_yticklabels(), visible=False)  
            ax.tick_params(axis='y', which='both', length=0)

        # x ticks and labels position
        if ax.is_first_row() or ax.is_last_row() :

            if ax.is_first_row() : ax.xaxis.set_ticks_position('top') 
            else : ax.xaxis.set_ticks_position('bottom') 

            ticks_x = ticks_pos * Dxlim + xlim[0]

            if density == True : ax.set_xticks( ticks_pos * bins )
            else : ax.set_xticks( ticks_x )

            ax.set_xticklabels( np.round(ticks_x,1) )

        else :
            plt.setp(ax.get_xticklabels(), visible=False)  
            ax.tick_params(axis='x', which='both', length=0)
            
    # >>>>>>>>>>>>>>>
    #  hide matrix  #
    # >>>>>>>>>>>>>>>

    for i,j in nega_indices :
        axes[i,j].axis('off')
        axes[i,j].set_visible(False)
        axes[i,j].set_xlabel('')
        axes[i,j].set_ylabel('')                     
                
    # >>>>>>>>>>>>>>>>>>>
    #  matrix diagonal  #
    # >>>>>>>>>>>>>>>>>>>
    
    for i, label in enumerate(names):  
        #[ axes[i,i].spines[p].set_color('white') for p in ['right','left', 'top','bottom'] ]       
        plt.setp(axes[i,i].spines.values(), visible=False) # set spines not visible
        
        # minimal labels
        axes[0,i].xaxis.set_label_position('top')
        axes[0,i].set_xlabel(label)            
        axes[i,-1].yaxis.set_label_position('right')
        axes[i,-1].set_ylabel(label)            

        # complete labels
        axes[-1,i].xaxis.set_label_position('bottom')
        axes[-1,i].set_xlabel(label)            
        axes[i,0].yaxis.set_label_position('left')
        axes[i,0].set_ylabel(label)    

        plt.setp(axes[i,i].get_xticklabels(), visible=False)  
        axes[i,i].tick_params(axis='x', which='both', length=0)   
        plt.setp(axes[i,i].get_yticklabels(), visible=False)  
        axes[i,i].tick_params(axis='y', which='both', length=0)
        
        if hist == False : #  Labels in the diagonal subplots  #          
            axes[i,i].annotate( label, (0.5, 0.5), xycoords='axes fraction',
                               ha='center', va='center' )
     
        else : #  Plot histograms and set labels  #
            h, b = np.histogram( data[ label ], bins=np.linspace( xlim[0], xlim[-1], bins ) )
            axes[i,i].set_yscale("log")
            axes[i,i].plot( b[:-1] + 0.5 * ( b[1] - b[0] ), h, 
                           color = _kol_dic_['azure'] ) 
            axes[i,i].set_xlim(xlim)
            
    if hist == False :
        for i in [0,-1] :
            axes[i,i].set_xlabel('')
            axes[i,i].set_ylabel('')
        
    if density is True :
        return axes, this_im
    else :
        return axes
###

#########################
#  COMPUTE_GYRATION_AX  #
#########################

from collections import namedtuple

def compute_gyration_ax( x, y ) :
    
    assert len(x) == len(y)
    lreg = namedtuple('intercept', 'slope')
     
    S = len(x)
    # baricenter
    x_A = np.sum(x) / S
    y_A = np.sum(y) / S
    # gyration tensor 
    G_xx = np.sum(np.power(x,2)) / S - x_A**2
    G_yy = np.sum(np.power(y,2)) / S - y_A**2
    G_xy = np.dot(x,y) / S - x_A * y_A
    # main eigenvalue
    mu_p = 0.5 * ( G_xx + G_yy + np.sqrt( ( G_xx - G_yy )**2 + 4 * G_xy**2 ) ) 
    # main eigenvector angle tangent
    tan = ( mu_p - G_xx ) / G_xy
    # assign results
    lreg.intercept = y_A - tan * x_A
    lreg.slope = tan
    return lreg
### 

################
#  SAFE_LOG10  #
################

def safe_log10( myarray ) :
    
    out = np.empty( np.shape( myarray ) )
    out[ : ] = np.nan
    loc = np.where( myarray > 0 )
    out[ loc ] =np.log10( myarray[ loc ] )
    
    return out
###
