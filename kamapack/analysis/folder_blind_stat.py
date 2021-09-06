# coding: utf-8

'''
Francesco Camaglia, June 2020 LPENS
'''

#####################
#  LOADING MODULES  #
#####################

import os, sys
import glob
import pandas as pd
import numpy as np
import optparse
from tqdm import tqdm

sys.path.append( os.path.realpath( __file__ ).split('kamapack')[0]  )
from kamapack.utils import fileScope, tryMakeDir

##########
#  MAIN  #
##########

def main( ) :
    '''
    Averages everything in the folder.
    '''
        
    parser = optparse.OptionParser( conflict_handler="resolve" )

    # >>>>>>>>>>>
    #  OPTIONS  #
    # >>>>>>>>>>>
    parser.add_option( '-B', '--BASKET', action='store', dest = 'BASKET', metavar='path/to/dir', type="string" )
    parser.add_option( '-c', '--index_col', action="store", dest = 'index_col', type=int, default=None )
    parser.add_option( '-h', '--header', action='store', dest='header', default=None  )
    parser.add_option( '-t', '--tag', action="store", dest='tag', default=None )
    parser.add_option( '-s', '--skip', action="store", dest='skiprows', type='int', default=0 )

    # >>>>>>>>>>>>>>>>>>>>>>
    #  OPTIONS ASSIGNMENT  #
    # >>>>>>>>>>>>>>>>>>>>>>

    options, args = parser.parse_args()
    BASKET = options.BASKET
    tag = options.tag
    index_col = options.index_col
    header = options.header
    skiprows = options.skiprows
    
    # >>>>>>>>>>>>>>
    #  INPUT CHECK
    # >>>>>>>>>>>>>>

    if tag is None : inputfileList = glob.glob( f"{BASKET}/*.*" )
    else : inputfileList = glob.glob( f"{BASKET}/*{tag}*.*" )

    inputfileList = [ f for f in inputfileList if "STAT-" not in f ]
    N = len( inputfileList )
    
    #  COMPRESSION & DELIMITER 
    #  infer from extension of first file in list
    in_scope = fileScope( inputfileList[0] )
    delimiter = in_scope.delimiter
    compression = in_scope.compression

    #  OUTPUTFILE  
    PathToOut = f"{BASKET}/STAT"
    tryMakeDir( PathToOut )
    outputfile = f"{PathToOut}/{tag}." + ('.').join( inputfileList[0].split('.')[ 1 : ] )
    devstdfile = f"{PathToOut}/{tag}-devStd." + ('.').join( inputfileList[0].split('.')[ 1 : ] )

    # >>>>>>>>>>>>>
    #  EXECUTION  #
    # >>>>>>>>>>>>>           

    mean_df = pd.read_csv(inputfileList[ 0 ], header=header, sep=delimiter, keep_default_na=False,
                          compression=compression, index_col=index_col, skiprows=skiprows)
    meanPWR2_df = np.power( mean_df, 2 )

    for i in tqdm( range( 1, N ) ) :
        
        df = pd.read_csv( inputfileList[ i ], header=header, sep=delimiter, keep_default_na=False,
                         compression=compression, index_col=index_col, skiprows=skiprows ) 
        mean_df = mean_df.add(df,fill_value=0)
        meanPWR2_df = meanPWR2_df.add(np.power( df, 2 ),fill_value=0)
        
    mean_df /= N # by step normalization
    meanPWR2_df /= N # by step normalization
    ###
    
    # sample standard deviation
    devStd_df = np.sqrt(N*(meanPWR2_df-np.power( mean_df, 2))/(N-1) )

    #  output index  #
    if index_col is None : index = False
    else : index = True     
    #  output header  #
    if header is None : save_header = False
    else : save_header = True

    mean_df.to_csv( outputfile, header = save_header,
                   sep=delimiter, compression=compression, index=index )
    devStd_df.to_csv( devstdfile, header = save_header,
                     sep=delimiter, compression=compression, index=index )

###

################
#  EXECUTABLE  #
################

if __name__ == "__main__" : main( )

