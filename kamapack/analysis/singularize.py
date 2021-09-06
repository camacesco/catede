# coding: utf-8

'''
Francesco Camaglia, June 2020 LPENS
'''

#####################
#  LOADING MODULES  #
#####################

import os, sys
import pandas as pd
import numpy as np
import optparse

sys.path.append( os.path.realpath( __file__ ).split('kamapack')[0]  )
from kamapack.utils import tryMakeDir, progress_bar, fileScope

##########
#  MAIN  #
##########

def main(  inputfile, delimiter, size, outputfile ) :

    """
    Drop duplicate lines of header-less file <inputfile>.
    """

    # >>>>>>>>>>>>>>
    #  INPUT CHECK
    # >>>>>>>>>>>>>>

    #  INPUTFILE & DELIMITER 
    #  infer compression from extension
    in_scope = fileScope( inputfile )
    if delimiter is None: 
        #  infer delimiter from extension
        delimiter = in_scope.delimiter
        if delimiter is None:
            raise IOError("Input file extension not recognized: please provide a delimiter.")
    else:
        try:
            delimiter = {'tab': '\t', 'space': ' ', ',': ',', ';': ';', ':': ':'}[delimiter]
        except KeyError:
            print("Unknown string passed as delimiter.")

    #  OUTPUTFILE  
    #  overwrite option
    if outputfile == None :
        outputfile = inputfile
        out_scope = in_scope
    else :
        out_scope = fileScope( outputfile )
        # WARNING!: to be completed

    # >>>>>>>>>>>>>>
    #  USER SUMMAR  
    # >>>>>>>>>>>>>>

    print("> Singularization of file: ", inputfile )
    print("> into file: ", outputfile )

    # >>>>>>>>>>>>
    #  EXECUTION
    # >>>>>>>>>>>>                

    progress_bar( 0., status=0 )

    df = pd.read_csv( inputfile, dtype=str, header=None, sep=delimiter, compression=in_scope.compression ) 
    df.drop_duplicates( inplace=True )

    if size <= 0 :
        df[:].to_csv( outputfile, sep=out_scope.delimiter, compression=out_scope.compression, quoting=None, index=None, header=False )
    else :
        df[:size].to_csv( outputfile, sep=out_scope.delimiter, compression=out_scope.compression, quoting=None, index=None, header=False )
    
    progress_bar( 1., status=2 )
###

################
#  EXECUTABLE  #
################

if __name__ == "__main__" :

    parser = optparse.OptionParser( conflict_handler="resolve" )

    # >>>>>>>>>>
    #  OPTIONS 
    # >>>>>>>>>>
    
    parser.add_option( '-i', '--inputfile', action="store", dest = 'inputfile', metavar='PATH/TO/FILE', type="string", help='' )
    parser.add_option( '-d', '--delimiter', action='store', dest='delimiter', type='choice', choices=['tab', 'space', ',', ';', ':'], default=None, help="Inputfile delimiter." )
    parser.add_option( '-n', action='store', dest='size', type="int", default=0, help='Number of single lines. Default is all available.' )
    parser.add_option( '-o', '--outputfile', action="store", dest='outputfile', metavar='PATH/TO/FILE', type="string", default=None, help='The file where file is saved.' )

    # >>>>>>>>>>>>>>>>>>>>>
    #  OPTIONS ASSIGNMENT 
    # >>>>>>>>>>>>>>>>>>>>>

    options, args = parser.parse_args()

    main(
        inputfile = options.inputfile,
        delimiter = options.delimiter,
        size = options.size,
        outputfile = options.outputfile
    )
