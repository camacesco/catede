# coding: utf-8

'''
Francesco Camaglia, June 2020 LPENS

WARNING: 
As it is, the program works only on files with the format of generated olga files.
'''


#####################
#  LOADING MODULES  #
#####################

import os, sys
import pandas as pd
import numpy as np
import optparse
from tqdm import tqdm 
import multiprocessing

sys.path.append( os.path.realpath( __file__ ).split('kamapack')[0]  )
from kamapack.handle import ngrams
from kamapack.utils import tryMakeDir, fileScope
from kamapack.handle.statbiophys import openOlga

##########
#  MAIN  #
##########

def main( ) :
    '''
    Evaluete n-grams entropy. Example of usage:
    > python ngram_entropy.py -n 5 -i file.csv.gz --samples 25   
    '''
    
    parser = optparse.OptionParser( conflict_handler="resolve" )

    #####################
    # MANDATORY OPTIONS #
    #####################
    
    parser.add_option( '-n', action='store', dest='num', type="int", help='The "n-grams" length.' )
    parser.add_option( '-a', '--alphabet', action='store', dest='alph', type="string", default="AA", help="The alphabet of the reads." )
    parser.add_option( '-M', '--max_categories', action='store', dest='max_categ', default=0, type="int", help="The maximum number of sequences considered." )
    parser.add_option( '-i', '--inputfile', action="store", dest = 'inputfile', metavar='PATH/TO/FILE', type="string", help='PATH/TO/FILE where Olga sequences are stored.' )

    ######################
    #  SPECIFIC OPTIONS  #
    ######################

    parser.add_option( '-s', '--skip', action="store", dest="skip", type="int", default=0, help="" )
    parser.add_option( '-B', '--beginning_const', action="store", dest="beg", type="int", default=0, help="" )
    parser.add_option( '-E', '--end_const', action="store", dest="end", type="int", default=0, help="" )
    parser.add_option( '-u','--unit', action='store', dest='unit_usr', type='choice', choices = [ "ln", "log2", "log10" ], default="log2", help="The unit of the logarithm." )
    parser.add_option( '--NSBstdDev', action='store_true', dest='NSBstdDev', default=False, help="Wheter to compute NSB std. deviation." )
    
    ##################
    #  USER OPTIONS  #
    ##################

    parser.add_option( '--samples', action='store', dest='n_subSamples', type="int", default=1, help="Number of subsamples (including all dataset)." )
    parser.add_option( '-o', '--outputfile', action="store", dest='outputfile', metavar='PATH/TO/FILE', type="string", default=None, help='The file where entropy has to be saved.' )
    parser.add_option( '--more_entropies', action='store_true', dest='more_entr', default=False, help="Wheter to compute other entropy estimators alongside with NSB." )

    ######################
    # OPTIONS ASSIGNMENT #
    ######################

    options, args = parser.parse_args()

    
    # >>>>>>>>>>>>>>>>>>>
    #  LOADING OPTIONS  #
    # >>>>>>>>>>>>>>>>>>>
    
    num = options.num
    alph = options.alph
    max_categ = options.max_categ
    inputfile = options.inputfile
    outputfile = options.outputfile
    n_subSamples = options.n_subSamples
    skip = options.skip
    beg = options.beg
    end = options.end
    more_entr = options.more_entr
    unit_usr = options.unit_usr
    NSBstdDev = options.NSBstdDev
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  OUTPUT DEFINITIONS  #
    # >>>>>>>>>>>>>>>>>>>>>>

    if outputfile is not None:
        # User name  
        # clean from eventual extension
        if len( outputfile.split( '.' ) ) != 1:  
            outputfile = outputfile.split('.')[0]
        # eventually create folder
        if os.path.dirname( outputfile ) != "" :
            FLDR = os.path.dirname( outputfile )
            tryMakeDir( FLDR )
            outputfile = os.path.basename( outputfile )
        else :
            FLDR = "."
    else :
        # Default name  
        if os.path.dirname( inputfile ) != "" : 
            FLDR = os.path.dirname( inputfile )
        else : 
            FLDR = "."
        outputfile = f"n{num}-{alph}-s{skip}-b{beg}e{end}"
    # Output names                                                  
    entropy_file = f"{FLDR}/{outputfile}-entr.csv"
    dict_file = f"{FLDR}/{outputfile}-dic.csv.gz"

    # initilialize entropy file  
    Results = []
    headers = ['K_seq', 'N_ngrams', 'K_obs', 'NSB']
    if NSBstdDev is True : headers.append( 'NSBstdDev' )
    if more_entr is True : headers.append( ['ML', 'MM', 'CS'] )
    Results.append( headers )

    ###################
    #  SEQUENCE FILE  #
    ###################                   

    # WARNING!: lack of generality
    df = openOlga( inputfile ).dropna( )
    
    #  subsamples definition
    if max_categ == 0 : max_categ = len(df)
    if n_subSamples == 1 : K_vec = [ max_cat ] # option for 1 single measure 
    else :
        # logarithmic subsample sets 
        K_vec = np.logspace( np.log10( 5e2 ), np.log10( max_categ ), num = n_subSamples ).astype(int)
    
    # check for max_categ greater than dataframe and change subsamples
    if max_categ > len( df ) :
        K_vec = [ i for i in K_vec if i < len(df) ]
        K_vec.append( len(df) )

    #################
    #  USER SUMMAR  #
    #################   

    print("> "+str(num)+"grams entropy computation")
    print("> Alphabet of the sequences: " + ngrams._Alphabet_[alph][1] )
    print("> Maximum number of clonotypes considered: ", max_categ)
    print("> Number of subsamples considered: ", n_subSamples)
    print("> skip: ", skip, "| beginning: ", beg, "| end: ", end)
    print("> Input file: ", inputfile )
    print("> Output file: ", entropy_file )
    print("> Number of threads: ", multiprocessing.cpu_count() )

    ###############
    #  EXECUTION  #
    ###############
    
    for subsamp_indx in tqdm( range( 0, len(K_vec), 1 ) ) :
        
        n_categ = K_vec[ subsamp_indx ]
        df_subsamp = df.sample( n = n_categ )   
        
        # the subsample counts_list and estimators are computed
        Gear = ngrams.ngram_gear( num=num, alph=alph ) # initilzation of counts dict
        Gear.counts_dict_update( df_subsamp["aa"].values, 
                                skip=skip, beg=beg, end=end )
        thisSample = [ n_categ, Gear.experiment.N, Gear.experiment.obs_categ ]
        
        if NSBstdDev == True :
            temp = Gear.experiment.entropy( method="NSB", unit=unit_usr, err=True )
            thisSample.append( temp[0] )
            thisSample.append( temp[1] )
        else :
            thisSample.append( Gear.experiment.entropy( method="NSB", unit=unit_usr, err=False ) )
              
        if more_entr == True :
            thisSample.append( Gear.experiment.entropy( method="ML", unit=unit_usr ) )
            thisSample.append( Gear.experiment.entropy( method="MM", unit=unit_usr ) )
            thisSample.append( Gear.experiment.entropy( method="CS", unit=unit_usr ) )

        Results.append( thisSample )   
  
    # >>>>>>>>>>>>>>
    #   CONCLUSION
    # >>>>>>>>>>>>>>

    # save entropy of subsamples
    pd.DataFrame( Results ).to_csv( entropy_file, sep=",", index=None, header=False )
    # save final dictionary of ngrams count if requested
    Gear.save_file_dict( dict_file )
    
###



################
#  EXECUTABLE  #
################

if __name__ == "__main__" : main( )
