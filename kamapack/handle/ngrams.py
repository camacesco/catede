# coding: utf-8
 
'''
ngrams Module:
Francesco Camaglia, LPENS June 2020
'''

import os, sys
import gzip
import numpy as np
import pandas as pd

import itertools
import multiprocessing

from kamapack.utils import *
from kamapack.entropy import shannon

##################################
#  DEFAULT ALPHABET DEFINITIONS  #
##################################

# alphabets dictionary
_Alphabet_ = { 
    'NT': (['A', 'C', 'G', 'T'], "nucleotide"),
    'AA': (['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'], "amminoacid"),
    'AA_Stop': (['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y', '*'], "amminoacid + stopping codon")
    }

############################
#  NGRAM EXPERIMENT CLASS  #
############################

class ngram_gear:
    
    def __init__( self, num=None, alph=None, ngrams_file_input=None ):
 
        '''
        Parameters
        ----------
                
        num: scalar
                the number of letters which form a ngram. It must be an integer greater than 0.
                Ignored if <ngrams_file_input> is chosen.                                          
        alph: option
                the alphabet where the elements belong to. The implemented options are:
                - "AA" : amino acid alphabet (20 letters);
                - "AA_Stop" : amino acid alphabet with stop codon "*" (21 letters);
                - "NT" : nucleotides (4 letters).       
                Ignored if <ngrams_file_input> is chosen. 
        ngrams_file_input: path/to/file
                    load the dictionary with counts has saved in "path/to/file.csv".
                       
        Attributes
        ----------
        num :
        alph :
        counts_dict :
        experiment :
        '''
        
        # >>>>>>>>>>>>>>
        #  INPUT LOAD  #
        # >>>>>>>>>>>>>>

        if ngrams_file_input is not None: 
            #  load parameters from file 
            self.num, self.alph, self.counts_dict = load_file_dict( ngrams_file_input ) 

        else :
            # load parameters from user
            
            #  num  #
            if num is None :    
                raise IOError('num must specified if ngrams_file_input is not.')
            elif type( num ) != int : 
                raise TypeError('num must be an integer.')
            elif num < 1 : 
                raise IOError('num must be greater than 0.')
            else : 
                self.num = num
            
            #  alph  #
            if alph is None :    
                raise IOError('alph must specified if ngrams_file_input is not.')
            elif type( alph ) != str : 
                raise TypeError('alph must be a string.')
            elif alph not in list( _Alphabet_.keys( ) ) :
                raise IOError('Alphabet unknown. Options are : '+str(list(_Alphabet_.keys())) )
            else : 
                self.alph = alph 
                
            # assign empty counts_dict
            self.counts_dict = pd.DataFrame()
                
        self._experiment_update_()  

    '''
    Methods
    -------
    '''
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  ASSIGN COUNTS DICT  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    def assign_counts_dict( self, counts_dict ) :
        '''
        Assign the attribute <counts_dict> which must be a pandas DataFrame
        with ngram for index and count in the column.
        '''
        
        # WARNING!: missing a check for the user counts_dict alphabet
        
        self.counts_dict = counts_dict          
        self._experiment_update_()
    ###
  
    # >>>>>>>>>>>>>>>>>>>>>>
    #  COUNTS LIST UPDATE  #
    # >>>>>>>>>>>>>>>>>>>>>>

    def counts_dict_update( self, sequences, file_output=None, skip=None, beg=None, end=None ):
        '''
        It updates counts computing ngrams on each entry of the list "sequences".
        Be careful : there is no control on sequences 

        Parameters
        ----------
        sequences: list
                the list of sequences from which ngrams are extracted.
        skip: scalar:
                the number of letters to skip after each ngram before considering the next one.
                If skip is set to <num>-1, ngrams are considered one after the other (from the left).
                Default is 0. 
        beg: scalar:
                constant number of letters to skip at the beginning of each sequence. Default is 0. 
        end: scalar:
                constant number of letters to skip at the end of each sequence. Default is 0. 
        file_output: path/to/file.csv.gz, optional
                the path/to/file.csv.gz where to save the output (otherways no file is produced). 
        '''                

        # >>>>>>>>>>>>>>
        #  INPUT LOAD  #
        # >>>>>>>>>>>>>>
        
        #  skip  #
        if skip is None : skip = 0 # Default
        elif type( skip ) != int : raise TypeError('"skip" must be an integer.')
        elif skip < 0 : raise IOError('"skip" must be greater or equal to 0.')
        else : pass 

        #  beg  #
        if beg is None : beg = 0 # Default
        elif type( beg ) != int : raise IOError('"beg" must be an integer.')
        elif beg < 0 : raise IOError('"beg" must be greater or equal to 0.')
        else : pass

        #  end  #
        if end is None : end = 0 # Default
        elif type( end ) != int : raise IOError('"end" must be an integer.')
        elif end < 0 : raise IOError('"end" must be greater or equal to 0.')
        else :  pass
                
        # >>>>>>>>>>>>>
        #  EXECUTION  #
        # >>>>>>>>>>>>>
        
        #  UPDATE DICTIONARY from SEQUENCES 
        # WARNING!: maybe parallel here is useless
        POOL = multiprocessing.Pool( multiprocessing.cpu_count() ) 
        results = POOL.starmap( inSequence, [ ( Seq, self.num, skip, beg, end ) for Seq in sequences ] )
        POOL.close()
        update_dict = pd.DataFrame.from_dict(dict_generator(reduceList(results)), orient='index')[0]
        
        if not self.counts_dict.empty : 
            if not update_dict.empty : 
                self.counts_dict = update_dict.add(self.counts_dict, fill_value=0)
        else : 
            self.counts_dict = update_dict
        self._experiment_update_()

        #  SAVING FILEOUT 
        if file_output : self.save_file_dict( file_output )
    
    ###
    
    # >>>>>>>>>>>>>>>>>>>>>>
    #  EXPERIMENT UPDATE  #
    # >>>>>>>>>>>>>>>>>>>>>>
    
    def _experiment_update_( self ) :
        
        categories = np.power( len( _Alphabet_[ self.alph ][0] ), self.num )
        self.experiment = shannon.experiment( list( self.counts_dict.values ), 
                                             categories=categories, iscount=True )
    ###
    
    # >>>>>>>>>>>>>>>>>>>>>>>>
    #  SAVE FILE DICTIONARY  #
    # >>>>>>>>>>>>>>>>>>>>>>>>

    def save_file_dict( self, file_output ) :
        '''
        Save n-grams dictionary to a gzipped file.
        '''

        if type( file_output ) is str : file_output = file_output.split(".")[0] + ".csv.gz"
        else : raise IOError( 'Unrecognized filename : ' + fileout )

        self.counts_dict.to_csv( file_output, header=False, index=True, 
                                sep = ",", compression="gzip" ) 
    ###
    
###


################
#  INSEQUENCE  #
################

def inSequence( thisSeq, num, skip, beg, end ):
    '''
    It returns the list of <num>-grams contained in thisSeq[beg, len(thisSeq)-end].
    Distance between grams is set by the <skip> parameter.
    It returns an empty list if no ngram can be extracted.
    e.g.
    ----    
        thisSeq :  A B C D E F G H
        indeces :  0 1 2 3 4 5 6 7

        num=3, skip=1, beg=1, end=2
        returns: [ "BCD", "DEF" ]

        num=4, skip=0, beg=3, end=1
        returns: [ "DEFG" ]

        num=4, skip=0, beg=3, end=2
        returns: [ ]
    '''

    first_indx = beg
    last_indx = len( thisSeq ) - end - num
    
    return [ thisSeq[ i : num + i ] for i in range ( first_indx , 1 + last_indx , 1 + skip ) ]  
###



##########################
#  LOAD FILE DICTIONARY  #
##########################

def load_file_dict( file_input ) :
    '''
    To open the ngrams dictionary <file_input>.
    '''
    
    # load dictionary
    in_scope = fileScope( file_input )
    df = pd.read_csv( file_input, header=None, index_col=0, na_filter = False,
                     compression=in_scope.compression, sep=in_scope.delimiter )
    
    # CHECK num :
    Lengths = list( map( len, df.index.astype(str) ) ) 
    if len(Lengths) == 0 :
        raise IOError( "The ngrams file is empty." )
    elif len( set(Lengths) ) > 1 :
        raise IOError( "The ngrams file contains ngrams with multiple lengths." )
    else :
        num = Lengths[0]
    
    # CHECK alph :
    thisAlph = set(reduceList([[c for c in i] for i in df.index]))
    for alph in _Alphabet_ :
        superset = set( _Alphabet_[ alph ][0] )
        if thisAlph.issubset( superset ) : break
        else : alph = None
    if alph is None:
        raise IOError( "The ngrams file alphabet is not available." )
    
    return num, alph, df
###
