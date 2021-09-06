import os, sys
import glob
import numpy as np
import pandas as pd

from kamapack.utils import fileScope

################
#  Load Genes  #
################

def load_genes( k, **kwargs  ) :
    '''
    List Gene
    '''
    
    # loading parameters    
    par = {}
    for key, value in kwargs.items():
        par[key]=value
    List = par["List"]
    Gene = par["Gene"]
    isData = bool(par["isData"])
    
    # infer file format
    pathToFile = List[k]
    if isData is True :
        Model = par["Model"]
        Chain = par["Chain"]        
        df = openProdData( pathToFile, Model=Model, Chain=Chain )
    else :
        df = openOlga( pathToFile )
    df = df.dropna( subset=["V", "J", "nt"])
    output = list( df[Gene].values )
    del df
    
    if not output : raise ValueError( "Error: empty list." )
    else : return output
###


############################
#  Load Genes Conditioned  #
############################

def load_genes_conditioned( k, **kwargs  ) :
    
    # WARNING: to be developed
    
    # loading parameters    
    par = {}
    for key, value in kwargs.items():
        par[key]=value
    List = par["List"]
    Gene = par["Gene"]
    Condition = par["Cond"]
    
    # infer file format
    pathToFile = List[k]
    df = openOlga( pathToFile )
    if "J" in Condition : df.mask( df["J"] != Condition, inplace = True )
    else : df.mask( df["V"] != Condition, inplace = True )
    df.dropna( subset=["V", "J", "nt"], inplace = True )
    
    return df[Gene].values
###



#######################################
#  Loading Indeces from Model Params  #
#######################################

def load_indx_from_params( params_file, Gene, Sep = ";" ) :
    '''
    Returns the list of genes "Gene" in the indexed order.
    '''
    
    myfile = open( params_file, 'r' ) 
    Lines = myfile.readlines() 
    Profile = "#GeneChoice;" + Gene.upper() + "_gene"
    
    List = []
    count = 0

    # look for the desired line
    while( Profile not in Lines[ count ].strip() ) :  
        count += 1

    # load table Name;Sequence;Indx
    while( '%' in Lines[ count + 1 ].strip() ) :
        data_str = Lines[ count + 1 ].strip()
        List.append( data_str[ 1 : ].split( Sep ) )
        count += 1

    output = [ x for _,x in sorted(zip(np.array(List)[:,2].astype(int),np.array(List)[:,0]) ) ]

    return output
###



######################################
#  Loading a Profile from Marginals  #
######################################

def read_marginals( marginals_file, Profile, Choice = "" ) :
    '''
    Returns the chosen Profile (e.g. vd_ins) from the marginals_file
    '''
    
    myfile = open( marginals_file, 'r' ) 
    Lines = myfile.readlines() 

    count = 0
    # look for the desired line
    while( '@' + Profile not in Lines[ count ].strip() ) : count += 1
        
    # go to the choice identifier, skipping :  @ <Profile> // $Dim[ ... ] //
    count += 2
    while( '#' + Choice not in Lines[ count ].strip() ) : count += 1
        
    # go to the data, skipping :  # <Choice> // %data0,data1, ...
    count += 1    
    data_str = Lines[ count ].strip()
    
    # read comma separated values eliminating %
    output = list( np.array( data_str[ 1 : ].split( ',' ), dtype = np.float ) )
    
    return output
###


###############
#  open Olga  #
###############

def openOlga( pathToFile ) :
    # WARNING! : 
    # lack of generality here for columns names
    
    in_scope = fileScope( pathToFile )
    df = pd.read_csv( pathToFile, 
                     sep=in_scope.delimiter, compression=in_scope.compression )
    
    # index
    if len( df.columns ) > 4 :
        index_col = 0
    else :
        index_col = None
    # headers (check if first line is headers)
    if df.columns.values[ int(index_col == 0) ] in ["a", "aa", "A", "AA", "CDR3aa"] :
        skiprows = 1
    else :
        skiprows = None
    
    df = pd.read_csv( pathToFile, index_col=index_col,
                     names = ["aa","V","J","nt"], header=None, skiprows = skiprows,
                     sep=in_scope.delimiter, compression=in_scope.compression )
    
    return df

##########################
#  open Productive Data  #
##########################

def openProdData( pathToFile, Model, Chain, verbose=False, pgen_filter=None ) :
        
    # avoid tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    from sonnia.processing import Processing

    # cdr3_aa, v, j, cdr3_nt
    cdr3 = openOlga( pathToFile ).dropna()    
    
    # >>>>>>>>>>>>>>>>>>>>>
    #  CLEAN non-F GENES  #
    # >>>>>>>>>>>>>>>>>>>>>
    
    # This makes a stricter selection than already included in Giulio's preprocessing
    Func_Set = set(["F"])
    
    J_gene_anch = pd.read_csv( f"{Model}/J_gene_CDR3_anchors.csv" , sep="," ) 
    nice_J = { gene for gene,func in J_gene_anch[["gene","function"]].values if func in Func_Set }
    V_gene_anch = pd.read_csv( f"{Model}/V_gene_CDR3_anchors.csv" , sep="," ) 
    nice_V = { gene for gene,func in V_gene_anch[["gene","function"]].values if func in Func_Set }
       
    mask_V = [ (cdr3.at[ i, "V" ] in nice_V) for i in cdr3.index ]
    mask_J = [ (cdr3.at[ i, "J" ] in nice_J) for i in cdr3.index ]
    cdr3 = cdr3[ np.logical_and(mask_V,mask_J) ]
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    #  GIULIO's preProcessing  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    
    if Chain == "CA" : vj_bool = True
    else : vj_bool = False
    
    cdr3.rename( columns = {"aa":"amino_acid", "V": "v_gene", "J": "j_gene", "nt": "nucleotide"},
          inplace = True )
    processor = Processing(custom_model_folder=Model, vj=vj_bool, verbose=verbose)
    aux = processor.filter_dataframe(cdr3,apply_selection=False)
    cdr3 = cdr3[aux["selection"]].copy()
    cdr3.rename( columns = {"amino_acid":"aa", "v_gene":"V", "j_gene":"J", "nucleotide":"nt"},
          inplace = True )
    
    # >>>>>>>>>>>>>>>
    #  PGEN FILTER  #
    # >>>>>>>>>>>>>>>   
    
    # to be better developed
    # WARNING! to be developed
    if pgen_filter is not None :
        mask = pd.read_csv( pgen_filter )['Pgen'].values > 0
        cdr3 = cdr3[ mask ].copy()

    return cdr3[["aa","V","J","nt"]]


#########################
#  get BioID from Olga  #
#########################

def getBioID_fromOlga( df, AminoAcid=True, Vgene=True, Jgene=True, sep='+' ) :
    '''
    It produces a TCR BioID for an Olga dataframe.
    '''
    
    tag = 'BioID_'
    if sep == '' :
        print( 'Warning: separation character ')
    if AminoAcid is True :
        tag = tag + 'aa'
    if Vgene is True :
        tag = tag + 'V'
    if Jgene is True :
        tag = tag + 'J'
        
    iddf = pd.DataFrame(index=df.index)
    
    # NOTE : is there a way to make this more elegant?
    if tag == 'BioID_aaVJ' :
        iddf[tag] = [ f'{aa}{sep}{V}{sep}{J}' for aa,V,J in df[['aa','V','J']].values ]
    elif tag == 'BioID_aaV' :
        iddf[tag] = [ f'{aa}{sep}{V}' for aa,V in df[['aa','V']].values ]
    elif tag == 'BioID_aaJ' :
        iddf[tag] = [ f'{aa}{sep}{J}' for aa,J in df[['aa','J']].values ]
    elif tag == 'BioID_VJ' :
        iddf[tag] = [ f'{V}{sep}{J}' for aa,J in df[['V','J']].values ]   
    else :
        raise IOError('At least two between AminoAcid, Vgene and Jgene must be `True`.')
   
    return iddf



####################
#  POISSON STDERR  #
####################

def poisson_stderr( frequencies, N ):
     
    freqs = np.array( frequencies )    
    stderr = np.sqrt( ( freqs + np.power(freqs,2) ) / N )
    
    return stderr
###
