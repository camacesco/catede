import os 
import numpy as np
import pandas as pd
from kamapack.utils import tryMakeDir

# >>>>>>>>>>>>>>>>>>>>>>>
#  BINARY DATA PROCESS  #
# >>>>>>>>>>>>>>>>>>>>>>>

class Binary_Data :
    '''
    A class to merge two pandas dataframe and divide them in training and testing.
    TO UPDATE
    '''
    
    def __init__( self, df_Pos=None, df_Neg=None, PERC=0.75, 
        return_sharing=False, max_size_ratio=1., upper_bound=None, load=None ) :
        
        if load :
            self.load( load )
        else :

            # Look for Shared sequences
            merged = df_Pos.merge( df_Neg, how="outer", indicator=True )
            both = merged.copy().loc[ lambda x : x['_merge']=='both' ]
            both.drop( labels="_merge", axis = 1, inplace = True )
            self.shared = both.values

            # delete shared sequences between the datasets
            if return_sharing is False :

                # redefine data excluding shared sequences
                df_Pos = merged.copy().loc[ lambda x : x['_merge']=='left_only' ]
                df_Pos.drop(labels="_merge", axis = 1, inplace = True)
                #
                df_Neg = merged.copy().loc[ lambda x : x['_merge']=='right_only' ]
                df_Neg.drop(labels="_merge", axis = 1, inplace = True)

            #
            # choice of the training data size    
            #

            P_len = len(df_Pos)
            N_len = len(df_Neg) 

            if upper_bound == None :  

                if np.max( [ P_len/N_len , N_len/P_len ] ) <= max_size_ratio :
                    size_Pos_train = int( PERC * P_len )
                    size_Neg_train = int( PERC * N_len ) 

                elif P_len > N_len :
                    size_Neg_train = int( PERC * N_len )  
                    size_Pos_train = int( max_size_ratio * size_Neg_train )

                else :
                    size_Pos_train = int( PERC * P_len )
                    size_Neg_train = int( max_size_ratio * size_Pos_train )      

            elif upper_bound > 0 :
                size_Pos_train = int( PERC * np.min( [ P_len, N_len, upper_bound ] ) )
                size_Neg_train = size_Pos_train

            # Positive data
            rand_indx = np.arange( P_len )
            np.random.shuffle( rand_indx )
            self.positive = np.split( df_Pos.iloc[ rand_indx ].values, [ size_Pos_train ] )

            # Negative data 
            rand_indx = np.arange( N_len )
            np.random.shuffle( rand_indx )
            self.negative = np.split( df_Neg.iloc[ rand_indx ].values, [ size_Neg_train ] )        
    ###        
    
    # >>>>>>>>>>
    #  saving  #
    # >>>>>>>>>>
    
    def save( self, outpath ):
        '''
        Where to save the dataframe
        '''
        df = pd.DataFrame()

        attach = pd.DataFrame(self.positive[0], columns=['aa','V','J'])
        attach[['Label','Use']] = '1', 'train'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.positive[1], columns=['aa','V','J'])
        attach[['Label','Use']] = '1', 'test'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.negative[0], columns=['aa','V','J'])
        attach[['Label','Use']] = '0', 'train'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.negative[1], columns=['aa','V','J'])
        attach[['Label','Use']] = '0', 'test'
        df = df.append(attach, ignore_index=True)

        attach = pd.DataFrame(self.shared, columns=['aa','V','J'])
        attach[['Label','Use']] = 'shared', 'test'
        df = df.append(attach, ignore_index=True)
        
        # saving
        df.to_csv( f'{outpath}/binary_data.csv.gz', sep=',', index=False, compression='gzip' )
        
    # >>>>>>>>>>
    #  loading  #
    # >>>>>>>>>>
    
    def load( self, outpath ):
        '''
        Where to load the dataframe
        '''
        df = pd.read_csv( f'{outpath}/binary_data.csv.gz', compression='gzip', low_memory=False, dtype=str )

        positive_train = df[np.logical_and(df['Label']=='1', df['Use']== 'train' )][['aa','V','J']].values
        positive_test = df[np.logical_and(df['Label']=='1', df['Use']== 'test' )][['aa','V','J']].values
        self.positive = [positive_train, positive_test]
        
        negative_train = df[np.logical_and(df['Label']=='0', df['Use']== 'train' )][['aa','V','J']].values
        negative_test = df[np.logical_and(df['Label']=='0', df['Use']== 'test' )][['aa','V','J']].values  
        self.negative = [negative_train, negative_test]
            
        self.shared = df[np.logical_and(df['Label']=='shared', df['Use']== 'test' )][['aa','V','J']].values

        del df
###



#############################
#  CLASSIFY ON SONIA CLASS  #
#############################

from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from kamapack.handle.sonia_one_sided import SoniaOneSided # WARNING!: to be generalized

# to avoid warning due to encoding sonia features
import warnings
warnings.filterwarnings("ignore")
# avoid tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class ClassifyOnSonia( object ):
    '''
    Logistic Regression over sonia features
    
    '''

    # >>>>>>>>>>>>>>
    #  INITIALIZE  #
    # >>>>>>>>>>>>>>

    def __init__( self, which_sonia_model=None, load_model=None, custom_pgen_model=None, 
                 vj=False, include_indep_genes=False, include_joint_genes=True ) :
        
        if load_model is not None :
            self.sonia_model = load_model
            
        elif which_sonia_model == "both" :
            # Default Sonia Left to Right Position model
            self.sonia_model = SoniaLeftposRightpos( custom_pgen_model=custom_pgen_model, vj=vj,
                                                    include_indep_genes=include_indep_genes,
                                                    include_joint_genes=include_joint_genes )
        elif which_sonia_model in ['left', 'right'] :
            # Default Sonia Left to Right Position model
            self.sonia_model = SoniaOneSided( custom_pgen_model=custom_pgen_model, vj=vj,
                                             include_indep_genes=include_indep_genes,
                                             include_joint_genes=include_joint_genes, 
                                             feat_side=which_sonia_model )      
        else :
            raise IOError('Unknwon option for `which_sonia_model`.')
        
        # Number of feautures associated to each sequence according to the model
        self.input_size = len(self.sonia_model.features)
    ###
    
    '''
    Methods
    '''
               
    # >>>>>>>>>>
    #  encode  #
    # >>>>>>>>>>

    def encode( self, x ):
        '''
        Extract features from sequence in x according to sonia model
        '''
        
        data = np.array( [ self.sonia_model.find_seq_features( d ) for d in x ] )
        data_enc = np.zeros( ( len(data), self.input_size ), dtype=np.int8 )
        for i in range( len(data_enc) ): 
            data_enc[ i ][ data[ i ] ] = 1
        #
        return data_enc
###



###############################
#  LINEAR LEFT POS RIGHT POS  #
###############################

from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# mono layer logistic regression on
class Logistic_Sonia_LeftRight( ClassifyOnSonia ):
    '''
    Logistic Regression over sonia features (i.e. activation='sigmoid')
    '''
                
    # >>>>>>>>>>>>>>>>>>>>>>>>>>
    #  update_model_structure  #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>

    def update_model_structure( self, input_layer=None, output_layer=None, 
                               initialize=True, activation='sigmoid', 
                               optimizer='Adam', loss='binary_crossentropy', 
                               metrics=["binary_accuracy"] ) :
        
        if initialize is True:
            # Initiliaze ML model layers which bring from n. features to 2 possibilities (categorical)
            input_layer = keras.layers.Input( shape = (self.input_size,) )
            output_layer = keras.layers.Dense( 1, activation=activation )( input_layer )

        # Define model from the specified layers 
        self.model = keras.models.Model( inputs=input_layer, outputs=output_layer )

        # Once the model is created it is then configurated with losses and metrics 
        self.model.compile( optimizer=optimizer, loss=loss, metrics=metrics )
    
    # >>>>>>>
    #  fit  #
    # >>>>>>>
    
    def fit( self, x, y, batch_size=300, epochs=100, val_split=0 ) :
        '''
        Fit the keras supervised model on data x with label y encoding features of x to x_enc.
        It shuffles data automatically.
        '''
        
        x_enc = self.encode( x )
        
        # shuffle indeces
        rand_indx = np.arange( len(y) )
        np.random.shuffle( rand_indx )
        # fit to the model
        self.history = self.model.fit( x_enc[ rand_indx ], y[ rand_indx ],
                                      batch_size=batch_size, epochs=epochs,
                                      verbose=0, validation_split=val_split )
 
    # >>>>>>>>>>>
    #  predict  #
    # >>>>>>>>>>>

    def predict( self, x ):
        x_enc = self.encode( x )
        #
        return self.model.predict( x_enc )
    
    # >>>>>>>>>>
    #  saving  #
    # >>>>>>>>>>
    
    def save( self, outpath ) :
        self.model.save( outpath )
    
    # >>>>>>>>>>
    #  saving  #
    # >>>>>>>>>>
    
    def load_model( self, outpath ) :
        self.model = keras.models.load_model( outpath )
        
###