#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) March 2022 Francesco Camaglia, LPENS 
'''

import os
import joblib # FIXME : is there a better option?
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# unit of the logarithm
_unit_Dict_ = {
    "n": 1., "ln": 1., 
    "2": 1./np.log(2), "log2": 1./np.log(2),
    "10": 1./np.log(10), "log10": 1./np.log(10),
}

class Skeleton_Class :

    '''
    update_categories( categories )
        Change the number of categories.
    show
        Print a short summary.
    '''

    def update_categories( self, categories ):
        '''Change the number of categories.

        Parameters
        ----------        
        categories : scalar
                The new number of categories of the system (int or float). 
                If the value is lower than the observed number of categories, 
                the observed number is used instead.'''

        if categories is None:
            self.usr_n_categ = self.obs_n_categ    
        else :
            try : 
                categories = int( categories )
            except :        
                raise TypeError("The parameter `categories` must be an integer.")
            if categories > self.obs_n_categ :
                self.usr_n_categ = categories
            else :
                self.usr_n_categ = self.obs_n_categ  
                if categories < self.obs_n_categ :
                    warnings.warn("The parameter `categories` is set equal to the observed number of categories.")
    
    def show( self ):
        '''Print a short summary.'''
        
        print("Total number of counts:")
        if type(self.tot_counts) == pd.Series :
            print(self.tot_counts.to_string())
        else :
            print(self.tot_counts)
            
        print("N. of Categories:")
        print(self.obs_n_categ, ' (observed)')
        print(self.usr_n_categ, ' (a priori)')
        
        print("Recurrencies:")
        print(self.counts_hist)
    
    def save( self, filename ) :
        '''Save the object to `filename`.'''
        # `tryMakeDir`
        path_choice = os.path.dirname(filename)
        if not os.path.exists( path_choice ):
            try: 
                os.makedirs( path_choice, mode=0o777, exist_ok=True ) 
            except OSError as error: 
                print(error)
        joblib.dump( self, filename )


################
#  SUMMATIONS  #
################

def ffsum_( ff, sumList ) :
    ''' Summing methods for histograms of counts.'''

    if type(sumList) is not list : # FIXME :
        '''
            sumList = "parallel" (i)
        '''    
        output = ff.dot( sumList )

    elif len(sumList) == 2 :
        '''
            sumList[0] = "parallel" (i==j)
            sumList[1] = "perpendic" (i!=j)
        '''

        # parallel, perpendic diagonal terms
        tmp1d = sumList[0] - sumList[1].diagonal() 
        # sum over j perpendic
        tmp1d += sumList[1].dot(ff)
        # sum over i
        output = ff.dot( tmp1d )

    elif len(sumList) == 5 :
        '''
            sumList[0] = "parallel" (i==j==k)
            sumList[1] = "cross_1" (i!=j, j==k)
            sumList[2] = "cross_2" (i==k, j!=k)
            sumList[3] = "cross_3" (i==j, j!=k)
            sumList[4] = "perpendic" (i!=j, j!=k, i!=k)
        '''
        idx = np.arange(len(ff))

        # cross terms
        tmp2d = sumList[1] + sumList[2] + sumList[3] 
        # perpendic minor diagonal terms
        tmp2d -= sumList[4][:,idx,idx] + sumList[4][idx,:,idx] + sumList[4][idx,idx,:]

        # parallel, matricial diagonal, perpendic main diagonal terms
        tmp1d = sumList[0] - tmp2d.diagonal() - sumList[4][idx,idx,idx] 
        # sum over j of all 2d matrices and sum over k of perpendic terms
        tmp1d += (tmp2d + sumList[4].dot(ff)).dot(ff)
        # sum over i
        output = ff.dot( tmp1d )
    
    else :
        raise IOError("FIXME: Developer Error in `ffsum`.")

    return output
        

# >>>>>>>>>>>>
#  GRAPHICS  #
# <<<<<<<<<<<<

def experiment_rank_plot( 
    experiment_object, figsize=(3,3), color="#ff0054", 
    xlabel="rank", ylabel="frequency", logscale=True, grid=True ) :

    fig, ax = plt.subplots(ncols=1, figsize=figsize )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logscale :
        ax.set_xscale("log")
        ax.set_yscale("log")
    if grid :
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlim([1, experiment_object.usr_n_categ])

    sequences = experiment_object.data_hist
    N = experiment_object.tot_counts
    x = np.arange(len(sequences)) + 1
    y = sequences.sort_values(ascending=False).values / N
    ax.plot( x ,y, ls="", marker="o", color=color )
    return ax

def divergence_rank_plot( 
    divergence_object, figsize=(3,3), color1="#ff0054", color2="#0088ff",
    xlabel="rank", ylabel="frequency", logscale=True, grid=True, by_first=True ) :

    fig, ax = plt.subplots(ncols=1, figsize=figsize )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logscale :
        ax.set_xscale("log")
        ax.set_yscale("log")
    if grid :
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlim([1, divergence_object.usr_n_categ])

    sort_by = [ 'Exp-1' if by_first else 'Exp-2' ]
    data_hist = divergence_object.data_hist.sort_values( by=sort_by, ascending=False)
    sequences_1 = data_hist['Exp-1']
    N_1 = divergence_object.tot_counts['Exp-1']
    sequences_2 = data_hist['Exp-2']
    N_2 = divergence_object.tot_counts['Exp-2']

    x = np.arange(len(data_hist)) + 1
    y1 = sequences_1.values / N_1
    ax.plot( x, y1, ls="", marker="o", color=color1 )
    y2 = sequences_2.values / N_2
    ax.plot( x, y2, ls="", marker="s", color=color2 )
    return ax
