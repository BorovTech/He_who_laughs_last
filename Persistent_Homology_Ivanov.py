import numpy as np
import pandas as pd
import gudhi
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re


class PHI:
    def __init__(self, bare_text):
        self.bare_text = bare_text

    def text_bow(self, splitter='\n'):
        cv = CountVectorizer()
        corpus = self.bare_text.split(splitter)
        count_v = cv.fit(corpus) # cv.fit() creates the dictionary of all the unique words in the corpus.
        self.bow = cv.transform(corpus).toarray()
        self.cv = cv
    
    def text_bow_spec(self, word_count=3):
        cv = CountVectorizer()
        corpus = re.findall(' '.join(["[^ ]+"]*word_count), self.bare_text.replace('\n',' '))
        count_v = cv.fit(corpus) # cv.fit() creates the dictionary of all the unique words in the corpus.
        self.bow = cv.transform(corpus).toarray()
        self.cv = cv
        
    def point_cloud(self):
        self.bow = self.bare_text
        
    def dist_calc(self, custom_dist):
        distance_matrix = pairwise_distances(self.bow, metric='euclidean')
        if custom_dist:
            distance_max = custom_dist
        else:
            distance_max = max(map(max, distance_matrix))
        return (distance_matrix, distance_max)
    
    def sif(self, max_dim=3, custom_dist=False):
        distance_matrix, distance_max = self.dist_calc(custom_dist)
        rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=distance_max)
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        
    def sifts(self, max_dim=3, custom_dist=False):
        distance_matrix, distance_max = self.dist_calc(custom_dist)
        for i in range(len(self.bow)-1):
            distance_matrix[i+1][i] = 0
        rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=distance_max)
        self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    
    def alpha(self):
        rips_complex = gudhi.AlphaComplex(points=self.bow)
        self.simplex_tree = rips_complex.create_simplex_tree()
        
    def ph_barcode(self, dimentions=[0,1]):
        # work in progress on for dimention in dimentions:
        self.simplex_tree.compute_persistence(min_persistence=0.3)
        diag0 = self.simplex_tree.persistence_intervals_in_dimension(dimentions[0])
        diag1 = self.simplex_tree.persistence_intervals_in_dimension(dimentions[1]) 
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(5, 4), dpi=100)
        gudhi.plot_persistence_barcode(diag0, axes=ax0)
        gudhi.plot_persistence_barcode(diag1, axes=ax1)
        
        ax0.set_title('dimention 0')
        ax1.set_title('dimention 1')
        fig.tight_layout()
        plt.show()
        
    def full_sif_sifts(self, dimentions=[0,1], custom_dist=False):
        self.sif(max_dim=3, custom_dist=False)        
        self.simplex_tree.compute_persistence(min_persistence=0.3)
        
        sif_diag0 = self.simplex_tree.persistence_intervals_in_dimension(dimentions[0])
        sif_diag1 = self.simplex_tree.persistence_intervals_in_dimension(dimentions[1])
        
        self.sifts(max_dim=3, custom_dist=False)
        self.simplex_tree.compute_persistence(min_persistence=0.3)
        
        sifts_diag0 = self.simplex_tree.persistence_intervals_in_dimension(dimentions[0])
        sifts_diag1 = self.simplex_tree.persistence_intervals_in_dimension(dimentions[1])

        fig, ((sif_ax0, sifts_ax0), (sif_ax1, sifts_ax1)) = plt.subplots(nrows=2, ncols=2, figsize=(5, 4), dpi=100)

        gudhi.plot_persistence_barcode(sif_diag0, axes=sif_ax0)
        gudhi.plot_persistence_barcode(sif_diag1, axes=sif_ax1)
        gudhi.plot_persistence_barcode(sifts_diag0, axes=sifts_ax0)
        gudhi.plot_persistence_barcode(sifts_diag1, axes=sifts_ax1)
        
        sif_ax0.set_title('SIF (dimention 0)')
        sif_ax1.set_title('SIF (dimention 1)')
        sifts_ax0.set_title('SIFTS (dimention 0)')
        sifts_ax1.set_title('SIFTS (dimention 1)')
        
        #sif_ax0.invert_yaxis()
        #sif_ax1.invert_yaxis()
        #sifts_ax0.invert_yaxis()
        #sifts_ax1.invert_yaxis()
        
        fig.tight_layout()
        plt.show()
        
    def barcode(self, dimentions=[0,1], custom_dist=False):
        self.sif_simplex_tree.compute_persistence(min_persistence=0.3)
        self.sifts_simplex_tree.compute_persistence(min_persistence=0.3)
        sif_diag0 = self.sif_simplex_tree.persistence_intervals_in_dimension(dimentions[0])
        sif_diag1 = self.sif_simplex_tree.persistence_intervals_in_dimension(dimentions[1])
        sifts_diag0 = self.sifts_simplex_tree.persistence_intervals_in_dimension(dimentions[0])
        sifts_diag1 = self.sifts_simplex_tree.persistence_intervals_in_dimension(dimentions[1])

        fig, ((sif_ax0, sifts_ax0), (sif_ax1, sifts_ax1)) = plt.subplots(nrows=2, ncols=2, figsize=(5, 4), dpi=100)

        gudhi.plot_persistence_barcode(sif_diag0, axes=sif_ax0)
        gudhi.plot_persistence_barcode(sif_diag1, axes=sif_ax1)
        gudhi.plot_persistence_barcode(sifts_diag0, axes=sifts_ax0)
        gudhi.plot_persistence_barcode(sifts_diag1, axes=sifts_ax1)
        
        sif_ax0.set_title('SIF (dimention 0)')
        sif_ax1.set_title('SIF (dimention 1)')
        sifts_ax0.set_title('SIFTS (dimention 0)')
        sifts_ax1.set_title('SIFTS (dimention 1)')
        
        #sif_ax0.invert_yaxis()
        #sif_ax1.invert_yaxis()
        #sifts_ax0.invert_yaxis()
        #sifts_ax1.invert_yaxis()
        
        fig.tight_layout()
        plt.show()
