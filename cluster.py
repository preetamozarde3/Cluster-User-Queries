import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from networkx.algorithms.components.connected import connected_components
import os
import json
import warnings
import networkx
import time
import numpy as np

# def toGraph(l):
#     G = networkx.Graph()
#     for part in l:
#         G.add_nodes_from(part)
#         G.add_edges_from(toEdges(part))
#     return G

# def toEdges(l):
#     it = iter(l)
#     last = next(it)

#     for current in it:
#         yield last, current
#         last = current 

def similarityIndex(s1, s2, wordmodel):
    '''
    To compare the two sentences for their similarity using the gensim wordmodel 
    and return a similarity index
    '''
    print("In similarityIndex function")
    if s1 == s2:
        return 1.0

    try:
        s1words = s1.split()
    except:
        return 0
    s2words = s2.split()

    s1words = set(s1words)    
    for word in s1words.copy():
        if word in stopwords.words('english'):
            s1words.remove(word)
    
    s2words = set(s2words)
    for word in s2words.copy():
        if word in stopwords.words('english'):
            s2words.remove(word)

    s1words = list(s1words)
    s2words = list(s2words)    

    s1set = set(s1words)
    s2set = set(s2words)
    
    if len(s1set & s2set) == 0:
        return 0.0
    for word in s1set.copy():
        try:
            flag = wordmodel.key_to_index[word]
        except KeyError:
                s1words.remove(word)
        # print(word)
        # print(wordmodel.key_to_index[word])
        # flag = wordmodel.key_to_index[word]
        # if not flag:
    for word in s2set.copy():
        try:
            flag = wordmodel.key_to_index[word]
        except KeyError:
                s2words.remove(word)

    if len(s1words) and len(s2words):
        return wordmodel.n_similarity(s1words, s2words)
    else:
        return

def cluster_queries(df_comp, df_base):
    df = df_comp['User_Says']
    # df_question = df_comp['Question']
    df_base = df_base['Base']
    wordmodelfile = 'MIN.bin'
    wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary = True, limit=20000)


    sentences = df
    # print(df)
    # print(df_base)
    no_of_sentences = len(df)
    no_of_comps = len(df_base)
    print(no_of_comps)
    print(no_of_sentences)
    st = time.time()
    similarity_matrix = [[-1 for c in range(no_of_comps)] for r in range(no_of_sentences)]
    et = time.time()
    print(len(similarity_matrix))

    row = 0
    for response1 in sentences:
        column = 0
        for response2 in df_base:
            # if response1 == response2:
            #     column += 1
            #     continue
            # print('DEBUG: {}'.format(len(sentences)))
            print(f'row: {row}')
            print(f'column: {column}')
            similarity_matrix[row][column] = similarityIndex(response1, response2, wordmodel)
            column += 1
        row += 1
    et = time.time()
    s = 'Similarity matrix populated in %f secs. ' % (et-st)
    print(s)

    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if similarity_matrix[i][j] is None:
                # print(f'similarity_matrix[i][j]:  {similarity_matrix[i][j]}')
                similarity_matrix[i][j] = 0.0

    # print('Similarity Matrix: ')
    # print(similarity_matrix)
    print(f'len(similarity): {len(similarity_matrix)}')
    setlist = []
    cluster = []
    index = 0
    for score_row, response in zip(similarity_matrix, sentences):
        if max(score_row) > 0.8:
            cluster.append(1)
        else:
            cluster.append(0)
        # print(f'Max score: {max_similarity}')
        max_sim_index = index
        # print(score_row)
        # print(np.array(score_row).sum())
        if np.array(score_row).sum() > 0:
            max_sim_index = np.array(score_row).argmax()
        #     print(f'Max sim index: {max_sim_index}')
        #     print() 
        # print(f'Respnse: {response}')
        # print(f'results array: ', df_base[max_sim_index])
        # if set([response, df_base[max_sim_index]]) not in setlist:
        #     setlist.append([response, df_base[max_sim_index]])
        # index += 1

    # G = toGraph(setlist)
    # setlist = list(connected_components(G))

    # novel_sub_categories = {}
    # index = 0
    # for category in setlist:
    #     novel_sub_categories[index] = list(category)
    #     index += 1

    # results = novel_sub_categories

    # print('***********************************************************')
    # with open('out_new_trial.json', 'w') as temp:
    #     json.dump(results, temp)
    # print(results)

    print(len(df))
    print(len(cluster))
    final_df = pd.DataFrame(list(zip(df, cluster)))
    final_df.to_csv('output.csv')

df_comp  = pd.read_csv('floatbot (6).csv')
df_base  = pd.read_csv('5c9c9cb4e6293615275e1e62_csvfile.csv')
cluster_queries(df_comp, df_base)