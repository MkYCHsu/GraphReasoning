from GraphReasoning.graph_tools import *
from GraphReasoning.utils import *
from GraphReasoning.graph_analysis import *

# import copy
# import re
from IPython.display import display, Markdown
# import markdown2
# import pdfkit
import uuid
import pandas as pd
import numpy as np
import networkx as nx
import os

import asyncio
# from langchain.document_loaders import (
#     PyPDFLoader,
#     UnstructuredPDFLoader,
#     PyPDFium2Loader,
#     PyPDFDirectoryLoader,
#     DirectoryLoader,
# )
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network
from tqdm.notebook import tqdm
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModel,
    logging
)

from hashlib import md5

# import torch
# from scipy.spatial.distance import cosine
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

logging.set_verbosity_error()

palette = "hls"
# Code based on: https://github.com/rahulnyk/knowledge_graph


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
           # **chunk.metadata,
            "chunk_id": md5(chunk.encode()).hexdigest(),#uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, generate, generate_figure=None, image_list=None, repeat_refine=0, do_distill=True, verbatim=False,
          
            ) -> list:
    
    os.makedirs('temp', exist_ok = True) 
    cache_file_name = (f'temp/{md5("".join(list(str(dataframe["chunk_id"]))).encode()).hexdigest()}.csv')

    try:
        results = pd.read_csv(cache_file_name,engine='python', on_bad_lines ='warn') 

    except FileNotFoundError:
        results = dataframe
        results['result']='empty'

    remaining_indices = np.where(results['result']=='empty')[0]
    random.shuffle(remaining_indices)
    for index in remaining_indices:
        try:
            results = pd.read_csv(cache_file_name,engine='python', on_bad_lines ='warn') 
        except FileNotFoundError:
            pass
        
        if results.loc[index,'result']!='empty':
            continue
        row = dataframe.iloc[index]
        results.loc[index, 'result'] = str(graphPrompt(
            row.text, 
            generate,
            generate_figure, 
            image_list,
            {"chunk_id": row.chunk_id}, 
            do_distill=do_distill,
            repeat_refine=repeat_refine, 
            verbatim=verbatim,
        ))
        # results.loc[index, 'result']= str(result)
        results.to_csv(cache_file_name, index=False)
            
    # Process results
    
    results = results.dropna().reset_index(drop=True)
    import ast
    results = results.apply(lambda row: ast.literal_eval(row.result), axis=1)
    
    # Flatten the list of lists to one single list of entities
    concept_list = np.concatenate(results).ravel().tolist()
    
    return concept_list


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: str(x).lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: str(x).lower())

    return graph_dataframe

import sys
from yachalk import chalk
sys.path.append("..")

import json

def graphPrompt(input: str, generate, generate_figure=None, image_list=None, metadata={}, #model="mistral-openorca:latest",
                do_distill=True, repeat_refine=0,verbatim=False,
               ):
    
    SYS_PROMPT_DISTILL = f'You are provided with a context chunk (delimited by ```) Your task is to respond with a concise scientific heading, summary, and a bullited list to your best understaninding and all of them should include reasoning. You should ignore human-names, references, or citations.'
    
    USER_PROMPT_DISTILL = f'In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own and provide all background needed, and include details. Ignore references. Extract the table if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.'
        
    SYS_PROMPT_FIGURE = f'You are provided a figure that contains important information. Your task is to analyze the figure very detailedly and report the scientific facts in this figure. If this figure is not an academic figure you should return "". Always return the full image location.'
    
    USER_PROMPT_FIGURE = f'In a matter-of-fact voice, rewrite this ```{input}```. The writing must stand on its own and provide all background needed, and include details. Extract the image if you think this is relevant and organize the information. Focus on scientific facts and includes citation in academic style if you see any.'
    input_fig = ''
    if generate_figure: # if image in the chunk
        
        for image_name in image_list:
            _image_name = image_name.split('/')[-1]
            if _image_name.lower() in input.lower():  
                input_fig =  f'Here is the information in the image: {image_name}' + \
                generate_figure( image = image_name, system_prompt=SYS_PROMPT_FIGURE, prompt=USER_PROMPT_FIGURE)
    
    if do_distill:
        #Do not include names, figures, plots or citations in your response, only facts."
        input = generate( system_prompt=SYS_PROMPT_DISTILL, prompt=USER_PROMPT_DISTILL)

    if input_fig:
        input += input_fig
    
    
    print(f'Refine input: {input[:100]}')
    SYS_PROMPT_GRAPHMAKER = (
        'You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. '
        'You are provided with a context chunk (delimited by ```) Your task is to extract the ontology of terms mentioned in the given context, representing the key concepts as per the context with well-defined and widely used names of materials, systems, methods.'
        'You always report a technical term or abbreviation and keep it as it is.'
        'Analyze the text carefully and produce around 10 pairs, also make sure they reflect consistent ontologies.'
        'You must format your output as a list of JSON where each element of the list contains a pair of terms packed in \", <node_1>, <node_2>, and <edge>. For details, see the following: \n'
        'You must focus on the information around the nodes you find and try to keep the nodes concise and elaborate on the edges. In other words, the node information should be concise while the edge information should be detailed.'
        '[\n'
        '   {\n'
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "Directed, succinct relationship between the two concepts, node_1 and node_2 and it must make sense when read from node_1 to edge and then to node_2," \n'
        '   }, {...}\n'
        ']'
        '\n'
        'Examples:'
        'Context: ```Alice is Marc\'s mother.```\n'
        '[\n'
        '   {\n'
        '       "node_1": "Alice",\n'
        '       "node_2": "Marc",\n'
        '       "edge": "is mother of"\n'
        '   },'
        '   {...}\n'
        ']'
        'Context: ```Silk is a strong natural fiber used to catch prey in a web. Beta-sheets control its strength.```\n'
        '[\n'
        '   {\n'
        '       "node_1": "silk",\n'
        '       "node_2": "fiber",\n'
        '       "edge": "is"\n'
        '   },' 
        '   {\n'
        '       "node_1": "beta-sheets",\n'
        '       "node_2": "strength",\n'
        '       "edge": "control"\n'
        '   },'        
        '   {\n'
        '       "node_1": "silk",\n'
        '       "node_2": "prey",\n'
        '       "edge": "catches"\n'
        '   },'
        '   {...}\n'
        ']\n'
        'Context: ```Semiconductor has an unique electrical conductive behavior between a conductor and an insulator that allows us to control its conductivity.```\n'
        '[\n'
        '   {\n'
        '       "node_1": "semiconductor",\n'
        '       "node_2": "unique electrical conductive behavior",\n'
        '       "edge": "has"\n'
        '   },'
        '   {\n'
        '       "node_1": "semiconductor",\n'
        '       "node_2": "a conductor and an insulator ",\n'
        '       "edge": "can be"\n'
        '   },' 
        '   {\n'
        '       "node_1": "unique electrical conductive behavior",\n'
        '       "node_2": "conductivity",\n'
        '       "edge": "allows us to control"\n'
        '   },'  
        '   {...}\n'
        ']\n'
        'Context: ```Samples consisted of pre-patterned photoresist on bulk silicon. A variety of trench and via structures were patterned (Fig. 1), but only 2 and 20 um width structures were characterized. The line density characterized was 4:1, although higher density 2:1 structures were also present and showed similar results (i.e., local loading effects were minimal). ![1_image_0.png](1_image_0.png)```\n'
        '[\n'
        '   {\n'
        '       "node_1": "<The full path to the image>",\n'
        '       "node_2": "trench",\n'
        '       "edge": "related to"\n'
        '   },'
        '   {\n'
        '       "node_1": "<The full path to the image>",\n'
        '       "node_2": "2 and 20 um width structure",\n'
        '       "edge": "characterizes"\n'
        '   },' 
        '   {...}\n'
        ']\n'

        )
        
    USER_PROMPT = f'Context: ```{input}``` \n\nOutput: '
    
    print ('Generating triplets...')
    response  =  generate( system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=USER_PROMPT)
    
    # Two-shots policy
    try:
        response=extract (response)
        result = json.loads(response)
        # print (result)
        result = [dict(item, **metadata) for item in result]

    except:
        
        if verbatim:
            print ('--------------------\n Fail to extract from ', response)
    

        USER_PROMPT = f'Context: ```{input}``` \n\n Your last output: ```{response}``` \n\n Corrected output:'
        response  =  generate( system_prompt=f'Make sure your output is proper json format. The node information should be concise while the edge information should be detailed, as explained as follows: {SYS_PROMPT_GRAPHMAKER}', prompt=USER_PROMPT)

        try:
            response=extract (response)
            result = json.loads(response)
            # print (result)
            result = [dict(item, **metadata) for item in result]
        except:
            print('\n\nERROR ### Here is the buggy response: ', response, '\n\n')
            result = None

    return result

def colors2Community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    df['node_1'] = df['node_1'].astype(str)
    df['node_2'] = df['node_2'].astype(str)
    df['edge'] = df['edge'].astype(str)
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2
    
def make_graph_from_text (txt,generate, generate_figure=None, image_list=None,
                          include_contextual_proximity=False,
                          graph_root='graph_root',
                          chunk_size=2500,chunk_overlap=0,do_distill=True,
                          repeat_refine=0,verbatim=False,
                          data_dir='./data_output_KG/',
                          save_HTML=False,
                          save_PDF=False,#TO DO
                         ):    
    
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
     
    outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
    
 
    splitter = RecursiveCharacterTextSplitter(
        #chunk_size=5000, #1500,
        chunk_size=chunk_size, #1500,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    pages = splitter.split_text(txt)
    print("Number of chunks = ", len(pages))
    if verbatim:
        display(Markdown (pages[0]) )
    
    df = documents2Dataframe(pages)

    ## To regenerate the graph with LLM, set this to True
    regenerate = True
    
    if regenerate:
        concepts_list = df2Graph(df,generate, generate_figure, image_list, do_distill =do_distill, repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )
        
        
        dfg1 = graph2Df(concepts_list)
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|", index=False)
        df.to_csv(outputdirectory/f"{graph_root}_chunks.csv", sep="|", index=False)
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph_clean.csv", #sep="|", index=False
                 )
        df.to_csv(outputdirectory/f"{graph_root}_chunks_clean.csv", #sep="|", index=False
                 )
    else:
        dfg1 = pd.read_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|")
    
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4 
      
    if verbatim:
        print("Shape of graph DataFrame: ", dfg1.shape)
    dfg1.head()### 
    
    if include_contextual_proximity:
        dfg2 = contextual_proximity(dfg1)
        dfg = pd.concat([dfg1, dfg2], axis=0)
        #dfg2.tail()
    else:
        dfg=dfg1
        
    
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )
    #dfg
        
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    print ("Nodes shape: ", nodes.shape)
    
    # G = nx.Graph()
    G = nx.DiGraph()
    node_list=[]
    node_1_list=[]
    node_2_list=[]
    title_list=[]
    weight_list=[]
    chunk_id_list=[]
    
    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )
        node_list.append (node)
    
    ## Add edges to the graph
    for _, row in dfg.iterrows():
        
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            chunk_id=row["chunk_id"],
            weight=row['count']/4
        )
        
        node_1_list.append (row["node_1"])
        node_2_list.append (row["node_2"])
        title_list.append (row["edge"])
        weight_list.append (row['count']/4)
        chunk_id_list.append (row['chunk_id'] )

    try:
            
        df_nodes = pd.DataFrame({"nodes": node_list} )    
        df_nodes.to_csv(f'{data_dir}/{graph_root}_nodes.csv')
        df_nodes.to_json(f'{data_dir}/{graph_root}_nodes.json')
        
        df_edges = pd.DataFrame({"node_1": node_1_list, "node_2": node_2_list,"edge": title_list, "weight": weight_list } )    
        df_edges.to_csv(f'{data_dir}/{graph_root}_edges.csv')
        df_edges.to_json(f'{data_dir}/{graph_root}_edges.json')
        
    except:
        
        print ("Error saving CSV/JSON files.")
    
    # communities_generator = nx.community.girvan_newman(G)
    # #top_level_communities = next(communities_generator)
    # next_level_communities = next(communities_generator)
    # communities = sorted(map(sorted, next_level_communities))
    
#     if verbatim:
#         print("Number of Communities = ", len(communities))
#         print("Communities: ", communities)
    
#     colors = colors2Community(communities)
#     if verbatim:
#         print ("Colors: ", colors)
    
#     for index, row in colors.iterrows():
#         G.nodes[row['node']]['group'] = row['group']
#         G.nodes[row['node']]['color'] = row['color']
#         G.nodes[row['node']]['size'] = G.degree[row['node']]

    graph_GraphML=  f'{data_dir}/{graph_root}_graph.graphml'  #  f'{data_dir}/resulting_graph.graphml',
    nx.write_graphml(G, graph_GraphML)
    
    graph_HTML = None
    net= None
    output_pdf = None
    if save_HTML:
        net = Network(
                notebook=True,
                cdn_resources="remote",
                height="900px",
                width="100%",
                select_menu=True,
                filter_menu=False,
            )

        net.from_nx(G)
        net.force_atlas_2based(central_gravity=0.015, gravity=-31)

        net.show_buttons()

        graph_HTML= f'{data_dir}/{graph_root}_graph.html'
        net.save_graph(graph_HTML,
                )
        if verbatim:
            net.show(graph_HTML,
                )


        if save_PDF:
            output_pdf=f'{data_dir}/{graph_root}_PDF.pdf'
            pdfkit.from_file(graph_HTML,  output_pdf)
        
    # res_stat=graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir,include_centrality=False, make_graph_plot=False,)
    # print ("Graph statistics: ", res_stat)
    
    return graph_HTML, graph_GraphML, G, net, output_pdf

import time
from copy import deepcopy

def add_new_subgraph_from_text(txt=None,generate=None,generate_figure=None, image_list=None, 
                               node_embeddings=None,tokenizer=None, model=None, original_graph=None,
                               data_dir_output='./data_temp/',graph_root='graph_root',
                               chunk_size=10000,chunk_overlap=2000,
                               do_update_node_embeddings=True, do_distill=True,
                               do_simplify_graph=True,size_threshold=10,
                               repeat_refine=0,similarity_threshold=0.95,
                               do_Louvain_on_new_graph=True, 
                               include_contextual_proximity=False,
                               #whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,
                               graph_GraphML_to_add=None,
                               verbatim=True,):

    display (Markdown(txt[:32]+"..."))
    graph_GraphML=None
    G_new=None
    
    res=None
    # try:
    start_time = time.time() 

    if verbatim:
        print ("Now create or load new graph...")

    if (G_to_add is not None and graph_GraphML_to_add is not None):
        print("G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added.")
        return
    elif graph_GraphML_to_add==None and G_to_add==None: #make new if no existing one provided
        print ("Make new graph from text...")
        _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                 include_contextual_proximity=include_contextual_proximity,
                                 data_dir=data_dir_output,
                                 graph_root=f'graph_root',
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                 repeat_refine=repeat_refine, 
                                 verbatim=verbatim,
                                 )
        if verbatim:
            print ("New graph from text provided is generated and saved as: ", graph_GraphML_to_add)
    elif G_to_add is None:
        if verbatim:
            print ("Loading or using provided graph... Any txt data provided will be ignored...:", G_to_add, graph_GraphML_to_add)
            G_to_add = nx.read_graphml(graph_GraphML_to_add)
    # res_newgraph=graph_statistics_and_plots_for_large_graphs(G_to_add, data_dir=data_dir_output,                                      include_centrality=False,make_graph_plot=False,                               root='new_graph')
    print("--- %s seconds ---" % (time.time() - start_time))
    # except:
        # print ("ALERT: Graph generation failed...")
        
    print ("Now grow the existing graph...")
    
    # try:
    #Load original graph
    if type(original_graph) == str:
        G = nx.read_graphml(original_graph)
    else:
        G = deepcopy(original_graph)
    print(G, G_to_add)
    G_new = nx.compose(G, G_to_add)

    if save_common_graph:
        print ("Identify common nodes and save...")
        try:

            common_nodes = set(G.nodes()).intersection(set(G_loaded.nodes()))
            subgraph = G_new.subgraph(common_nodes)
            graph_GraphML=  f'{data_dir_output}/{graph_root}_common_nodes_before_simple.graphml' 
            nx.write_graphml(subgraph, graph_GraphML)
        except: 
            print ("Common nodes identification failed.")
        print ("Done!")
    if do_update_node_embeddings:
        if verbatim:
            print ("Now update node embeddings")
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model)

    if do_simplify_graph:
        if verbatim:
            print ("Now simplify graph.")
        G_new, node_embeddings =simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                verbatim=verbatim,)


    if size_threshold >0:
        if verbatim:
            print ("Remove small fragments")            
        G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if return_only_giant_component:
        if verbatim:
            print ("Select only giant component...")   
        connected_components = sorted(nx.connected_components(G_new), key=len, reverse=True)
        G_new = G_new.subgraph(connected_components[0]).copy()
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)

    if do_Louvain_on_new_graph:
        G_new=graph_Louvain (G_new, graph_GraphML=None)
        if verbatim:
            print ("Done Louvain...")

    if verbatim:
        print ("Done update graph")

    graph_GraphML= f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'
    if verbatim:
        print ("Save new graph as: ", graph_GraphML)

    nx.write_graphml(G_new, graph_GraphML)
    if verbatim:
        print ("Done saving new graph")
    
    # res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,make_graph_plot=False,root='assembled')
    # print ("Graph statistics: ", res)

    # except:
        # print ("Error adding new graph.")
    print(G_new, graph_GraphML)
        # print (end="")

    return graph_GraphML, G_new, G_to_add, node_embeddings, res
