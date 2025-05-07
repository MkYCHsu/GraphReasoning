#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis

# In[1]:


print('hello')

import sys

try:
    doc_thread_i = int(sys.argv[1])
    doc_total_threads = int(sys.argv[2])

except: 
    print('something\'s wrong')
    raise Exception("Abort")

# In[2]:


# import openai
config_list = [
    {
        "model":"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "api_key":"960e3a3bf627e826c526c3d6cd45e3a7a94302ff95d330b35e7f27bed2e5a166",
        "max_tokens": 20000
    },
]


# In[3]:


from together import Together
client = Together(api_key=config_list[0]["api_key"])


# In[4]:


#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis


import os
from GraphReasoning import *

# In[2]:



# In[5]:


verbatim=False


# In[6]:


doc_data_dir = '/home/mkychsu/pool/SG_abstracts/'
data_dir='./GRAPHDATA'    
data_dir_output='./GRAPHDATA_OUTPUT'

max_tokens = config_list[0]['max_tokens']


# ### Load dataset

# In[7]:


### Load dataset of papers

# In[3]:

import pandas as pd
import glob

doc_list=sorted(glob.glob(f'{doc_data_dir}/*.xls'))
df_list = []
for i, doc in enumerate(doc_list):
    print(i, doc)
    df_list.append(pd.read_excel(doc))
    
df = pd.concat(df_list, axis=0)


# In[9]:


df = df.drop_duplicates()
df = df.reset_index(drop=True)


# In[10]:


df.shape


# ### Set up LLM client:

# In[ ]:





# In[11]:


import instructor
from typing import List
from PIL import Image
import base64

from pydantic import BaseModel

class Node(BaseModel):
    id: str
    type: str
        
class Edge(BaseModel):
    source: str
    target: str
    relation: str
        
class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

response_model = KnowledgeGraph
system_prompt = '''
You are a scientific assistant extracting knowledge graphs from text.
Return a JSON with two fields: <nodes> and <edges>.\n
Each node must have <id> and <type>.\n
Each edge must have <source>, <target>, and <relation>.
'''


def generate(system_prompt=system_prompt, 
             prompt="",temperature=0.333,
             max_tokens=config_list[0]['max_tokens'], response_model=KnowledgeGraph, 
            ):     

    if system_prompt==None:
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ]

    else:
        messages=[
            {"role": "system",  "content": f"{system_prompt}"},
            {"role": "user", "content": f"{prompt}"},
        ]

    
    create = instructor.patch(
        create=client.chat.completions.create,
        mode=instructor.Mode.JSON_SCHEMA,
    )

    return create(messages=messages,   
                    model=config_list[0]["model"],
                    max_tokens=max_tokens,
                    temperature=0.333,
                    response_model=response_model,
                   )

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

def generate_figure(image, system_prompt=system_prompt, 
                prompt="", temperature=0,
                ):
    try:
        pwd = os.getcwd()
        image = image.split(pwd)[-1]
        image=Path('.').glob(f'**/{image}', case_sensitive=False)
        image = list(image)[0]
    except:
        return '' 
    image_uri = image_to_base64_data_uri(image)
    
    messages = [
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": "Describe this image in detail please."},
            ],
        },
    ]
        
    return create(messages=messages,   
                    model=config_list[0]["model"],
                    max_tokens=max_tokens,
                    temperature=0.333,
                    response_model=response_model,
                   ).choices[0].message.content


# In[ ]:





# In[ ]:


import networkx as nx
from GraphReasoning import make_graph_from_text
from datetime import datetime
import time
import torch

G=nx.DiGraph()
with torch.no_grad():
    # for i, doc in enumerate(doc_list):

    for i, row in df.iterrows():
        if i % doc_total_threads != doc_thread_i:
            continue
   # for i, row in df.iterrows():

        title = row['Article Title']
        title=title.replace('/','|')
        
        doi = row['DOI'] # not using it for now
        txt=row['Abstract']
        
        graph_root = f'{title}'
        _graph_GraphML= f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'
        
        image_list = glob.glob(''.join(doc.split('/')[:-1])+'/*png') # if running for 
        print(image_list)
        break
        current_graph = f'{data_dir}/{graph_root}_graph.graphml'
        if os.path.exists(_graph_GraphML):
            G = nx.read_graphml(_graph_GraphML)
            print(f'Main KG loaded: {_graph_GraphML}, {G}')
            continue
            
        if os.path.exists(f'{title}_err.txt'):
            print(f'No. {i}: {title} got something wrong.')
            continue

        while not os.path.exists(current_graph):
            print(f"generating KG for {title}")
            try:
                if type(txt) is not str:
                    break # format of abstract is wrong 
                now = datetime.now()
                _, current_graph, _, _, _ = make_graph_from_text(txt,generate,
                                      generate_figure, image_list,
                                      graph_root=graph_root,do_distill=False,
                                      chunk_size=200000,chunk_overlap=0,
                                      repeat_refine=0,verbatim=False,
                                      data_dir=data_dir,
                                                                 
                                      save_PDF=False,
                                     )
                print("Time: ", datetime.now()-now)
                break # successfully generate KGs without error
            except:
                print('Reach rate limit')
                time.sleep(60)
            
 


# In[ ]:





# ### Test

# In[ ]:


system_prompt = '''
You are a scientific assistant extracting knowledge graphs from text.
Return a JSON with two fields: <nodes> and <edges>.\n
Each node must have <id> and <type>.\n
Each edge must have <source>, <target>, and <relation>.
'''

prompt = f"Text: {txt}\n\nExtract the knowledge graph in structured JSON."
result = generate(system_prompt = system_prompt,
                 prompt = prompt, response_model=response_model, max_tokens=10240)


# In[ ]:


result


# In[ ]:


# print(type(result.nodes[0]))
# print(result.edges)


# In[ ]:


# G = nx.DiGraph()
# # Add nodes
# for node in result.nodes:
#     G.add_node(node.id, type=node.type)

# # Add edges
# for edge in result.edges:
#     G.add_edge(edge.source, edge.target, relation=edge.relation)


# In[ ]:


# import matplotlib.pyplot as plt
# import networkx as nx

# # Print stats
# print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
# graphml_path: str = "knowledge_graph_2.graphml"
# # Save as GraphML
# nx.write_graphml(G, graphml_path)
# print(f"💾 Graph saved to: {graphml_path}")

# # Display graph
# plt.figure(figsize=(10, 7))
# pos = nx.spring_layout(G, seed=42)  # consistent layout
# nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1000, font_size=10, font_weight="bold", edge_color="gray")
# edge_labels = nx.get_edge_attributes(G, "relation")
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
# plt.title("Knowledge Graph")
# plt.axis("off")
# plt.show()


# In[ ]:


# # doc = doc_list[0]
# # title = doc.split('/')[-1].split('.pdf')[0]
# # graph_root = f'{title}'
# import networkx as nx

# G = nx.read_graphml(f'{data_dir_output}/TSMC_KG_70b.graphml')
# # G = nx.read_graphml(f'{data_dir_output}/4books_integrated.graphml')
# print(f'KG loaded: {G}')
# # node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )



# In[ ]:


# from GraphReasoning import load_embeddings
# embedding_file='TSMC_KG_70b.pkl'
# generate_new_embeddings=True

# if os.path.exists(f'{data_dir}/{embedding_file}'):
#     generate_new_embeddings=False

# if generate_new_embeddings:
#     try:
#         node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
#     except:
#         node_embeddings = generate_node_embeddings(nx.DiGraph(), embedding_tokenizer, embedding_model, )
        
#     save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')

# else:
#     filename = f"{data_dir}/{embedding_file}"
#     # file_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir='./')
#     # print(f"File downloaded at: {file_path}")
#     node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




