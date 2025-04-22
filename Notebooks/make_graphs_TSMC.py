#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis

# In[1]:


print('hello')


# In[2]:


#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis


import os

# from tqdm.notebook import tqdm
# from IPython.display import display, Markdown
# from huggingface_hub import hf_hub_download
from GraphReasoning import *
from GraphReasoning.graph_generation import *

# In[2]:

# VLM cephalo performance vs Qwen?


# In[3]:


verbatim=False


# In[ ]:





# ### Load dataset

# In[15]:


### Load dataset of papers

# In[3]:


import glob

doc_data_dir = './paper/'
doc_list=sorted(glob.glob(f'{doc_data_dir}/*/*.md'))

print(doc_list)


# In[14]:


# # In[6]:


data_dir='./GRAPHDATA_TSMC'    
data_dir_output='./GRAPHDATA_TSMC_OUTPUT'

filename = 'Meta-Llama-3.3-70B-Instruct-Q4_K_L.gguf'
n_ctx = 20000

file_path = f'/home/mkychsu/pool/llm/{filename}'

from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm.notebook import tqdm
# from IPython.display import display, Markdown


tokenizer_model=f'/home/mkychsu/pool/llm/SEMIKONG-8b-GPTQ'
embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,use_fast=False, device_map="cuda:0")
embedding_model = AutoModelForCausalLM.from_pretrained(tokenizer_model,output_hidden_states=True).to('cuda:0')

from GraphReasoning import load_embeddings, save_embeddings, generate_node_embeddings

embedding_file='TSMC_KG_70b.pkl'
# generate_new_embeddings=True

# from PIL import Image
# from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "/home/mkychsu/pool/llm/Cephalo-Phi-3-vision-128k-4b-alpha"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:1", trust_remote_code=True, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, device_map="cuda:1", trust_remote_code=True) 




# In[ ]:





# In[6]:


import torch

if os.path.exists(f'{data_dir}/{embedding_file}'):
    generate_new_embeddings=False

generate_new_embeddings=True

with torch.no_grad():
    if generate_new_embeddings:

        try:
            import networkx as nx

            graph_root="TSMC_KG_70b"
            graph_GraphML= f'{data_dir_output}/{graph_root}.graphml'
            G = nx.read_graphml(graph_GraphML)
            node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
        except:
            node_embeddings = generate_node_embeddings(nx.DiGraph(), embedding_tokenizer, embedding_model, )

        save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')

    else:
        filename = f"{data_dir}/{embedding_file}"
        node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')
        
        


# In[ ]:





# ### Set up LLM client:

# In[7]:

# In[10]:


# In[19]:


import instructor
from typing import List
from pydantic import BaseModel
from PIL import Image

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
response_model = KnowledgeGraph

def generate(system_prompt=system_prompt, 
             prompt="",temperature=0.333,
             max_tokens=n_ctx, response_model=KnowledgeGraph, 
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

    if 'json' in prompt.lower() and 'graph' in prompt.lower():
        create = instructor.patch(
            create=llm.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON_SCHEMA,
        )

        result = create(messages=messages, 
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_model=response_model,
                       )
        return result
    else:
        
        result=llm.create_chat_completion_openai_v1(
    
        # result=llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return result.choices[0].message.content #['choices'][0]['message']['content']



def generate_figure(image, system_prompt=system_prompt, 
                prompt="", model=model, processor=processor, temperature=0,
                           ):
    if system_prompt==None:
        messages=[
            {"role": "user", "content": f"Here is the image: <|image_1|>.\n" + prompt},
        ]

    else:
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user", "content": f"Here is the image: <|image_1|>.\n" + prompt},
        ]
        
    try:
        pwd = os.getcwd()
        image = image.split(pwd)[-1]
        image=Path('.').glob(f'**/{image}', case_sensitive=False)
        image = list(image)[0]
    except:
        return '' 
    image = Image.open(image)
    print(f'Extracting infomation from {image}')
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:1") 
    generation_args = { 
                        "max_new_tokens": 1024, 
                        "temperature": 0.1, 
                        "do_sample": True, 
                        "stop_strings": ['<|end|>',
                                         '<|endoftext|>'],
                        "tokenizer": processor.tokenizer,
                      } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    


from llama_cpp import Llama

from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

llm = Llama(model_path=file_path,
             n_gpu_layers=-1,verbose= True, #False,#False,
             n_ctx=n_ctx,
             main_gpu=0,
             n_threads= 8 ,
             n_threads_batch=32,
             draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
             logits_all=True,
             # chat_format='mistral-instruct',
             )

# In[20]:


# response_model = KnowledgeGraph
# system_prompt = '''
# You are a scientific assistant extracting knowledge graphs from text.
# Return a JSON with two fields: <nodes> and <edges>.\n
# Each node must have <id> and <type>.\n
# Each edge must have <source>, <target>, and <relation>.
# '''
# passage = '''
# Chemical resistance refers to a material's ability to withstand prolonged exposure to various chemicals without significant degradation or loss of properties. Factors like material composition, chemical concentration, temperature, and exposure time influence a material's chemical resistance. 
# Here's a more detailed explanation:
# Factors Affecting Chemical Resistance:
# Material Composition:
# The type of polymer bonds, the degree of crystallinity, branching, and the distance between the bonds are crucial factors in determining the chemical resistance of a material. 
# Chemical Type and Concentration:
# Different chemicals have varying effects on materials. Strong acids or bases, for example, can cause significant degradation, while others may have little effect. 
# Temperature:
# Higher temperatures can accelerate chemical reactions and potentially reduce a material's resistance. 
# Exposure Time:
# Prolonged exposure to a chemical can lead to greater degradation than short-term exposure. 
# Stress:
# Mechanical stress can also influence chemical resistance, as a stressed material may be more susceptible to chemical attack. 
# Examples of Materials with Good Chemical Resistance:
# Polytetrafluoroethylene (PTFE) (Teflon):
# Known for its resistance to almost all chemicals and solvents due to its highly crystalline structure and strong carbon-fluorine bonds. 
# LDPE (low density polyethylene), Silicone, PTFE, PFA, FEP, and certain types of polyurethane tubing:
# These materials have excellent chemical resistance for specific applications. 
# Testing Chemical Resistance:
# Immersion Tests:
# Samples are immersed in various test fluids to evaluate their resistance to chemicals. 
# Visual Inspection:
# Changes in color, shine, softening, swelling, detachment, or blistering are observed after exposure. 
# Physical Tests:
# Measurements of weight, volume, or dimensional changes, retention of tensile strength, elongation, or impact strength are used to assess the material's performance. 
# Standards:
# ISO 2812 and DIN EN ISO 4628-1 to -5 provide guidelines for determining the chemical resistance of materials and surfaces. 
# Chemical Resistance Charts:
# Many resources, like those from Chemline Plastics, Mettler Toledo, and Bürkert, offer chemical resistance charts to help users select appropriate materials for specific applications. 
# These charts provide ratings or classifications (e.g., A = Excellent, B = Good, C = Fair, X = Not Recommended) based on the chemical's effect on the material. 
# It's important to remember that these charts are general guidelines and actual performance may vary depending on specific conditions. 
# '''
# prompt = f"Text: {passage}\n\nExtract the knowledge graph in structured JSON."
# result = generate(system_prompt = system_prompt,
#                  prompt = prompt, response_model=response_model, max_tokens=10240)


# In[21]:


# print(type(result.nodes[0]))
# print(result.edges)


# In[22]:


# G = nx.DiGraph()
# # Add nodes
# for node in result.nodes:
#     G.add_node(node.id, type=node.type)

# # Add edges
# for edge in result.edges:
#     G.add_edge(edge.source, edge.target, relation=edge.relation)


# In[23]:


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


import networkx as nx

from GraphReasoning import make_graph_from_text, add_new_subgraph_from_text, save_embeddings

import numpy as np
import sys

try:
    doc_i = int(sys.argv[1])
    doc_list = [doc_list[doc_i]]  # quick hack from existing codes to generate only one knowledge graph at a time.
except: 
    # If no doc index is specified, it by default will go through all the documents and bad for parallel operations, but this is bad for parallel operation so it's banned.
    # raise Exception("No index provided. Abort")
    pass

with torch.no_grad():
    for i, doc in enumerate(doc_list):

        title = doc.split('/')[-1].split('.md')[0]
        graph_root = f'{title}'

        _graph_GraphML= f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'
        txt=''
        # print(f'{doc}')
        image_list = glob.glob('/'.join(doc.split('/')[:-1])+'/*png')
        # break
        if os.path.exists(_graph_GraphML):
            G = nx.read_graphml(_graph_GraphML)
            print(f'Main KG loaded: {_graph_GraphML}, {G}')
            continue
        elif i == 0:
            G = nx.DiGraph()
            

        if os.path.exists(f'{title}_err.txt'):
            print(f'No. {i}: {title} got something wrong.')
            continue

        else:
            print(f'Generating a knowledge graph from {doc}')
            with open(doc, "r") as f:
                txt = " ".join(f.read().splitlines())  # separate lines with a single space

            _, graph_GraphML, _, _, _ = make_graph_from_text(txt,generate,
                                  generate_figure, image_list,
                                  graph_root=graph_root,
                                  chunk_size=2000,chunk_overlap=500,
                                  repeat_refine=0,verbatim=False,
                                  data_dir=data_dir,
                                  save_PDF=False,#TO DO
                                 )
 




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




