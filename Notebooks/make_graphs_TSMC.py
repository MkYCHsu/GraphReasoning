#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis


import os

from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from huggingface_hub import hf_hub_download
from GraphReasoning import *


# In[2]:


verbatim=False


### Load dataset of papers

# In[3]:


import glob

doc_data_dir = './paper/'
doc_list=sorted(glob.glob(f'{doc_data_dir}/*/*.md'))

print(doc_list)

# In[6]:


data_dir='./GRAPHDATA'    
data_dir_output='./GRAPHDATA_OUTPUT/'

filename = 'Meta-Llama-3.1-70B-Instruct-Q4_K_L.gguf'
n_ctx = 20000

file_path = f'~/pool/llm/{filename}'

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.notebook import tqdm
from IPython.display import display, Markdown

tokenizer_model=f'~/pool/llm/SEMIKONG-8b-GPTQ'
embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,use_fast=False)
embedding_model = AutoModelForCausalLM.from_pretrained(tokenizer_model,output_hidden_states=True).to('cuda')

from GraphReasoning import load_embeddings, save_embeddings, generate_node_embeddings

embedding_file='KG.pkl'
generate_new_embeddings=True

from PIL import Image
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "~/pool/llm/Cephalo-Phi-3-vision-128k-4b-alpha"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 


import torch

if os.path.exists(f'{data_dir}/{embedding_file}'):
    generate_new_embeddings=False
    
with torch.no_grad():
    if generate_new_embeddings:

        try:
            import networkx as nx

            graph_root="mainKG"
            graph_GraphML= f'{data_dir_output}/{graph_root}.graphml'
            G = nx.read_graphml(graph_GraphML)
            node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
        except:
            node_embeddings = generate_node_embeddings(nx.DiGraph(), embedding_tokenizer, embedding_model, )

        save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')

    else:
        filename = f"{data_dir}/{embedding_file}"
        node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')
        
        


# In[8]:


from llama_cpp import Llama
llm = Llama(model_path=file_path,
             n_gpu_layers=-1,verbose= True, #False,#False,
             n_ctx=n_ctx,
             main_gpu=0,
             n_threads= 8 ,
             n_threads_batch=32,
             # chat_format='mistral-instruct',
             )
# In[10]:


def generate(system_prompt='You are a senior engineer. Try to find the clear relationship in the provided information', 
                         prompt="",temperature=0.333,
                         max_tokens=n_ctx, 
                         ):     
    try:
        if system_prompt==None:
            messages=[
                {"role": "user", "content": prompt},
            ]

        else:
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        # result=llm.create_chat_completion_openai_v1(
        result=llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        # return result.choices[0].message.content
        return result['choices'][0]['message']['content']
    except:

        return ''


def generate_figure(image, system_prompt="You are an expert in this field. Try your best to give a clear and concise answer.", 
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
        
    image = Image.open(image)
    print(f'Extracting infomation from {image}')
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 
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
    
    

import networkx as nx

from GraphReasoning import make_graph_from_text, add_new_subgraph_from_text, save_embeddings

import numpy as np
import sys

try:
    doc_i = int(sys.argv[1])
    doc_list = [doc_list[doc_i]]  # quick hack from existing codes to generate only one knowledge graph at a time.
except: 
    # If no doc index is specified, it by default will go through all the documents and bad for parallel operations, but this is bad for parallel operation so it's banned.
    raise Exception("No index provided. Abort")

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


        if os.path.exists(f'{title}_err.txt'):
            print(f'No. {i}: {title} got something wrong.')
            continue

        elif os.path.exists(f'{data_dir}/{graph_root}_graph.graphml'):
            print(f'Found a graph fragment to merge: {graph_root}: {doc}.')
            graph_GraphML = f'{data_dir}/{graph_root}_graph.graphml'

            print(f'Merging graph No. {i}: {doc} to the main one')
            # try:
            _, G, _, node_embeddings, res = add_new_subgraph_from_text(
                               node_embeddings=node_embeddings,
                               embedding_tokenizer=embedding_tokenizer,
                               embedding_model=embedding_model,
                               original_graph=G, data_dir_output=data_dir_output, graph_root=graph_root,
                               do_simplify_graph=True,size_threshold=10,
                               repeat_refine=0,similarity_threshold=0.95,
                               do_Louvain_on_new_graph=True, include_contextual_proximity=False,
                               #whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,graph_GraphML_to_add=graph_GraphML,
                               verbatim=True,)

            save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')

        else:
            # continue
            print(f'Generating a knowledge graph from {doc}')
            with open(doc, "r") as f:
                txt = " ".join(f.read().splitlines())  # separate lines with a single space

            _, graph_GraphML, _, _, _ = make_graph_from_text(txt,generate,
                                  generate_figure, image_list,
                                  include_contextual_proximity=False,
                                  graph_root=graph_root,
                                  chunk_size=2000,chunk_overlap=100,
                                  repeat_refine=0,verbatim=False,
                                  data_dir=data_dir,
                                  save_PDF=False,#TO DO
                                 )
 


