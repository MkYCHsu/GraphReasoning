#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis

# In[1]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device='cuda:0'

from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from huggingface_hub import hf_hub_download
from GraphReasoning import *


# In[2]:


verbatim=True


# ### Load dataset

# In[3]:


import glob

doc_data_dir = '/home/mkychsu/pool/TSMC/dataset_manual/'
doc_list=[f'{doc_data_dir}LAMMPS_Manual.pdf',
]

# ### Load the LLM and the tokenizer

# In[6]:


#Hugging Face repo
# repository_id = "lamm-mit/GraphReasoning"
data_dir='./GRAPHDATA_TSMC'    
data_dir_output='./GRAPHDATA_TSMC_OUTPUT/'

# data_dir_output='./GRAPHDATA_OUTPUT/'
# graph_name='BioGraph.graphml'

# make_dir_if_needed(data_dir)
# make_dir_if_needed(data_dir_output)

tokenizer_model="BAAI/bge-large-en-v1.5"
# tokenizer_model="f'/home/mkychsu/pool/llm/Mistral-7B-Instruct-v0.3/tokenizer.json"

embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, ) 
embedding_model = AutoModel.from_pretrained(tokenizer_model, )
# embedding_model.to('cuda:0')



# In[7]:


# filename = f"{data_dir}/{graph_name}"
# file_path = hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='./')
# print(f"File downloaded at: {file_path}")

# graph_name=f'{data_dir}/{graph_name}'
# G = nx.read_graphml(graph_name)


# repository_id='TheBloke/Mistral-7B-Instruct-v0.1-GGUF'
filename='mistral-7b-instruct-v0.1.Q8_0.gguf'

# repository_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF'
# filenmame='Meta-Llama-3.1-8B-Instruct-Q8_0.gguf'

# file_path=hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='/home/mkychsu/pool/llm')
file_path = f'/home/mkychsu/pool/llm/{filename}'


# ### Load LLM: clean Mistral 7B

# In[8]:


from llama_cpp import Llama
# import llama_cpp

llm = Llama(model_path=file_path,
             n_gpu_layers=-1,verbose= True, #False,#False,
             n_ctx=8192,
             main_gpu=0,
             n_threads= 8 ,
             n_threads_batch=32,
             # chat_format='mistral-instruct',
             )



# In[9]:


# llm.verbose = False


# In[10]:


def generate_Mistral (system_prompt='You are a semiconductor engineer. Try to find the clear relationship in the provided information', 
                         prompt="How to make silicon into chip?",temperature=0.333,
                         max_tokens=8192, 
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
        pass
       # return generate_Mistral( system_prompt=system_prompt, prompt=prompt[:len(prompt)//2+100], temperature=temperature, max_tokens=max_tokens) + \
       # generate_Mistral( system_prompt=system_prompt, prompt=prompt[len(prompt)//2-100:], temperature=temperature, max_tokens=max_tokens)
       


import numpy as np

while doc_list != []:
    doc = np.random.choice(doc_list)   
    i = doc_list.index(doc)
    
    title = doc.split('/')[-1].split('.pdf')[0]
    doc = doc.split('/')
    doc[-1]=f'{title}/{title}.md'
    doc='/'.join(doc)
    
    title = doc.split('/')[-1].split('.md')[0]
    graph_root = f'{title}'
    if os.path.exists(f'{title}.txt'):
        print(f'No. {i}: {title} has been read')
        doc_list.pop(i)
        continue
    
    if os.path.exists(f'{title}_err.txt'):
        print(f'No. {i}: {title} got something wrong.')
        doc_list.pop(i)
        continue
    print(f'{doc}')
    with open(doc, "r") as f:
        txt = " ".join(f.read().splitlines())  # separate lines with a single space

    try:
        _, graph_GraphML, _, _, _ = make_graph_from_text(txt,generate_Mistral,
                              include_contextual_proximity=False,
                              graph_root=graph_root,
                              chunk_size=2000,chunk_overlap=200,
                              repeat_refine=0,verbatim=False,
                              data_dir=data_dir,
                              save_PDF=False,#TO DO
                             )
    except Exception as e:
        print(f'Something is wrong with No. {i}: {title}.')
        f = open(f'{title}_err.txt', 'w')
        f.write(f'{e}\n{txt}')
        f.close()          
        continue
