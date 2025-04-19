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
# from huggingface_hub import hf_hub_download
# from GraphReasoning import *


# In[2]:


verbatim=False


# ### Load dataset

# In[3]:


import glob

# doc_data_dir = '/home/mkychsu/pool/TSMC/dataset_textbook/'
doc_data_dir = './paper/'
doc_list=[f'{doc_data_dir}dry-etching-technology-for-semiconductors_compress.pdf',
          f'{doc_data_dir}plasma-etching-an-introduction_compress.pdf',
          f'{doc_data_dir}handbook-of-silicon-wafer-cleaning-technology-third-edition_compress.pdf',
          f'{doc_data_dir}Ultraclean Surface Processing of Silicon Wafers - PDF Free Download.pdf',
          f'{doc_data_dir}Atomic Layer Processing_semiconductor.pdf'   
]

doc_data_dir = './paper_new/'

# doc_list=[]

doc_list_all=sorted(glob.glob(f'{doc_data_dir}*.pdf'))

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

for i, doc in enumerate(doc_list_all):
    if doc in doc_list:
        continue
    # try:
    #     temp_doc = doc_list_all[i+1]
    #     sim = similar(temp_doc.lower(), doc.lower())
    #     if sim < 0.9:
    #         doc_list.append(doc)
    #     else:
    #         if abs(os.stat(doc).st_size - os.stat(temp_doc).st_size)/os.stat(doc).st_size < 1e-3:
    #             print(f'{i}:{sim},\n {doc} \n {temp_doc}')
    #         else:
    #             doc_list.append(doc)
    # except:
    #     pass
    doc_list.append(doc)
    


# In[ ]:





# In[4]:


doc_list


# In[5]:


import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.notebook import tqdm
from IPython.display import display, Markdown

verbatim=False

data_dir='./GRAPHDATA_TSMC'    
data_dir_output='./GRAPHDATA_TSMC_OUTPUT'

tokenizer_model=f'/home/mkychsu/pool/llm/SEMIKONG-8b-GPTQ'
# embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)
# embedding_model = AutoModelForCausalLM.from_pretrained(tokenizer_model, device_map='cuda', torch_dtype='auto', output_hidden_states=True)

embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model,use_fast=False)
embedding_model = AutoModelForCausalLM.from_pretrained(tokenizer_model,output_hidden_states=True).to('cuda')



# In[6]:


import networkx as nx

graph_root="5books_70b"
graph_GraphML= f'{data_dir_output}/{graph_root}.graphml'
G = nx.read_graphml(graph_GraphML)


# In[7]:


# edges = list(G.out_edges(data=True))
# nodes = set()
# nodes.add(edges[0][0])
# nodes.add(edges[1][0])

# G=G.subgraph(nodes)


# In[9]:


from GraphReasoning import load_embeddings, save_embeddings, generate_node_embeddings
embedding_file='TSMC_KG_70b.pkl'
generate_new_embeddings=True

# if os.path.exists(f'{data_dir}/{embedding_file}'):
#     generate_new_embeddings=False

if generate_new_embeddings:
    
    # try:
    node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
    # except:
    #     node_embeddings = generate_node_embeddings(nx.DiGraph(), embedding_tokenizer, embedding_model, )
        
    save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')

else:
    filename = f"{data_dir}/{embedding_file}"
    # file_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir='./')
    # print(f"File downloaded at: {file_path}")
    node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')


# ### Set up LLM client:

# In[7]:


import autogen, openai
config_list = [
    {
        "model":"Llama3.1",
        "base_url": "http://localhost:8080/v1",
        "api_key":"NULL",
        "max_tokens": 10000
    },
]


# In[8]:


from openai import OpenAI
class llm:
    def __init__(self, llm_config):
        self.client = OpenAI(api_key=llm_config["api_key"],
                             base_url=llm_config["base_url"],
                             )
        self.model = llm_config["model"]
        self.max_tokens = llm_config["max_tokens"]
        
    def generate_cli(self, system_prompt="You are an expert in this field. Try your best to give a clear and concise answer.", 
                           prompt="Hello world! I am", temperature=0,
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
            result=self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )

            return result.choices[0].message.content
        except:
            return ''
        


# In[9]:


llm=llm(config_list[0])


# In[10]:


generate = llm.generate_cli

import networkx as nx

from GraphReasoning import make_graph_from_text, add_new_subgraph_from_text, save_embeddings
for i, doc in enumerate(doc_list):

    title = doc.split('/')[-1].split('.pdf')[0]
    doc = doc.split('/')
    doc[-2]+=f'_txt'
    doc[-1]=title+f'/{title}.md'
    doc='/'.join(doc)
    
    graph_root = f'{title}'
    
    _graph_GraphML= f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'
    txt=''
    print(f'{doc}')
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
        _, G, _, node_embeddings, res = add_new_subgraph_from_text('', generate,
                           node_embeddings, embedding_tokenizer, embedding_model,
                           original_graph=G, data_dir_output=data_dir_output, graph_root=graph_root,
                           chunk_size=2000,chunk_overlap=200,
                           do_simplify_graph=True,size_threshold=10,
                           repeat_refine=0,similarity_threshold=0.95,
                           do_Louvain_on_new_graph=True, include_contextual_proximity=False,
                           #whether or not to simplify, uses similiraty_threshold defined above
                           return_only_giant_component=False,
                           save_common_graph=False,G_to_add=None,graph_GraphML_to_add=graph_GraphML,
                           verbatim=True,)

        save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')
        # except:
            # print(f'No. {i}: {doc} fail to add')
        
    else:
        # continue
        
        print(f'Generating a knowledge graph from {doc}')
        with open(doc, "r") as f:
            txt = " ".join(f.read().splitlines())  # separate lines with a single space

        try:
            _, graph_GraphML, _, _, _ = make_graph_from_text(txt,generate,
                                  include_contextual_proximity=False,
                                  graph_root=graph_root,
                                  chunk_size=1000,chunk_overlap=100,
                                  repeat_refine=0,verbatim=False,
                                  data_dir=data_dir,
                                  save_PDF=False,#TO DO
                                 )
        except Exception as e:
            print(f'Something is wrong with No. {i}: {doc}.')
            f = open(f'{title}_err.txt', 'w')
            f.write(f'{e}\n{txt}')
            f.close()          
            pass


# In[ ]:


# doc = doc_list[0]
# title = doc.split('/')[-1].split('.pdf')[0]
# graph_root = f'{title}'
import networkx as nx

G = nx.read_graphml(f'{data_dir_output}/5books_70b.graphml')
# G = nx.read_graphml(f'{data_dir_output}/4books_integrated.graphml')
print(f'KG loaded: {G}')
# node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )



# In[ ]:


from GraphReasoning import load_embeddings
embedding_file='TSMC_KG_70b.pkl'
generate_new_embeddings=True

if os.path.exists(f'{data_dir}/{embedding_file}'):
    generate_new_embeddings=False

if generate_new_embeddings:
    try:
        node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
    except:
        node_embeddings = generate_node_embeddings(nx.DiGraph(), embedding_tokenizer, embedding_model, )
        
    save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')

else:
    filename = f"{data_dir}/{embedding_file}"
    # file_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir='./')
    # print(f"File downloaded at: {file_path}")
    node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')


# In[ ]:


node_sorted=sorted(list(G.nodes), key= lambda x: -len(x.split()))
node_sorted[0]


# In[ ]:


G.out_edges(list(G.nodes)[1234])


# In[57]:


list(G.nodes)[1234]


# In[58]:


len(node_sorted)


# In[61]:


from GraphReasoning import node_report

node_report(G)


# In[62]:


graph=G


# In[ ]:


commnity_file='community_data_70b.pkl'
import pickle
try:
    community_data = pickle.load( open( f"{data_dir_output}/{commnity_file}", "rb" ) )
    communities = community_data['communities']
    community_summaries = community_data['community_summaries']
    
except:
    communities = detect_communities(graph.to_undirected())
    community_summaries = summarize_communities(graph, communities, generate)
    dict_community = {'communities': communities, 'community_summaries': community_summaries} 
    pickle.dump( dict_community, open( f"{data_dir_output}/{commnity_file}", "wb" ))
    


# In[85]:


from GraphReasoning import local_search


# In[82]:


Q=[]
Q.append('How to make a silicon (si) radical etch with aspect ratio = 15 and cd = 2.5 nm, at 1 atm, 300K?')
Q.append('What are the knobs that can change the uniformity in radical si etching process?')
Q.append('How to increase the selectivity ratio (gas/power/pressure) of si to oxide in ICP (Inductively Coupled Plasma) etching?')
Q.append('How to reduce the particle in the dechuck step?')
Q.append('How to improve the cleaning or etching ability of Al particles?')



# In[84]:


response = local_search(Q[0], generate, graph, node_embeddings, embedding_tokenizer, embedding_model, N_samples=3, similarity_threshold=0.95)
print(response)


# In[81]:


question = 'What is cvd uniformity and etching uniformity?'

response = local_search(question, generate, graph, node_embeddings, embedding_tokenizer, embedding_model, N_samples=3, similarity_threshold=0.95)
print(response)


# In[93]:


A1_local=[]
for q in Q:
    response_local = local_search(q, generate, graph, node_embeddings, embedding_tokenizer, embedding_model, N_samples=3, similarity_threshold=0.9)
    
    A1_local.append(response_local)
    


# In[29]:


A1_global=[]
for q in Q:
    response_global = global_search(q, generate, graph, communities, community_summaries, node_embeddings, embedding_tokenizer, embedding_model, N_samples=3, similarity_threshold=0.9)
    
    A1_global.append(response_global)
    


# In[30]:


A2=[]
for q in Q:
    final_response = generate(system_prompt= "Answer the query detailedly.",
                                     prompt=f"Query: {q}.")
    A2.append(final_response)


# In[59]:


for q,a in zip(Q,A1_local):
    print(q)
    print(a)
    print('----------------------')


# In[32]:


for q,a in zip(Q,A1_global):
    print(q)
    print(a)
    print('----------------------')


# In[33]:


for q,a in zip(Q,A2):
    print(q)
    print(a)
    print('----------------------')


# In[ ]:





# In[ ]:





# In[34]:


visualize_embeddings_2d_pretty_and_sample(node_embeddings, n_clusters=10, n_samples=10, data_dir=data_dir_output, alpha=.7)


# In[35]:


# describe_communities_with_plots_complex(G, N=6, data_dir=data_dir_output)


# In[36]:


# graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir_output,include_centrality=False,
                                               # make_graph_plot=False,)


# In[37]:


is_scale_free (G, data_dir=data_dir_output)


# In[38]:


# find_best_fitting_node_list("semiconductor", node_embeddings, embedding_tokenizer, embedding_model, 5)


# In[39]:


# find_best_fitting_node_list("better manufactoring process for semiconductor", node_embeddings , embedding_tokenizer, embedding_model, 5)


# In[ ]:





# In[ ]:





# In[ ]:




