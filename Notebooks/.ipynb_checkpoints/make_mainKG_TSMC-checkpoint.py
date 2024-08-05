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

doc_data_dir = '/home/mkychsu/pool/TSMC/GraphRAG/'
# doc_list = []
doc_list=[f'{doc_data_dir}dry-etching-technology-for-semiconductors_compress.txt',
          f'{doc_data_dir}plasma-etching-an-introduction_compress.txt',
          f'{doc_data_dir}handbook-of-silicon-wafer-cleaning-technology-third-edition_compress.txt',
          f'{doc_data_dir}Ultraclean Surface Processing of Silicon Wafers - PDF Free Download.txt',
          f'{doc_data_dir}Atomic Layer Processing_semiconductor.txt'   
]

doc_list_all=sorted(glob.glob(f'{doc_data_dir}/*.txt'))

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

for i, doc in enumerate(doc_list_all):
    if doc in doc_list:
        continue
    try:
        temp_doc = doc_list_all[i+1]
        sim = similar(temp_doc.lower(), doc.lower())
        if sim < 0.9:
            doc_list.append(doc)
        else:
            if abs(os.stat(doc).st_size - os.stat(temp_doc).st_size)/os.stat(doc).st_size < 1e-3:
                print(f'{i}:{sim},\n {doc} \n {temp_doc}')
            else:
                doc_list.append(doc)
    except:
        pass
print(len(doc_list),doc_list[0])


# In[4]:


# import glob

# doc_data_dir = '/home/mkychsu/pool/TSMC/dataset/'
# doc_list=[f'{doc_data_dir}dry-etching-technology-for-semiconductors_compress.pdf',
#           f'{doc_data_dir}plasma-etching-an-introduction_compress.pdf',
#           f'{doc_data_dir}handbook-of-silicon-wafer-cleaning-technology-third-edition_compress.pdf',
#           f'{doc_data_dir}Ultraclean Surface Processing of Silicon Wafers - PDF Free Download.pdf',
#           f'{doc_data_dir}Atomic Layer Processing_semiconductor.pdf'   
# ]

# doc_list_all=sorted(glob.glob(f'{doc_data_dir}*.pdf'))

# from difflib import SequenceMatcher

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# for i, doc in enumerate(doc_list_all):
#     if doc in doc_list:
#         continue
#     try:
#         temp_doc = doc_list_all[i+1]
#         sim = similar(temp_doc.lower(), doc.lower())
#         if sim < 0.9:
#             doc_list.append(doc)
#         else:
#             if abs(os.stat(doc).st_size - os.stat(temp_doc).st_size)/os.stat(doc).st_size < 1e-3:
#                 print(f'{i}:{sim},\n {doc} \n {temp_doc}')
#             else:
#                 doc_list.append(doc)
#     except:
#         pass
# print(doc_list)


# In[5]:


# file_to_check = doc_list[0].split('/')
# file_to_check[-2] = 'dataset_textbook'
# file_to_check[-1]=f'0.txt'
# file_to_check='/'.join(file_to_check)
# file_to_check


# In[6]:


# if not os.path.exists(file_to_check):
#     from langchain_community.document_loaders import PyPDFium2Loader as PDFLoader
#     for i, doc in enumerate(doc_list[:5]):
#         try:
#             doc_pages = PDFLoader(doc).load_and_split()
#             txt=''
#             for page in doc_pages:
#                 txt += page.page_content.replace('\n', ' ')
#             with open(f'/home/mkychsu/pool/TSMC/dataset_textbook/{i}.txt', 'w') as f:
#                 f.write(f'{txt}')
#                 f.close()
#         except: # Exception as e:
#             pass


# ### Load the LLM and the tokenizer

# In[7]:


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



# In[8]:


# filename = f"{data_dir}/{graph_name}"
# file_path = hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='./')
# print(f"File downloaded at: {file_path}")

# graph_name=f'{data_dir}/{graph_name}'
# G = nx.read_graphml(graph_name)

# repository_id='MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF'
# filename='Mistral-7B-Instruct-v0.3Q8_0.gguf'

repository_id='bartowski/Mistral-7B-Instruct-v0.3-GGUF'
filename='Mistral-7B-Instruct-v0.3-Q8_0.gguf'

file_path = hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='/home/mkychsu/pool/llm')
# file_path = f'{model}/'


# ### Load LLM: clean Mistral 7B

# In[9]:


from llama_cpp import Llama
import llama_cpp

llm = Llama(model_path=file_path,
             n_gpu_layers=-1,verbose= True, #False,#False,
             n_ctx=10000,
             main_gpu=0,
             # chat_format='mistral-instruct',
             )


# In[10]:


def generate_Mistral (system_prompt='You are a semiconductor engineer. Try to find the clear relationship in the provided information', 
                         prompt="How to make silicon into chip?",temperature=0.333,
                         max_tokens=10000, 
                         ):

    if system_prompt==None:
        messages=[
            {"role": "user", "content": prompt},
        ]
    else:
        messages=[
            {"role": "system",  "content": system_prompt, },
            {"role": "user", "content": prompt},
        ]

    result=llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return result['choices'][0]['message']['content']
     


# In[11]:


# q='''Explain how semiconductor is made in a very professional way with as much detail as possible'''
# start_time = time.time()
# res=generate_Mistral( system_prompt='You are an expert in semiconductor fields. Try to find the clear relation in the provided information. Skip the authorship information if it is not relevant', 
#          prompt=q, max_tokens=1024, temperature=0.3,  )

# print (res)
# deltat=time.time() - start_time
# print("--- %s seconds ---" % deltat)
# display (Markdown(res))


# In[13]:


# graph_HTML, graph_GraphML, G, net, output_pdf = make_graph_from_text(res, generate_Mistral,
#                                                                      chunk_size=1000,chunk_overlap=200,
#                                                                      do_distill=True, data_dir='temp', verbatim=True,
#                                                                      repeat_refine=0)


# In[14]:


os.environ['TOKENIZERS_PARALLELISM']='true'

embedding_file='TSMC_KG_mistral_instruct_v0.3.pkl'
file_path = f"{data_dir}/{embedding_file}"

if not os.path.exists(file_path):
    node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
    save_embeddings(node_embeddings, file_path)
    
else:
    # file_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir='./')
    # print(f"File downloaded at: {file_path}")
    node_embeddings = load_embeddings(file_path)

for i, doc in enumerate(doc_list):
    
    graph_root=f'graph_{i}'
    _graph_GraphML= f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'
    txt=''

    if os.path.exists(_graph_GraphML):
        G = nx.read_graphml(_graph_GraphML)
        print(f'Main KG loaded: {_graph_GraphML}, {G}')
        
    elif os.path.exists(f'{i}_err.txt'):
        print(f'No. {i}: {doc} got something wrong.')

    elif os.path.exists(f'{data_dir}/graph_{i}_graph.graphml'):
        print(f'Found a graph fragment to merge: {i}: {doc}.')
        graph_GraphML = f'{data_dir}/graph_{i}_graph.graphml'
    else:
        # file = doc.split('/')
        # file[-2] = 'dataset_textbook'
        # file[-1]=f'{i}.txt'
        # file='/'.join(file)
        print(doc)
        with open(doc, "r") as f:
            txt = " ".join(f.read().splitlines())  # separate lines with a single space

        try:
            _, graph_GraphML, _, _, _ = make_graph_from_text(txt,generate_Mistral,
                                  include_contextual_proximity=False,
                                  graph_root=graph_root,
                                  chunk_size=2000,chunk_overlap=500,
                                  repeat_refine=0,verbatim=False,
                                  data_dir=data_dir,
                                  save_PDF=False,#TO DO
                                  save_HTML=True,
                                 )
        except Exception as e:
            print(f'Something is wrong with No. {i}: {doc}.')
            f = open(f'{i}_err.txt', 'w')
            f.write(f'{e}\n{doc}\n{txt}')
            f.close()         
   
    print(f'Merging graph No. {i}: {doc} to the main one')
    _, G, _, node_embeddings, res = add_new_subgraph_from_text(txt, generate_Mistral,
                           node_embeddings, embedding_tokenizer, embedding_model,
                           original_graph=G, data_dir_output=data_dir_output, graph_root=graph_root,
                           chunk_size=2000,chunk_overlap=500,
                           do_simplify_graph=True,size_threshold=10,
                           repeat_refine=0,similarity_threshold=0.95,
                           do_Louvain_on_new_graph=False, include_contextual_proximity=False,
                           #whether or not to simplify, uses similiraty_threshold defined above
                           return_only_giant_component=False,
                           save_common_graph=False,G_to_add=None,graph_GraphML_to_add=graph_GraphML,
                           verbatim=True,)

        save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')


