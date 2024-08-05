#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis

# In[ ]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device='cuda:0'

from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from huggingface_hub import hf_hub_download
from GraphReasoning import *


# In[ ]:


verbatim=True


# ### Load graph and embeddings 

# In[ ]:


#Hugging Face repo
# repository_id = "lamm-mit/GraphReasoning"
# data_dir='./GRAPHDATA'    

# data_dir_output='./GRAPHDATA_OUTPUT/'

# graph_name='BioGraph.graphml'

# make_dir_if_needed(data_dir)
# make_dir_if_needed(data_dir_output)

tokenizer_model="BAAI/bge-large-en-v1.5"

embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, ) 
embedding_model = AutoModel.from_pretrained(tokenizer_model, ).to('cuda:0') 

# filename = f"{data_dir}/{graph_name}"
# file_path = hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='./')
# print(f"File downloaded at: {file_path}")

# graph_name=f'{data_dir}/{graph_name}'
# G = nx.read_graphml(graph_name)


# In[ ]:

# ### Load LLM: clean Mistral 7B

# In[ ]:


from llama_cpp import Llama
import llama_cpp

# repository_id='SanctumAI/Meta-Llama-3-8B-Instruct-GGUF'
# filename='meta-llama-3-8b-instruct.Q2_K.gguf'

# repository_id='MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF'
# filename='Mistral-7B-Instruct-v0.3.Q8_0.gguf'

repository_id='bartowski/Mistral-7B-Instruct-v0.3-GGUF'
filename='Mistral-7B-Instruct-v0.3-Q8_0.gguf'

file_path = hf_hub_download(repo_id=repository_id, filename=filename,  local_dir='/home/mkychsu/pool/llm')

# chat_format="mistral-instruct"

llm = Llama(model_path=file_path,
             n_gpu_layers=-1,verbose= True, #False,#False,
             n_ctx=10000,
             main_gpu=0,
#              chat_format=chat_format,
             )


# In[ ]:


file_path


# In[ ]:


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
     


# In[ ]:


q='''Explain how semiconductor is made in a very professional way with as much detail as possible'''
start_time = time.time()
res=generate_Mistral( system_prompt='You are an expert in semiconductor fields. Try to find the clear relation in the provided information. Skip the authorship information if it is not relevant', 
         prompt=q, max_tokens=1024, temperature=0.3,  )

print (res)
deltat=time.time() - start_time
print("--- %s seconds ---" % deltat)
display (Markdown(res))


# In[ ]:


data_dir_output = './data_output_KG_TSMC/'


# In[ ]:


graph_HTML, graph_GraphML, G, net, output_pdf = make_graph_from_text(res,generate_Mistral,
                                                                     chunk_size=6000,chunk_overlap=1000,
                                                                     data_dir=data_dir_output,
                                                                     repeat_refine=0)


# In[ ]:

import glob

doc_data_dir = '/home/mkychsu/pool/TSMC/dataset/'
doc_list=[f'{doc_data_dir}dry-etching-technology-for-semiconductors_compress.pdf',
          f'{doc_data_dir}plasma-etching-an-introduction_compress.pdf',
          f'{doc_data_dir}handbook-of-silicon-wafer-cleaning-technology-third-edition_compress.pdf',
          f'{doc_data_dir}Ultraclean Surface Processing of Silicon Wafers - PDF Free Download.pdf',
          f'{doc_data_dir}Atomic Layer Processing_semiconductor.pdf'   
]

doc_list_all=sorted(glob.glob(f'{doc_data_dir}*.pdf'))

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

file_to_check = doc_list[0].split('/')
file_to_check[-2] = 'dataset_textbook'
file_to_check[-1]=f'0.txt'
file_to_check='/'.join(file_to_check)

if not os.path.exists(file_to_check):
    from langchain_community.document_loaders import PyPDFium2Loader as PDFLoader
    for i, doc in enumerate(doc_list[:5]):
        try:
            doc_pages = PDFLoader(doc).load_and_split()
            txt=''
            for page in doc_pages:
                txt += page.page_content.replace('\n', ' ')
            with open(f'/home/mkychsu/pool/TSMC/dataset_textbook/{i}.txt', 'w') as f:
                f.write(f'{txt}')
                f.close()
        except: # Exception as e:
            pass

import np

# for i, doc in enumerate(doc_list):

while doc_list != []:
    doc = np.random.choice(doc_list)
    i = doc_list.index(doc)

    if os.path.exists(f'{doc}.txt'):
        print(f'No. {i}: {doc} has been read')
        doc_list.pop(i)
        continue
    
    if os.path.exists(f'{doc}_err.txt'):
        print(f'No. {i}: {doc} got something wrong.')
        doc_list.pop(i)
        continue

    file = doc.split('/')
    file[-2] = 'dataset_textbook'
    file[-1]=f'{file[-1][-4:]}.txt'
    file='/'.join(file_to_check)

    with open(file, "r") as f:
        txt = " ".join(f.read().splitlines())  # separate lines with a single space

    graph_root=f'graph_{i}'
    try:
        graph_HTML, graph_GraphML, G, net, output_pdf = make_graph_from_text(txt,generate_Mistral,
                              include_contextual_proximity=False,
                              graph_root=graph_root,
                              chunk_size=10000,chunk_overlap=2000,
                              repeat_refine=0,verbatim=False,
                              data_dir=data_dir,
                              save_PDF=False,#TO DO
                              save_HTML=True,
                             )
        f = open(f'{i}.txt', 'w')
        f.write(doc+'\n'+txt)
        f.close()
    except:
        f = open(f'{i}_err.txt', 'w')
        f.write(f'{doc} \n {txt}')
        f.close()       

print('You are ready.')
