#!/usr/bin/env python
# coding: utf-8

# # GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning
# 
# Markus J. Buehler, MIT, 2024 mbuehler@MIT.EDU
# 
# ### Example: GraphReasoning: Loading graph and graph analysis

# In[16]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device='cuda:0'

from tqdm.notebook import tqdm
from IPython.display import display, Markdown
from huggingface_hub import hf_hub_download
from GraphReasoning import *


# In[17]:


verbatim=False


# ### Load dataset

# In[18]:


import glob

doc_data_dir = '/home/mkychsu/pool/TSMC/dataset_textbook/'
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


# In[19]:


# 


# In[20]:


# import glob

# doc_data_dir = '/home/mkychsu/pool/TSMC/dataset_textbook_txt/'
# # doc_list = []
# doc_list=[f'{doc_data_dir}dry-etching-technology-for-semiconductors_compress/dry-etching-technology-for-semiconductors_compress.md',
#           f'{doc_data_dir}plasma-etching-an-introduction_compress/plasma-etching-an-introduction_compress.md',
#           f'{doc_data_dir}handbook-of-silicon-wafer-cleaning-technology-third-edition_compress/handbook-of-silicon-wafer-cleaning-technology-third-edition_compress.md',
#           f'{doc_data_dir}Ultraclean Surface Processing of Silicon Wafers - PDF Free Download/Ultraclean Surface Processing of Silicon Wafers - PDF Free Download.md',
#           f'{doc_data_dir}Atomic Layer Processing_semiconductor/Atomic Layer Processing_semiconductor.md'   
# ]

# doc_list_all=sorted(glob.glob(f'{doc_data_dir}*/*.md'))

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
# print(len(doc_list),doc_list[0])


# In[ ]:





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
# filename='Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf'

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
                         max_tokens=8192, stream = True
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
        # return generate_Mistral( system_prompt=system_prompt, prompt=prompt[:len(prompt)//2+100], temperature=temperature, max_tokens=max_tokens) + \
        #       generate_Mistral( system_prompt=system_prompt, prompt=prompt[len(prompt)//2-100:], temperature=temperature, max_tokens=max_tokens)


# In[11]:


import time
q='''Explain how semiconductor is made in a very professional way with as much detail as possible'''
start_time = time.time()
res=generate_Mistral( system_prompt='You are an expert in semiconductor fields. Try to find the clear relation in the provided information. Skip the authorship information if it is not relevant', 
         prompt=q, max_tokens=1024, temperature=0.3,  )

deltat=time.time() - start_time
print("--- %s seconds ---" % deltat)
display (Markdown(res))


# In[12]:


graph_HTML, graph_GraphML, G, net, output_pdf = make_graph_from_text(res, generate_Mistral,
                                                                     chunk_size=1000,chunk_overlap=200,
                                                                     do_distill=True, data_dir='temp', verbatim=True,
                                                                     repeat_refine=0)


# In[13]:


os.environ['TOKENIZERS_PARALLELISM']='true'

embedding_file='TSMC_KG_mistral_instruct_v0.3.pkl'
generate_new_embeddings=True

if os.path.exists(f'{data_dir}/{embedding_file}'):
    generate_new_embeddings=False

if generate_new_embeddings:
    node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )
    save_embeddings(node_embeddings, f'{data_dir}/{embedding_file}')
    
else:
    filename = f"{data_dir}/{embedding_file}"
    # file_path = hf_hub_download(repo_id=repository_id, filename=filename, local_dir='./')
    # print(f"File downloaded at: {file_path}")
    node_embeddings = load_embeddings(f'{data_dir}/{embedding_file}')


# In[14]:


doc_list[:5]


# In[15]:


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
        _, G, _, node_embeddings, res = add_new_subgraph_from_text('', generate_Mistral,
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
        #     pass
        
    else:
        # break

        
        print(f'Generating a knowledge graph from {doc}')
        with open(doc, "r") as f:
            txt = " ".join(f.read().splitlines())  # separate lines with a single space

        try:
            _, graph_GraphML, _, _, _ = make_graph_from_text(txt,generate_Mistral,
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


import numpy as np

while doc_list != []:
    doc = np.random.choice(doc_list)   
    i = doc_list.index(doc)
    
    title = doc.split('/')[-1].split('.pdf')[0]
    doc = doc.split('/')
    doc[-2]+=f'_txt'
    doc[-1]=title+f'/{title}.md'
    doc='/'.join(doc)
    
    title = doc.split('/')[-1].split('.md')[0]
    graph_root = f'{title}'
    print(f'{doc}')
    if os.path.exists(f'{title}.txt'):
        print(f'No. {i}: {title} has been read')
        doc_list.pop(i)
        continue
    
    if os.path.exists(f'{title}_err.txt'):
        print(f'No. {i}: {title} got something wrong.')
        doc_list.pop(i)
        continue
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




# In[ ]:


# doc = doc_list[0]
# title = doc.split('/')[-1].split('.pdf')[0]
# graph_root = f'{title}'

G = nx.read_graphml(f'{data_dir_output}/graph_30_augmented_graphML_integrated.graphml')
print(f'KG loaded: {G}')
# node_embeddings = generate_node_embeddings(G, embedding_tokenizer, embedding_model, )


# In[ ]:





# In[ ]:


# def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
#     chunks = []
#     for document in documents:
#         for i in range(0, len(document), chunk_size - overlap_size):
#             chunk = document[i:i + chunk_size]
#             chunks.append(chunk)
#     return chunks

# def extract_elements_from_chunks(chunks):
#     elements = []
#     for index, chunk in enumerate(chunks):
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Extract entities and relationships from the following text."},
#                 {"role": "user", "content": chunk}
#             ]
#         )
#         entities_and_relations = response.choices[0].message.content
#         elements.append(entities_and_relations)
#     return elements

# def summarize_elements(elements):
#     summaries = []
#     for index, element in enumerate(elements):
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Summarize the following entities and relationships in a structured format. Use \"->\" to represent relationships, after the \"Relationships:\" word."},
#                 {"role": "user", "content": element}
#             ]
#         )
#         summary = response.choices[0].message.content
#         summaries.append(summary)
#     return summaries

# def build_graph_from_summaries(summaries):
#     G = nx.Graph()
#     for summary in summaries:
#         lines = summary.split("\n")
#         entities_section = False
#         relationships_section = False
#         entities = []
#         for line in lines:
#             if line.startswith("### Entities:") or line.startswith("**Entities:**"):
#                 entities_section = True
#                 relationships_section = False
#                 continue
#             elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
#                 entities_section = False
#                 relationships_section = True
#                 continue
#             if entities_section and line.strip():
#                 entity = line.split(".", 1)[1].strip() if line[0].isdigit() and line[1] == "." else line.strip()
#                 entity = entity.replace("**", "")
#                 entities.append(entity)
#                 G.add_node(entity)
#             elif relationships_section and line.strip():
#                 parts = line.split("->")
#                 if len(parts) >= 2:
#                     source = parts[0].strip()
#                     target = parts[-1].strip()
#                     relation = " -> ".join(parts[1:-1]).strip()
#                     G.add_edge(source, target, label=relation)
#     return G

def detect_communities(graph):
    # communities = []
#     for component in nx.weakly_connected_components(graph):
#         subgraph = graph.subgraph(component)
#         if len(subgraph.nodes) > 1:
#             try:
#                 # sub_communities = algorithms.leiden(subgraph)
#                 sub_communities = nx.community.girvan_newman(subgraph)
                
#                 # for community in sub_communities.communities:
#                 for community in tqdm(sub_communities):
#                     communities.append(list(community))
       
#                 communities = sorted(map(sorted, next_level_communities))
#             except Exception as e:
#                 print(f"Error processing community: {e}")
#         else:
#             communities.append(list(subgraph.nodes))

    communities_generator = nx.community.girvan_newman(G)
    next_level_communities = next(communities_generator)
    # next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities), key = lambda x: -len(x) )
    return communities

def summarize_communities(communities, graph, generate):
    community_summaries = []
    for index, community in tqdm(enumerate(communities)):
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['title']} -> {edge[1]}")
        description += ", ".join(relationships)
        # try:
        response = generate(system_prompt= "Summarize the following community of entities and relationships.",
                                       prompt= description)
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "Summarize the following community of entities and relationships."},
        #         {"role": "user", "content": description}
        #     ]
        # )
        # summary = response.choices[0].message.content.strip()
        # except:
        print(description)
        summary = response.strip()
        community_summaries.append(summary)
    return community_summaries

def generate_answers_from_communities(community_summaries, generate, query):
    intermediate_answers = []
    for summary in tqdm(community_summaries):
        try:
            response = generate(system_prompt= "Answer the following query based on the provided summary.",
                                       prompt=f"Query: {query} Summary: {summary}")
            # response = client.chat.completions.create(
            #     model="gpt-4",
            #     messages=[
            #         {"role": "system", "content": "Answer the following query based on the provided summary."},
            #         {"role": "user", "content": f"Query: {query} Summary: {summary}"}
            #     ]
            # )
            intermediate_answers.append(response)
        except:
            print(f'TL;DR: {summary[0:100]}...{summary[-100:]}')
            return 0
    final_response = generate(system_prompt= "Combine these answers into a final, concise response.",
                                prompt=f"Intermediate answers: {' '.join(intermediate_answers)}")

    # final_response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "Combine these answers into a final, concise response."},
    #         {"role": "user", "content": }
    #     ]
    # )
    # final_answer = final_response.choices[0].message.content
    return final_response

# def graph_rag_pipeline(documents, query, chunk_size=600, overlap_size=100):
def graph_rag_pipeline(graph, generate, query):
    # chunks = split_documents_into_chunks(documents, chunk_size, overlap_size)
    # elements = extract_elements_from_chunks(chunks)
    # summaries = summarize_elements(elements)
    # graph = build_graph_from_summaries(summaries)
    
    communities = detect_communities(graph)
    if verbatim:
        print("Number of Communities = ", len(communities))
    community_summaries = summarize_communities(communities, graph, generate)
    final_answer = generate_answers_from_communities(community_summaries, generate, query)
    return final_answer



# In[ ]:


graph=G
generate = generate_Mistral
communities = detect_communities(graph)


# In[ ]:


community_summaries = summarize_communities(communities, graph, generate)


# In[ ]:


query = "What are the main techniques to make semiconductors?"

last_response=''
for i, summary in tqdm(enumerate(community_summaries)):
    response = generate(system_prompt= "Answer the query detailedly based on the collected information and the combined with the last thought you have. ",
                               prompt=f"Query: {query} Collected information: {summary} You last thought: {last_response}")
    last_response=response
    print(last_response)


# In[ ]:


last_response


# In[ ]:


final_response = generate(system_prompt= "Combine these answers into a final, concise response.",
                            prompt=f" answers: {last_response}")


# In[ ]:





# In[ ]:


response, (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML = find_path_and_reason(
    G, 
    node_embeddings,
    embedding_tokenizer, 
    embedding_model, 
    generate_Mistral, 
    data_dir=data_dir_output,
    verbatim=verbatim,
    include_keywords_as_nodes=True,  # Include keywords in the graph analysis
    keyword_1="Temperature",
    keyword_2="Semiconductors",
    N_limit=9999,  # The limit for keywords, triplets, etc.
    instruction='What is the best temperature when manufacturing semiconductors.',
    keywords_separator=', ',
    graph_analysis_type='nodes and relations',
    temperature=0.3, 
    inst_prepend='### ',  # Instruction prepend text
    prepend='''You are given a set of information from a graph that describes the relationship 
               between materials and manufacturing process. You analyze these logically 
               through reasoning.\n\n''',  # Prepend text for analysis
    visualize_paths_as_graph=True,  # Whether to visualize paths as a graph
    display_graph=True,  # Whether to display the graph
)
display(Markdown(response))


# In[ ]:





# In[ ]:


path


# In[ ]:


visualize_embeddings_2d_pretty_and_sample(node_embeddings, n_clusters=10, n_samples=10, data_dir=data_dir_output, alpha=.7)


# In[ ]:


# describe_communities_with_plots_complex(G, N=6, data_dir=data_dir_output)


# In[ ]:


# graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir_output,include_centrality=False,
                                               # make_graph_plot=False,)


# In[ ]:


is_scale_free (G, data_dir=data_dir_output)


# In[ ]:


# find_best_fitting_node_list("semiconductor", node_embeddings, embedding_tokenizer, embedding_model, 5)


# In[ ]:


# find_best_fitting_node_list("better manufactoring process for semiconductor", node_embeddings , embedding_tokenizer, embedding_model, 5)


# In[ ]:





# In[20]:


(best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML=find_path( G, node_embeddings,
                                embedding_tokenizer, embedding_model , second_hop=False, data_dir=data_dir_output,
                                  keyword_1 = "new materials", keyword_2 = "semiconductor",
                                      similarity_fit_ID_node_1=0, similarity_fit_ID_node_2=0,
                                       )



# In[ ]:





# In[ ]:


# path


# In[ ]:


# path_list, path_string=print_path_with_edges_as_list(G , path)
# path_list,path_string


# In[ ]:


# visualize_paths_pretty([path_list], 'knowledge_graph_paths.svg', display_graph=True,data_dir=data_dir_output, scale=0.75)


# In[ ]:


# triplets=find_all_triplets(path_graph) 


# In[ ]:


# triplets


# In[ ]:





# In[ ]:




