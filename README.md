# GraphReasoning: Scientific Discovery through Knowledge Extraction and Multimodal Graph-based Representation and Reasoning

Markus J. Buehler, MIT, 2024
mbuehler@MIT.EDU

Leveraging generative Artificial Intelligence (AI), we have transformed a dataset comprising 1,000 scientific papers into an ontological knowledge graph. Through an in-depth structural analysis, we have calculated node degrees, identified communities and connectivities, and evaluated clustering coefficients and betweenness centrality of pivotal nodes, uncovering fascinating knowledge architectures. The graph has an inherently scale-free nature, is highly connected, and can be used for graph reasoning by taking advantage of transitive and isomorphic properties that reveal unprecedented interdisciplinary relationships that can be used to answer queries, identify gaps in knowledge, propose never-before-seen material designs, and predict material behaviors. We compute deep node embeddings for combinatorial node similarity ranking for use in a path sampling strategy links dissimilar concepts that have previously not been related. One comparison revealed structural parallels between biological materials and Beethoven's 9th Symphony, highlighting shared patterns of complexity through isomorphic mapping. In another example, the algorithm proposed a hierarchical mycelium-based composite based on integrating path sampling with principles extracted from Kandinsky's 'Composition VII' painting. The resulting material integrates an innovative set of concepts that include a balance of chaos/order, adjustable porosity, mechanical strength, and complex patterned chemical functionalization. We uncover other isomorphisms across science, technology and art, revealing a nuanced ontology of immanence that reveal a context-dependent heterarchical interplay of constituents. Graph-based generative AI achieves a far higher degree of novelty, explorative capacity, and technical detail, than conventional approaches and establishes a widely useful framework for innovation by revealing hidden connections.
This library provides all codes and libraries used in the paper: https://arxiv.org/abs/2403.11996

![image](https://github.com/lamm-mit/GraphReasoning/assets/101393859/3baa3752-8222-4857-a64c-c046693d6315)

# Installation and Examples

Install directly from GitHub:
```
pip install git+https://github.com/lamm-mit/GraphReasoning
```
Or, editable:
```
pip install -e git+https://github.com/lamm-mit/GraphReasoning
```
Install X-LoRA, if needed:
```
pip install git+https://github.com/EricLBuehler/xlora.git
```
You may need wkhtmltopdf for the multi-agent model:
```
sudo apt-get install wkhtmltopdf
```
If you plan to use llama.cpp, install using:
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on " pip install  'git+https://github.com/abetlen/llama-cpp-python.git#egg=llama-cpp-python[server]' --force-reinstall --upgrade --no-cache-dir
```
Model weights and other data: 

[lamm-mit/GraphReasoning
](https://huggingface.co/lamm-mit/GraphReasoning/tree/main)

 
