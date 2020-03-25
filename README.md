# hybrid-kge

This project is based on code from https://github.com/thunlp/OpenKE and https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding. All datasets (eg. FB15K-237, WN18RR, YAGO3-10) are available in the above links. For Rule mining, we use code from http://resources.mpi-inf.mpg.de/yago-naga/amie/downloads.html 

# Running the Code
 - Project uses the latest version of Pytorch, numpy, sklearn, matplotlib
 - Download AIME+ rule mining code and place the jar file in a folder called mining. 
 - Download KG datasets to a newly created folder.
 - Update parameters.py to specify the dataset and other essential parameters.
 - Run the code using python3 exp_emb_mine1.py
