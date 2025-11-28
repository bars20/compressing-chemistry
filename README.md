# compressing-chemistry

Code for discovering compressing substructures from a dataset of chemicals. The `MML87_multiprocessing.py` file runs a greedy search procedure which finds the maximally compressing substructure at each iteration, and adds it to a list. The search continues until no further substructure can be found which further compresses the data. We use the [Minimum Message Length principle](https://link.springer.com/book/10.1007/0-387-27656-4) to calculate the amount that a set of substructures compresses the dataset.

Original [paper](https://www.arxiv.org/abs/2511.05728)


The CHEMBL_36 file is too large to upload. One may use the .sdf file found at [the official CHEMBL website.](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)

To run the algorithm, run:
`python MML87_multiprocessing.py`
