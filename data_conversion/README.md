# How to make the datasets

## 1.a UniRef

Using the SPARQL query in UniRefALL_SPARQLquery.txt, a csv file is made. Either from https://sparql.uniprot.org/ or programatically using the script UniRefALL_SPARQL.py, however the script has a tendency to crash.
It contains all UniProtKB protein sequences (reviewed and unreviewed), along with the domainid (Eukaryotic = 2759, Prokaryotic = 2). Only Prokaryotic and Eukaryotic sequences are included. They are filtered to not include Non-terminal residues, and Non-adjacent residues, to ignore fragments.
The headers are: **cluster50id**, **cluster90id**, **cluster100id**, **domainid**, **sequence**
It is very important, that the first headers are the id's, as we sort the file using unix sort.

## 1.b InterPro

Entries from InterPro can be downloaded programatically using the script InterPro_download_by_IPR_domain.py.
I use the entries for IPR036736 to download all proteins with domains in the *ACP-like superfamily*. This includes *PKSs* and *NRPSs*.
The script creates a fasta file of all the entries which we will have to cluster using **CD-HIT**, which is the same algorithm used to cluster UniRef.

### clustering fasta files

installed by "sudo apt install cd-hit"

cd-hit -i **infile** -o **outfile** -c **0.9** -n 5 -d 0 -M 0 -T 0 -g 1
infile, outfile, 0.9 threshold, wordsize of 5, entire defline of fasta, no memory cap, no CPU thread cap, accurate clusters
The threshold for clustering can be changed if that is desired. 0.9 => UniRef90 clustering.

This will create **outfile**.clstr, containing an overview of which entries are in which clusters.

### converting clusters into a csv

This is then converted into a bunch of fasta files, where each fasta file contains the entries of a single cluster, using the perl script make_multi_seq.pl:
make_multi_seq **ACP_by_IPR036736.faa** **90_ACP_by_IPR036736.clstr** multi-seq **1**

**1** signifies, that every cluster must have at least 1 entry, or else it is discarded. This can be set to a higher value, if desired.

Lastly, the folder of cluster fasta files is converted to a csv file using the script convert_folder_to_csv.py

## 2. Sort the csv

For the sorting, the ordering of the csv headers is very important. The first header has to be the id for which to cluster.

Sort the file using UNIX sort:
(head -n 1 <inputfile.csv> && tail -n +2 <inputfile.csv> | sort -u) > <inputfile-sorted.csv>

This sorts the file, ignoring the header, and deletes duplicates.

## 3. Encode into a dataset

The dataset is then encoded using the script convert_csv_to_nested_dataset.py. This script both creates an encoded dataset, and then afterwards creates a grouped dataset, grouped by the clusterid. They are saved to disk as huggingface Datasets.
The grouped dataset is the final one used for model training.
