import Bio
from Bio import SeqIO
import os
import argparse

def convert_fasta_to_csv(fasta_path, output_path, filename):

    '''
    Input:
    A fasta file path
    '''
    with open(f'{output_path}{filename}', 'w') as output_handle:
        output_handle.write("clusterid,proteinid,domainid,sequence\n") # clusterid should be the first to be able to sort the csv file

        for i, record in enumerate(SeqIO.parse(f"{fasta_path}", "fasta")):
            clusterid = i + 1
            
            domainid = record.description.split('|Domain:')[-1]
            if domainid == "Bacteria":
                domainid = 2
            elif domainid == "Eukaryota":
                domainid = 2759
            elif domainid == "Archaea":
                domainid = 2157
            else:
                raise ValueError(f"Unknown domain: {domainid}")

            accession = record.id
            output_handle.write(f"{clusterid},{accession},{domainid},{record.seq}\n")

# %%
if __name__ == "__main__":
    
    INPUT = '/mnt/e/uniref50_with_domain.fasta'
    OUTPUT = '/mnt/e/'
    OUTFILE = 'uniref50_unclustered.csv'

    parser = argparse.ArgumentParser(description="Convert a folder of fasta files to a csv file")
    parser.add_argument("--input", default=INPUT, type=str)
    parser.add_argument("--output", default=OUTPUT, type=str)
    parser.add_argument("--outfile", default=OUTFILE, type=str)

    args = parser.parse_args()

    convert_fasta_to_csv(args.input, args.output, args.outfile)