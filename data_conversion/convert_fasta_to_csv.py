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
            header, domainid = record.id.split('|Domain:')
            accession = header.split(' ')[0]
            output_handle.write(f"{clusterid},{accession},{domainid},{record.seq}\n")

# %%
if __name__ == "__main__":
    
    INPUT = '/mnt/e/uniref50_with_domain.fasta'
    OUTPUT = '/datasets/'
    OUTFILE = 'test.csv'

    parser = argparse.ArgumentParser(description="Convert a folder of fasta files to a csv file")
    parser.add_argument("--input", default=INPUT, type=str)
    parser.add_argument("--output", default=OUTPUT, type=str)
    parser.add_argument("--outfile", default=OUTFILE, type=str)

    convert_fasta_to_csv(parser.input, parser.output, parser.outfile)