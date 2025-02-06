import Bio
from Bio import SeqIO
import os
import argparse

def convert_folder_to_csv(folder_path, output_path, filename):

    '''
    Input:
    A folder with fasta files.
    '''
    with open(f'{output_path}{filename}', 'w') as output_handle:
        output_handle.write("clusterid,proteinid,domainid,sequence\n") # clusterid should be the first to be able to sort the csv file
        clusters = os.listdir(folder_path)
        print(f"Found {len(clusters)} clusters")
        for file in clusters:
            for record in SeqIO.parse(f"{folder_path}{file}", "fasta"):
                clusterid = file.split('.')[0]
                accession, domainid, name = record.id.split('|')
                output_handle.write(f"{clusterid},{accession},{domainid},{record.seq}\n")

# %%
if __name__ == "__main__":
    
    INPUT = '/datasets/test/'
    OUTPUT = '/datasets/'
    OUTFILE = 'test.csv'

    parser = argparse.ArgumentParser(description="Convert a folder of fasta files to a csv file")
    parser.add_argument("--input", default=INPUT, type=str)
    parser.add_argument("--output", default=OUTPUT, type=str)
    parser.add_argument("--outfile", default=OUTFILE, type=str)

    convert_folder_to_csv(parser.input, parser.output, parser.outfile)