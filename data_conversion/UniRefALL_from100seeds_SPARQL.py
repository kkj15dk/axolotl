import requests
import time
import concurrent.futures
import logging
import json
import os

OUTPUT = '/mnt/e/UniRefALL_from100seeds'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the SPARQL endpoint and query
endpoint = "https://sparql.uniprot.org/"
query_template = """
PREFIX taxon: <http://purl.uniprot.org/taxonomy/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX up: <http://purl.uniprot.org/core/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT
  (substr(str(?cluster100), 32) AS ?cluster100id)
  (substr(str(?cluster90), 32) AS ?cluster90id)
  (substr(str(?cluster50), 32) AS ?cluster50id)
  (substr(str(?domain), 34) AS ?domainid)
  ?sequence
WHERE {{
  VALUES ?domain {{ taxon:2 taxon:2759 taxon:2157 }}
  ?organism rdfs:subClassOf ?domain .
  
  ?SequenceClass a up:Sequence ;
            up:seedFor ?cluster100 ;
            up:memberOf ?cluster90 ;
            up:memberOf ?cluster50 ;
            rdf:value ?sequence ;
            up:organism ?organism .
  
  ?cluster100 up:identity "1.0"^^xsd:decimal .
  ?cluster90 up:identity "0.9"^^xsd:decimal .
  ?cluster50 up:identity "0.5"^^xsd:decimal .
}}
LIMIT {limit} OFFSET {offset}
"""

# Function to read configuration file
def read_config(config_file="config.json"):
    with open(config_file, "r") as f:
        return json.load(f)

# Function to execute the SPARQL query with retries
def execute_query(query, delay=5, offset=0, timeout=3600):
    headers = {
        "Accept": "text/csv",
        "User-Agent": "s204514@dtu.dk"
    }
    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.get(endpoint, params={"query": query}, headers=headers, timeout=timeout)
            response.raise_for_status()
            logging.info(f"Query successful for offset {offset}")
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed for offset {offset}: {e}")
            time.sleep(delay)

# Function to fetch data for a specific offset
def fetch_data(offset, limit, timeout):
    query = query_template.format(limit=limit, offset=offset)
    return execute_query(query, offset=offset, timeout=timeout)

# Initial read of configuration
config = read_config()
limit = config["limit"]
max_workers = config["max_workers"]
timeout = config["timeout"]


# Initialize offset
offset = 0

# Open a file to write the results
with open(f"{OUTPUT}.csv", "w") as f:

    # Write the header for the first chunk
    f.write("cluster100id,cluster90id,cluster50id,domainid,sequence\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            futures = []
            # Submit tasks to fetch data in parallel
            config = read_config()
            if config["limit"] != limit or config["max_workers"] != max_workers or config["timeout"] != timeout:
                limit = config["limit"]
                max_workers = config["max_workers"]
                timeout = config["timeout"]
                logging.info(f"Updated configuration: limit={limit}, max_workers={max_workers}, timeout={timeout}")
            for i in range(max_workers):
                current_offset = offset + i * limit
                logging.info(f"Submitting query for offset {current_offset}")
                futures.append(executor.submit(fetch_data, current_offset, limit, timeout))
                time.sleep(1)  # Add a delay to avoid overloading the server
            
            all_results_empty = True
            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                
                all_results_empty = False
                
                # Write the result to the file
                f.write(result.split("\n", 1)[1])  # Skip header for subsequent chunks
                
            # Update offset for the next chunk
            offset += max_workers * limit
        
            # Break the loop if all results are empty
            if all_results_empty:
                break

logging.info("Data download complete.")