#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl, os
from urllib import request
from urllib.error import HTTPError
from time import sleep
import argparse


def output_list(handle, url, domainid):
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()

  next = url
  last_page = False

  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        #no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        sleep(61)
        continue
      else:
        # If there is a different HTTP error, it wil re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          sleep(61)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e

    for i, item in enumerate(payload["results"]):
      
      handle.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR + domainid + HEADER_SEPARATOR + item["metadata"]["name"] + "\n")

      seq = item["extra_fields"]["sequence"]
      # fastaSeqFragments = [seq[0+i:LINE_LENGTH+i] for i in range(0, len(seq), LINE_LENGTH)]
      # for fastaSeqFragment in fastaSeqFragments:
      #   handle.write(fastaSeqFragment + "\n")

      handle.write(seq + "\n")
      
    # Don't overload the server, give it time before asking for more
    if next:
      sleep(1)

if __name__ == "__main__":
  
  BASE_URL_PRO = "https://www.ebi.ac.uk/interpro/wwwapi/protein/UniProt/taxonomy/uniprot/2" # Prokaryotic
  BASE_URL_EUK = "https://www.ebi.ac.uk/interpro/wwwapi/protein/UniProt/taxonomy/uniprot/2759" # Eukaryotic
  HEADER_SEPARATOR = "|"

  # ACP-like superfamily IPR036736
  URL_EXTENSION = "/entry/InterPro/IPR036736/?page_size=200&extra_fields=sequence"
  OUTPUT_FILE = "IPR036736.faa"

  # # Alternatively use IPR009081 for ACP
  # URL_EXTENSION = "/entry/InterPro/IPR009081/?page_size=200&extra_fields=sequence"
  # OUTPUT_FILE = "IPR009081.faa"

  parser = argparse.ArgumentParser(description="Download sequences from InterPro")
  parser.add_argument("--output", type=str, default=OUTPUT_FILE)
  parser.add_argument("--url_extension", type=str, default=URL_EXTENSION)
  args = parser.parse_args()

  if os.path.isfile(args.output):
    print(args.output + " already exists, please remove or rename it before running this script")
    sys.exit(errno.EEXIST)
  else:
    open(args.output, 'a').close()
  
  with open(args.output, "a") as output_handle:

    url = BASE_URL_EUK + args.url_extension
    output_list(output_handle, url, domainid="2759")
    url = BASE_URL_PRO + args.url_extension
    output_list(output_handle, url, domainid="2")
