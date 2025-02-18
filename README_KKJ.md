### Installation

nested tensors need newest nightly version of torch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

In addition to the offsets metadata, NJTs can also compute and cache the minimum and maximum sequence lengths for its components, which can be useful for invoking particular kernels (e.g. SDPA). There are currently no public APIs for accessing these, but this will change for the beta release.
https://docs-preview.pytorch.org/pytorch/pytorch/145402/nested.html

### Observations
- at about cfg_w of 5, the model starts outputting very similar sequences. Makes sense, it is also seen in images as a tradeoff between IS and FID


### Folding using chai-1 or boltz-1

chai:
chai fold --num-trunk-recycles 0 --num-diffn-timesteps 2 --num-diffn-samples 0 chai_input.fasta chai_output

boltz:
boltz predict temp --use_msa_server --accelerator gpu --num_workers 2 --output_format pdb --out_dir temp_boltz_output