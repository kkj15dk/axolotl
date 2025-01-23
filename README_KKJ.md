### Installation

nested tensors need newest nightly version of torch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

In addition to the offsets metadata, NJTs can also compute and cache the minimum and maximum sequence lengths for its components, which can be useful for invoking particular kernels (e.g. SDPA). There are currently no public APIs for accessing these, but this will change for the beta release.
https://docs-preview.pytorch.org/pytorch/pytorch/145402/nested.html