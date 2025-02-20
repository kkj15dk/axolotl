### Axolotl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo contains the source code for axolotl - a protein diffusion model based on score entropy discrete diffusion


### Installation

Axolotl utilizes nested jagged tensors (NJTs) for training, but they are not nessecary for inference. Nested tensors need a newer nightly version of torch as of writing this (18/02/2025)

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126


### Inference

## unconditional

To do unconditional sampling, run sample.py

```
python sample.py --length LENGTH
```

## conditional

To do unconditional sampling, run sample_cond.py

```
python sample_cond.py --length LENGTH --prefix PREFIX --suffix SUFFIX
```
or
```
python sample_cond.py --length LENGTH --input INPUT --input_locations INPUT_LOCATIONS
```


### Observations

- at about cfg_w of 5, the model starts outputting very similar sequences. Makes sense, it is also seen in images as a tradeoff between IS and FID


### Folding using chai-1 or boltz-1

chai:
chai fold --num-trunk-recycles 0 --num-diffn-timesteps 2 --num-diffn-samples 0 chai_input.fasta chai_output

boltz:
boltz predict temp --use_msa_server --accelerator gpu --num_workers 2 --output_format pdb --out_dir temp_boltz_output


### Acknowledgements

This repository builds heavily off of [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), which in turn is built of off [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).