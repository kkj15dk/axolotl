import torch
import argparse
from typing import Optional, List, Union

from transformers import PreTrainedTokenizerFast

from .load_model import load_model
from . import sampling
from .utils import float_or_testing


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--predictor", type=str, default="analytic", choices=sampling._PREDICTORS.keys())
    parser.add_argument("--denoise", type=bool, default=True)
    parser.add_argument("--cfg_w", type=float_or_testing, default=1.0)
    parser.add_argument("--label", type=str, default=None, choices=['prokaryotic', 'eukaryotic', 'random'])
    parser.add_argument("--prefix", type=Optional[str], default="[")
    parser.add_argument("--suffix", type=Optional[str], default="]")
    parser.add_argument("--input", type=Optional[str], default=None)
    parser.add_argument("--input_locations", type=Optional[List[int]], default=None)
    parser.add_argument("--output", type=str, default="samples.txt")
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_absorb')

    # more generally commands can be defined with something like below:
    # input = "AFGKLM"
    # input_locs = [5, 6, 19, 20, 1000, 10001]
    if args.input is not None: # If input is provided, use it instead of prefix and suffix
        assert args.input_locations is not None, "If input is provided, input_locations must be provided as well"
        print("Using input and input_locations instead of prefix and suffix")
        input_ids = tokenizer(args.input)['input_ids'].squeeze(0)
        input_locs = args.input_locations
    else: # If input is not provided, use prefix and suffix for conditional sampling
        prefix_ids = tokenizer(args.prefix)['input_ids'].squeeze(0)
        suffix_ids = tokenizer(args.suffix)['input_ids'].squeeze(0)
        input_ids = prefix_ids + suffix_ids
        input_locs = list(range(len(prefix_ids))) + list(range(args.length-len(suffix_ids), args.length))
    
    assert len(input_ids) <= args.length, "The sum of the prefix and suffix is longer than the maximum length"


    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    

    sampling_fn = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(args.batch_size, args.length), 
        predictor=args.predictor,
        steps=args.steps, 
        denoise=args.denoise,
        proj_fun=proj_fun,
        cfg=args.cfg_w,
        label=args.label,
        num_labels=model.num_labels,
        device=device
    )

    samples, sampling_label, sampling_cfg_w = sampling_fn(model)
    samples = proj_fun(samples)
    sequences = tokenizer.batch_decode(samples)

    with open(args.output, "a") as file:
        for i, seq in enumerate(sequences):
            if sampling_label[i] == 0:
                sequence_label = "prokaryotic"
            elif sampling_label[i] == 1:
                sequence_label = "eukaryotic"
            else:
                raise ValueError(f"Invalid label: {sampling_label[i]}")
            w = sampling_cfg_w[i].item()
            
            file.write(f">{i} | label: {sequence_label} | cfg_w: {w}\n")
            file.write(seq + "\n")

if __name__=="__main__":
    main()