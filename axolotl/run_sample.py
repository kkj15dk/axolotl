import torch
import argparse

from transformers import PreTrainedTokenizerFast
import torch.nn.functional as F

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
    parser.add_argument("--output", type=str, default="samples.txt")
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_absorb')

    sampling_fn = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(args.batch_size, args.length), 
        predictor=args.predictor,
        steps=args.steps, 
        denoise=args.denoise,
        cfg=args.cfg_w,
        label=args.label,
        num_labels=model.num_labels,
        device=device
    )

    samples, sampling_label, sampling_cfg_w = sampling_fn(model)
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