import torch
import argparse
from typing import Optional, List, Union

from transformers import PreTrainedTokenizerFast

from axolotl import sampling
from axolotl.load_model import load_model
from axolotl.utils import float_or_testing


def get_args():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024) # remember to add 2 for prefix and suffix
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--predictor", type=str, default="analytic", choices=sampling._PREDICTORS.keys())
    parser.add_argument("--denoise", type=bool, default=True)
    parser.add_argument("--cfg_w", type=float_or_testing, default=1.0)
    parser.add_argument("--label", type=str, default=None, choices=['prokaryotic', 'eukaryotic', 'random'])
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_locations", type=List[int], default=None)
    parser.add_argument("--output", type=str, default="samples_cond.txt")
    parser.add_argument("--name", type=str, default="cond_sample")
    args = parser.parse_args()
    return args

def sample_conditional(model_path: str,
                       tokenizer: PreTrainedTokenizerFast,
                       input: str,
                       input_locations: List[int],
                       length: int,
                       batch_size: int = 1,
                       steps: int = 1024,
                       predictor: str = "analytic",
                       denoise: bool = True,
                       cfg_w: Union[float, str] = 1.0,
                       label: str = None,
                       output: str = "samples_cond.fasta",
                       name: str = "cond_sample",
):

    # more generally commands can be defined with something like below:
    # input = "AFGKLM"
    # input_locations = [5, 6, 19, 20, 1000, 10001]

    input_ids = tokenizer(input)['input_ids'][1:-1]

    assert len(input_ids) <= length, "Input length must be less than or equal to the specified length"
    assert max(input_locations) < length, "Input locations must be less than the specified length"

    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(batch_size, 1)
    batch_dims = (batch_size, length)

    def proj_fun(x):
        x[:, input_locations] = input_ids
        return x

    device = torch.device('cuda')
    model, graph, noise = load_model(model_path, device)

    sampling_fn = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims, 
        predictor=predictor,
        steps=steps, 
        denoise=denoise,
        proj_fun=proj_fun,
        cfg=cfg_w,
        label=label,
        num_labels=model.num_labels,
        device=device,
        use_tqdm=True,
    )

    samples, sampling_label, sampling_cfg_w = sampling_fn(model)
    samples = proj_fun(samples)
    sequences = tokenizer.batch_decode(samples)

    sampling.write_samples(output=output, sequences=sequences, sampling_label=sampling_label, sampling_cfg_w=sampling_cfg_w, name=name, steps=steps)

if __name__=="__main__":
    args = get_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_absorb')

    if args.input is not None: # If input is provided, use it instead of prefix and suffix
        assert args.input_locations is not None, "If input is provided, input_locations must be provided as well"
        print("Using input and input_locations instead of prefix and suffix")
        input_sequence = args.input
        input_locs = args.input_locations
    else: # If input is not provided, use prefix and suffix for conditional sampling
        assert args.prefix is not None or args.suffix is not None, "If input is not provided, prefix and/or suffix must be provided"
        input_sequence = args.prefix + args.suffix
        input_locs = list(range(len(args.prefix))) + list(range(args.length-len(args.suffix), args.length))

    sample_conditional(
        model_path=args.model_path,
        tokenizer=tokenizer,
        input=input_sequence,
        input_locations=input_locs,
        length=args.length,
        batch_size=args.batch_size,
        steps=args.steps,
        predictor=args.predictor,
        denoise=args.denoise,
        cfg_w=args.cfg_w,
        label=args.label,
        output=args.output,
        name=args.name,
    )