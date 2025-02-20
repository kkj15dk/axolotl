import torch
import argparse

from transformers import PreTrainedTokenizerFast
import torch.nn.functional as F

from axolotl import sampling
from axolotl.load_model import load_model
from axolotl.utils import float_list_or_testing

from typing import List, Union


def get_args():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--predictor", type=str, default="analytic", choices=sampling._PREDICTORS.keys())
    parser.add_argument("--denoise", type=bool, default=True)
    parser.add_argument("--cfg_w", type=float_list_or_testing, default=1.0)
    parser.add_argument("--label", type=str, default='random', choices=['prokaryotic', 'eukaryotic', 'random'])
    parser.add_argument("--output", type=str, default="samples.txt")
    parser.add_argument("--name", type=str, default="sample")
    args = parser.parse_args()
    return args

def sample(model_path: str,
           tokenizer: PreTrainedTokenizerFast,
           batch_size: int,
           length: int,
           steps: int,
           predictor: str,
           denoise: bool,
           cfg_w: Union[float, List[float], str],
           label: str,
           output: str,
           name: str,
):
    
    device = torch.device('cuda')
    model, graph, noise = load_model(model_path, device)

    sampling_fn = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(batch_size, length), 
        predictor=predictor,
        steps=steps, 
        denoise=denoise,
        cfg=cfg_w,
        label=label,
        num_labels=model.num_labels,
        device=device,
        use_tqdm=True,
    )

    samples, sampling_label, sampling_cfg_w = sampling_fn(model)
    sequences = tokenizer.batch_decode(samples)

    sampling.write_samples(output=output, sequences=sequences, sampling_label=sampling_label, sampling_cfg_w=sampling_cfg_w, steps=steps, name=name)


if __name__=="__main__":
    args = get_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_absorb')

    sample(model_path=args.model_path,
           tokenizer=tokenizer,
           batch_size=args.batch_size,
           length=args.length,
           steps=args.steps,
           predictor=args.predictor,
           denoise=args.denoise,
           cfg_w=args.cfg_w,
           label=args.label,
           output=args.output,
           name=args.name,
    )