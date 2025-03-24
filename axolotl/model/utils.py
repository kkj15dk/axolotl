import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, t, label):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            t: A mini-batch of time steps.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, t, label)

    return model_fn


def get_output_fn(model, train=False, exponentiate=False, use_cfg=False, num_labels=None):

    model_fn = get_model_fn(model, train=train)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        def output_fn(x, t, label):

            if use_cfg: # use classifier-free guidance for sampling
                assert not train, "Must be sampling when using cfg"
                assert num_labels is not None, "Must provide num_labels if using cfg"
                x = torch.cat([x, x], dim=0)
                t = torch.cat([t, t], dim=0)
                uncond = torch.ones_like(label, dtype=torch.long) * num_labels # assume that the uncond label is the last label
                label = torch.cat([label, uncond], dim=0)

            output = model_fn(x, t, label)
            
            if exponentiate:
                # when sampling return true score (not log used for training)
                return output.exp()
                
            return output

    return output_fn
