from dataclasses import dataclass

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing, BertProcessing

from transformers import PreTrainedTokenizerFast

unk_token = "?" # unknown token

vocab = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "?": 20,
    "[": 21,
    "]": 22,
    "-": 23, # masking token, used as absorbing state, not used for uniform tokenizer
}

# tokenizer
tokenizer = Tokenizer(WordLevel(vocab, unk_token=unk_token))
tokenizer.add_special_tokens(["[", "]"])
tokenizer.pre_tokenizer = Split("", behavior='removed')
tokenizer.post_processor = BertProcessing(sep=("]", tokenizer.token_to_id("]")), cls=("[", tokenizer.token_to_id("[")))
tokenizer.decoder = decoders.Fuse()

# tokenizer fast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                         unk_token=unk_token,
                                         clean_up_tokenization_spaces=False,
)

# save the tokenizer
name = "tokenizer_absorb"
# name = "tokenizer_uniform"
fast_tokenizer.save_pretrained(name)
fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(name)

# The encode function
def encode(example):
    return fast_tokenizer(example['sequence'],
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    return_tensors='pt',
    )

# The decode function
def decode(encoded):
    return fast_tokenizer.decode(encoded['input_ids'],
                                 skip_special_tokens=False, # Set to false to not skip unknown tokens ("?")
    )

# test the tokenizer
example = {'sequence': 'ACDEFGHIKLMNPQRSTVWYX'}
encoded = encode(example)
print(encoded)
decoded = decode(encoded)
print(decoded)