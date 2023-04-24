import argparse
from transformers import BertTokenizer,RobertaTokenizer
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--input', type=str, default='data/text/wikitext/train.txt')
    parser.add_argument('--model_token', type=str, default='qtbert')
    parser.add_argument('--attribute_female', type=str, default='data/female.txt')
    parser.add_argument('--attribute_male', type=str, default='data/male.txt')
    parser.add_argument('--target_stereotype', type=str, default='data/stereotype.txt')
    parser.add_argument('--output', type=str, default='data/tensor/')
    args = parser.parse_args()

    return args

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_token, max_len=args.max_len)
    with open(args.input, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')
    batch = tokenizer.batch_encode_plus(lines[:100], padding=True,truncation=True)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    target_words = []
    attribute_words = []
    with open(args.attribute_female, 'r') as f:
        contents = f.read()
        attribute_words += contents.replace('\n', ' ').split()
    with open(args.attribute_male, 'r') as f:
        contents = f.read()
        attribute_words += contents.replace('\n', ' ').split()
    with open(args.target_stereotype, 'r') as f:
        contents = f.read()
        target_words += contents.replace('\n', ' ').split()
    attribute_words_tokenizer_id = tokenizer.encode(attribute_words, add_special_tokens=False)
    target_words_tokenizer_id = tokenizer.encode(target_words, add_special_tokens=False)
    all_special_ids = torch.tensor(tokenizer.all_special_ids)

    attribute_words_tokenizer_id = torch.tensor(attribute_words_tokenizer_id)
    target_words_tokenizer_id = torch.tensor(target_words_tokenizer_id)

    labels = torch.tensor(input_ids)
    mask = torch.tensor(attention_mask)
    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    qt_mask = torch.zeros_like(input_ids)
    for i in range(input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        do_qt = False
        for mask_id in input_ids[i, selection]:
            if mask_id in target_words_tokenizer_id and mask_id not in all_special_ids:
                do_qt = True
        if do_qt:
            for j in range(input_ids[i].shape[0]):
                if input_ids[i][j] in attribute_words_tokenizer_id and input_ids[i][j] not in input_ids[i][j]:
                    qt_mask[i, j] = 1
        input_ids[i, selection] = tokenizer.mask_token_id
    torch.save(input_ids, f'{args.output}input_ids.pt')
    torch.save(mask, f'{args.output}attention_mask.pt')
    torch.save(labels, f'{args.output}labels.pt')
    torch.save(qt_mask, f'{args.output}qt_mask.pt')

if __name__ == "__main__":
    args = parse_args()
    main(args)