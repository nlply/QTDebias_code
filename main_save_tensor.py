from datasets import load_dataset
from tqdm.auto import tqdm
import torch

from transformers import BertTokenizer

# 28996
# tokenizer = BertTokenizer.from_pretrained('qtbert', max_len=512)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)
tokens = tokenizer('I am a PhD')
print(tokens)

with open('data/text/news/news-commentary-v15.en.lower', 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n')

print('tokenizer...')

batch = tokenizer.batch_encode_plus(lines[:10000],padding=True,truncation=True,max_length=512)
input_ids = batch['input_ids']
attention_mask = batch['attention_mask']

target_words = []
attribute_words = []
print('loading bias words...')
with open('data/female.txt', 'r') as f:
    contents = f.read()
    attribute_words += contents.replace('\n',' ').split()
with open('data/male.txt', 'r') as f:
    contents = f.read()
    attribute_words += contents.replace('\n',' ').split()
with open('data/stereotype.txt', 'r') as f:
    contents = f.read()
    target_words += contents.replace('\n',' ').split()
print('decoding bias words...')
# tokenizer.encode(['i','like','me'],add_special_tokens=False)
attribute_words_tokenizer_id = tokenizer.encode(attribute_words,add_special_tokens=False)
target_words_tokenizer_id = tokenizer.encode(target_words,add_special_tokens=False)
all_special_ids = torch.tensor(tokenizer.all_special_ids)

attribute_words_tokenizer_id = torch.tensor(attribute_words_tokenizer_id)
target_words_tokenizer_id = torch.tensor(target_words_tokenizer_id)


labels = torch.tensor(input_ids)
mask = torch.tensor(attention_mask)
input_ids = labels.detach().clone()
rand = torch.rand(input_ids.shape)
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
qt_mask = torch.zeros_like(input_ids)
print('masking words...')
for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    if i % 10000 == 0:
        print(f'{i}/{input_ids.shape[0]}')
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    do_qt = False
    for mask_id in input_ids[i, selection]:
        if mask_id in target_words_tokenizer_id and mask_id not in all_special_ids:
            do_qt = True
    if do_qt:
        for j in range(input_ids[i].shape[0]):
            if input_ids[i][j] in attribute_words_tokenizer_id and input_ids[i][j] not in input_ids[i][j]:
                qt_mask[i,j] = 1
    # mask input_ids
    input_ids[i, selection] = 3  # our custom [MASK] token == 3


torch.save(input_ids, 'data/tensor/input_ids.pt')
torch.save(mask, 'data/tensor/attention_mask.pt')
torch.save(labels, 'data/tensor/labels.pt')
torch.save(qt_mask, 'data/tensor/qt_mask.pt')