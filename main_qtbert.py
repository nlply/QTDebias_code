from datasets import load_dataset
from tqdm.auto import tqdm
import torch

input_ids = torch.load('data/tensor/input_ids.pt')
attention_mask = torch.load('data/tensor/attention_mask.pt')
labels = torch.load('data/tensor/labels.pt')
qt_mask = torch.load('data/tensor/qt_mask.pt')

encodings = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels , 'qt_mask': qt_mask}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


dataset = Dataset(encodings)

loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)


# 初始化模型
from transformers_local.src.transformers import BertConfig

config = BertConfig(
    vocab_size=28996,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

from transformers_local.src.transformers import BertForMaskedLM

model = BertForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

# from transformers import AdamW

import torch

# activate training mode
model.train()
# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 5
print('begin training...')

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        qt_mask = batch['qt_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,qt_mask=qt_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model.save_pretrained('qtbert')  # and don't forget to save filiBERTo!


