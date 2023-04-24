import argparse
from tqdm.auto import tqdm
import torch
from transformers_local.src.transformers import BertConfig,BertForMaskedLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='bert, qtbert', default='qtbert')
    parser.add_argument('--input', type=str, help='file path of data', default='data/tensor/')
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--epochs', type=int, help='epochs', default=5)
    parser.add_argument('--vocab_size', type=int, help='vocab size', default=30522)
    args = parser.parse_args()

    return args


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


def main(args):
    input_ids = torch.load(f'{args.input}input_ids.pt')
    attention_mask = torch.load(f'{args.input}attention_mask.pt')
    labels = torch.load(f'{args.input}labels.pt')
    if args.model == 'qtbert':
        qt_mask = torch.load(f'{args.input}qt_mask.pt')
    else:
        qt_mask = torch.zeros_like(attention_mask)
    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'qt_mask': qt_mask}

    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.model == 'qtbert' or args.model == 'bert':
        config = BertConfig(
            vocab_size=args.vocab_size,  # we align this to the tokenizer vocab_size
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )
        model = BertForMaskedLM(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
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
            outputs = model(input_ids, attention_mask=attention_mask, qt_mask=qt_mask,
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

    model.save_pretrained(args.model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
