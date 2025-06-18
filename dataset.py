import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class WikiDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self): # called by dataloader to get length of dataset
        return len(self.chunks)
    
    def __getitem__(self, idx):
        input_ids = self.chunks[idx][:-1]
        labels = self.chunks[idx][1:] # output same as input for gpt training, but shifted right tp predic next token
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), 'labels': torch.tensor(labels, dtype=torch.long)}
    


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad with 0 (usually <pad>) for input_ids, and -100 for labels to ignore in loss
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {'input_ids': input_ids_padded, 'labels': labels_padded}
