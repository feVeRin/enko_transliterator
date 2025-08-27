import torch


class TextDataset:
    '''
    Transliteration dataset class
    
    Args:
        data: dataframe of transliteration dataset
        tokenizer: Marian tokenizer 
        max_length: maximum sequence length
    '''
    
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])

        return batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row['input_text']
        target_text = row['target_text']

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = targets['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }
