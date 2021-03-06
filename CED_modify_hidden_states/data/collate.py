import torch



class PadCollate:
    """
    A collation function that pads all the sequences to the longest one
    """

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        output = self.tokenizer.batch_encode_plus(
            [(src, trg) for src, trg, _, _ in batch],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        output['ner'] = torch.tensor([ner for _, _, ner, _ in batch], dtype=torch.long)
        output['labels'] = torch.tensor([label for _, _, _, label in batch], dtype=torch.long).unsqueeze(1)

        for key in output.keys():
            output[key] = output[key].to(torch.device("cuda"))
        return output
