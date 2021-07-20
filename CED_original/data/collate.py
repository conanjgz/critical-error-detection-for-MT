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
            [(src, trg) for src, trg, _ in batch],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        output['labels'] = torch.tensor([label for _, _, label in batch], dtype=torch.long).unsqueeze(1)
        
        # for i in batch[0].keys():
        #     # assume each item in the batch has the same keys in their dicts
        #     for j in batch:
        #         print(j[i].shape)
        #     output[i] = torch.nn.utils.rnn.pad_sequence(
        #         [j[i] for j in batch], batch_first=True, padding_value=self.pad
        #     )
        #     print("*****************")
        #     print(output[i].shape)
        #     print("*****************")

        for key in output.keys():
            output[key] = output[key].to(torch.device("cuda"))
        return output
