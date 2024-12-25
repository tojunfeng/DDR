from transformers import Trainer


class BiencoderTrainer(Trainer):
    '''Custom Trainer for Biencoder'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''Compute loss'''
        query = inputs['query']
        passage = inputs['passage']
        kl_label = inputs['kl_label']
        outputs = model(query=query, passage=passage, kl_label=kl_label)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
