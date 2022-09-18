from copy import copy
from datetime import datetime
from random import choices

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch


class CowsEvents(torch.utils.data.Dataset):

    def __init__(self, path, filter_events, n_items=5):
        self.n_items = n_items
        # reading cows events and filter them
        df = pd.read_feather(path)
        df = df[df['event'].apply(lambda ev: ev.split('_', 1)[0] in filter_events)]

        # sort by date to ensure that sequences in right order
        df.sort_values(['id', 'date'], ascending=True, inplace=True)

        # generate days feature
        df['days'] = (df['date'] - datetime(2019, 1, 1)).dt.days
        df.drop(['date'], axis=1, inplace=True)

        # we are only interested in cows with mastit and sequence of enough length
        cows = df.groupby('id').agg(list).reset_index()
        cows = cows[cows['event'].apply(lambda events: self.contains_mastit(events))]
        # normalize days for each cow
        cows['days'] = cows['days'].apply(lambda days: [(d - days[0]) / days[-1] for d in days])
        # preprocess events sequence
        cows['event'] = cows['event'].apply(self.process_events)
        cows = cows[cows['event'].apply(lambda events: (len(events) >= n_items))]

        self.unique_events = cows['event'].explode().value_counts().index.tolist()
        self.mapper = {event: event_id for event_id, event in enumerate(self.unique_events)}
        mast_indexes = cows['event'].apply(self.mastit_ids).tolist()
        self.mast_indexes = [(row, col) for row, cols in enumerate(mast_indexes) for col in cols if
                             col >= self.n_items - 1]
        self.mastit_events = [event for event in self.unique_events if event.split('_', 1)[0] == 'mast']

        cows = cows[['id', 'event', 'days']]
        self.items = list(cows.itertuples(index=False, name=None))

    def mastit_ids(self, events):
        return [event_id for event_id, event in enumerate(events) if event.split('_', 1)[0] == 'mast']

    def contains_mastit(self, events):
        return any([ev.split('_', 1)[0] == 'mast' for ev in events])

    def process_events(self, events):
        seq = []
        for event in events:
            if event.startswith('mast'):
                values = event.split('_')
                if len(values) == 3:
                    # numbers = values[2]
                    # for num in numbers:
                    #     seq.append('_'.join([values[0], num]))
                    seq.append(values[0])
                else:
                    seq.append(event)
            elif event.startswith('lame'):
                values = event.split('_')
                if len(values) == 3:
                    numbers = values[2]
                    for num in numbers:
                        seq.append('_'.join([values[0], num]))
                else:
                    seq.append(event)
            else:
                seq.append(event)

        _seq = []
        for item in seq:
            if (len(_seq) == 0) or seq[-1] != item:
                _seq.append(item)
        return _seq

    def decode_sequence(self, seq):
        return [self.unique_events[t] for t in seq]

    def encode_sequence(self, seq):
        return [self.mapper[item] for item in seq]

    def get_dict_size(self):
        return len(self.unique_events)

    def get_mastit_events(self):
        return self.mastit_events

    def __len__(self):
        return len(self.mast_indexes)

    def __getitem__(self, idx):
        row, col = self.mast_indexes[idx]
        item = self.items[row]
        lower_index = col - self.n_items + 1
        upper_index = col + 1
        tokens = self.encode_sequence(item[1][lower_index:upper_index][::-1])
        # days = item[1][lower_index:upper_index:-1]
        days = range(self.n_items)
        return {
            'id': item[0],
            'events': np.array(tokens, dtype=np.int64),
            'days': np.array(days, dtype=np.float32)
        }


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k, attn_pdrop):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

        self.dropout = torch.nn.Dropout(attn_pdrop)

    def forward(self, q, k, v, attn_mask):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn_score.masked_fill_(attn_mask, -1e9)
        attn_weights = torch.nn.Softmax(dim=-1)(attn_score)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, attn_pdrop):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = torch.nn.Linear(d_model, d_model)
        self.WK = torch.nn.Linear(d_model, d_model)
        self.WV = torch.nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k, attn_pdrop)
        self.linear = torch.nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        outputs = self.linear(attn)

        return outputs, attn_weights


class PositionWiseFeedForwardNetwork(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.gelu = torch.nn.GELU()

        # torch.nn.init.normal_(self.linear1.weight, std=0.02)
        # torch.nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        outputs = self.gelu(self.linear1(inputs))
        outputs = self.linear2(outputs)
        return outputs


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_pdrop, resid_pdrop):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads, attn_pdrop)
        self.dropout1 = torch.nn.Dropout(resid_pdrop)
        self.layernorm1 = torch.nn.LayerNorm(d_model, eps=1e-5)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = torch.nn.Dropout(resid_pdrop)
        self.layernorm2 = torch.nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, inputs, attn_mask):
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)

        return ffn_outputs, attn_weights


class Transformer(pl.LightningModule):

    def __init__(self, path, train_ratio, filter_events, n_items, batch_size,
                 embed_dim=64, n_heads=8, n_layers=6, learning_rate=1e-3, label_smoothing=0.1,
                 ff_dim=128, embed_drop=0.1, attn_drop=0.1, resid_drop=0.1, pad_id=-1):
        super(Transformer, self).__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.n_items = n_items
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size

        # load data
        self.dataset = CowsEvents(path=path, filter_events=filter_events, n_items=n_items)
        self.train_size = int(len(self.dataset) * train_ratio)
        self.test_size = len(self.dataset) - self.train_size
        self.resample_dataset()
        self.dict_size = self.dataset.get_dict_size()

        self.embedding = torch.nn.Embedding(self.dict_size, embed_dim, padding_idx=self.pad_id)
        # torch.nn.init.(self.embedding.weight, std=0.02)

        self.dropout = torch.nn.Dropout(embed_drop)
        self.layers = torch.nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, ff_dim, attn_drop, resid_drop) for _ in range(n_layers)])
        self.out = torch.nn.Linear(embed_dim, self.dict_size)

        self.save_hyperparameters()

    def pos_embedding(self, x):
        powers = 10000 ** (2 / self.embed_dim * torch.arange(self.embed_dim // 2, device=x.device, dtype=torch.float32))
        invert_powers = 1 / powers
        x = torch.matmul(x.unsqueeze(-1), invert_powers.unsqueeze(0))  # b n d/2
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x  # b n d

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        return attn_pad_mask

    def get_attention_subsequent_mask(self, q):
        bs, q_len = q.size()
        subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
        return subsequent_mask

    def cross_entropy(self, logits, targets, smoothing=0.0):
        # logits (b n c)
        # targets (b n)
        b, n, c = logits.shape
        small_smoothing = smoothing / (c - 1)
        one_hot = torch.ones_like(logits) * small_smoothing
        one_hot = torch.nn.functional.one_hot(targets, num_classes=self.dict_size) \
                  * (1 - smoothing - small_smoothing) + one_hot
        loss = torch.mean(torch.sum(torch.sum(- torch.log(logits + 1e-6) * one_hot, dim=-1), dim=-1))
        return loss

    def forward(self, x, positions):
        outputs = self.dropout(self.embedding(x)) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(x, x, self.pad_id)
        subsequent_mask = self.get_attention_subsequent_mask(x).to(device=attn_pad_mask.device)
        attn_mask = torch.gt((attn_pad_mask.to(dtype=subsequent_mask.dtype) + subsequent_mask), 0)

        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_mask)
            attention_weights.append(attn_weights)

        logits = torch.nn.functional.softmax(self.out(outputs), dim=-1)
        return logits, attention_weights

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer

    def sample(self, n_items, start=(), device=torch.device('cpu')):
        start_items = self.dataset.get_mastit_events()
        if len(start) == 0:
            starts = [[ch] for ch in choices(start_items, k=n_items)]
        else:
            starts = [copy(start) for _ in range(n_items)]

        sequences = []
        for sample_id, start in enumerate(starts):
            tokens = self.dataset.encode_sequence(start)

            for n_tokens in range(len(tokens), self.n_items + 1):
                x = torch.tensor([tokens], device=device, dtype=torch.int64)
                positions = torch.arange(n_tokens, dtype=torch.float32, device=device).view(1, n_tokens)
                with torch.no_grad():
                    logits, att_weights = self.forward(x, positions)
                logits = logits.view(n_tokens, self.dict_size)[-1].cpu().detach().numpy()  # (dict size)

                # inverse transform sampling
                value = np.random.uniform(size=1)
                cdf = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(logits)])
                msk = cdf <= value
                token_id = int(np.argmax(msk * np.arange(self.dict_size + 1)))
                tokens.append(token_id)

            sequences.append(self.dataset.decode_sequence(tokens)[::-1])

        return sequences

    def training_step(self, batch, batch_id):
        events = batch['events']
        positions = batch['days'][..., :-1]

        x = events[..., :-1]
        label = events[..., 1:]

        logits, attn_weights = self.forward(x, positions)
        loss = self.cross_entropy(logits, label)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_id):
        events = batch['events']
        positions = batch['days'][..., :-1]

        x = events[..., :-1]
        label = events[..., 1:]

        logits, attn_weights = self.forward(x, positions)
        loss = self.cross_entropy(logits, label)
        self.log('val_loss', loss)

    def resample_dataset(self):
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.test_size])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                           num_workers=8, drop_last=True, prefetch_factor=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                           num_workers=8, drop_last=True, prefetch_factor=4)


class SamplerCallback(pl.Callback):

    def __init__(self, n_samples=10, device=torch.device('cpu')):
        super(SamplerCallback, self).__init__()
        self.n_samples = n_samples
        self.device = device

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.resample_dataset()
        print("----------------------------------------------------")
        print(f"Sampling {self.n_samples} items from distribution")
        samples = pl_module.sample(self.n_samples, device=self.device)
        for sample_id, sample in enumerate(samples):
            print(f'--{sample_id}-- {" ".join(sample)}')
        print("----------------------------------------------------")


if __name__ == '__main__':
    kaggle = False
    if kaggle:
        path = '/kaggle/input/cow-treatment-data/simple_events.ftr'
    else:
        path = '../data/simple_events.ftr'

    filter_events = {
        'mast',
        'ill_other',
        'vaccine',
        'footrim',
        'lame',
        'ketos',
        'metr',
        'pares',
        'placenta',
        'defect',
        'hormones',
        'stop_hormones',
        'birth',
        'stop_milking',
        'stop_milking2',
        'weight',
        'fail_birth',
        'abort',
        'after_birth_treatment',
        # 'move'
    }
    n_items = 7
    train_ratio = 0.8
    n_samples = 10
    batch_size = 128

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # create transformer
    model = Transformer(path=path, train_ratio=train_ratio, filter_events=filter_events, n_items=n_items,
                        batch_size=batch_size, learning_rate=1e-3,
                        embed_dim=64, ff_dim=128, attn_drop=0.0, embed_drop=0.0, label_smoothing=0, resid_drop=0.0,
                        n_layers=2)
    # tune model
    trainer = pl.Trainer(auto_select_gpus=True, auto_lr_find=True, max_epochs=-1,  enable_model_summary=True,
                         weights_save_path='models', detect_anomaly=True,
                         # accelerator='gpu',gpus=1, devices=1, sa
                         reload_dataloaders_every_n_epochs=1, callbacks=[SamplerCallback(n_samples, device=device)])
    # trainer.tune(model)
    # fit model
    trainer.fit(model)
