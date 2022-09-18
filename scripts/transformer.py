from datetime import datetime

import pandas as pd
import numpy as np
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
        cows = cows[cows['event'].apply(lambda events: (len(events) >= n_items) and self.contains_mastit(events))]
        # normalize days for each cow
        cows['days'] = cows['days'].apply(lambda days: [(d - days[0]) / days[-1] for d in days])
        # preprocess events sequence
        cows['event'] = cows['event'].apply(self.process_events)

        self.unique_events = cows['event'].explode().value_counts().index.tolist()
        self.mapper = {event: event_id for event_id, event in enumerate(self.unique_events)}
        mast_indexes = cows['event'].apply(self.mastit_ids).tolist()
        self.mast_indexes = [(row, col) for row, cols in enumerate(mast_indexes) for col in cols if
                             col >= self.n_items - 1]
        cows = cows[['id', 'event', 'days']]
        self.items = list(cows.itertuples(index=False, name=None))

    def mastit_ids(self, events):
        return [event_id for event_id, event in enumerate(events) if event.split('_', 1)[0] == 'mast']

    def contains_mastit(self, events):
        return any([ev.split('_', 1)[0] == 'mast' for ev in events])

    def process_events(self, events):
        seq = []
        for event in events:
            if event.startswith(('mast', 'lame')):
                values = event.split('_')
                if len(values) == 3:
                    numbers = values[2]
                    for num in numbers:
                        seq.append('_'.join([values[0], values[1], num]))
                else:
                    seq.append(event)
            else:
                seq.append(event)
        return seq

    def decode_sequence(self, seq):
        return [self.unique_events[t] for t in seq]

    def get_dict_size(self):
        return len(self.unique_events)

    def __len__(self):
        return len(self.mast_indexes)

    def __getitem__(self, idx):
        row, col = self.mast_indexes[idx]
        item = self.items[row]
        lower_index = col - self.n_items + 1
        upper_index = col + 1
        tokens = [self.mapper[ev] for ev in item[1][lower_index:upper_index]][::-1]
        # days = item[1][lower_index:upper_index:-1]
        days = range(n_items)
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

    def __init__(self, dict_size, embed_dim=64, n_heads=8, n_layers=6, learning_rate=1e-3, label_smoothing=0.1,
                 ff_dim=128, embed_drop=0.1, attn_drop=0.1, resid_drop=0.1, pad_id=-1):
        super(Transformer, self).__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.save_hyperparameters()
        self.embedding = torch.nn.Embedding(dict_size, embed_dim)
        self.dropout = torch.nn.Dropout(embed_drop)
        self.layers = torch.nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, ff_dim, attn_drop, resid_drop) for _ in range(n_layers)])
        self.out = torch.nn.Linear(embed_dim, dict_size)

    def pos_embedding(self, x):
        powers = 10000 ** (2 / self.embed_dim * torch.arange(self.embed_dim // 2, device=x.device, dtype=torch.float32))
        x = torch.matmul(x.unsqueeze(-1), powers.unsqueeze(0))  # b n d/2
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x  # b n d

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        return attn_pad_mask

    def get_attention_subsequent_mask(self, q):
        bs, q_len = q.size()
        subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
        return subsequent_mask

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

    def training_step(self, batch, batch_id):
        events = batch['events']
        positions = batch['days'][..., :-1]

        x = events[..., :-1]
        label = events[..., 1:]

        logits, attn_weights = self.forward(x, positions)
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), label, label_smoothing=self.label_smoothing)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_id):
        events = batch['events']
        positions = batch['days'][..., :-1]

        x = events[..., :-1]
        label = events[..., 1:]

        logits, attn_weights = self.forward(x, positions)
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), label, label_smoothing=self.label_smoothing)
        self.log('val_loss', loss)


if __name__ == 'main':
    kaggle = False
    if kaggle:
        path = '/kaggle/input/cow-treatment-data/simple_events.ftr'
    else:
        path = '../data/simple_events.ftr'

    filter_events = {
        'mast',
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
        'move'
    }
    n_items = 7
    train_ratio = 0.8

    batch_size = 32

    # load data
    dataset = CowsEvents(path=path, filter_events=filter_events, n_items=n_items)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=2, drop_last=True, prefetch_factor=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              num_workers=2, drop_last=True, prefetch_factor=2)

    # create transformer
    model = Transformer(dict_size=dataset.get_dict_size(), learning_rate=1e-3,
                        embed_dim=256, ff_dim=512, attn_drop=0, embed_drop=0, label_smoothing=0, resid_drop=0,
                        n_layers=4)
    # tune model
    trainer = pl.Trainer(auto_select_gpus=True, auto_lr_find=True, max_epochs=-1)
    # fit model
    trainer.fit(model, train_loader, test_loader)
