import argparse

import torch

from transformer import Transformer

event_type_mapper = {
    'аборт': 'abort',
    'abort': 'abort',
    'на_схему': 'hormones',
    'toscm': 'hormones',
    'неосем': 'not_fertilize',
    'dnb': 'not_fertilize',
    'осемен': 'fertilized',
    'bred': 'fertilized',
    'отел': 'birth',
    'fresh': 'birth',
    'со_схемы': 'stop_hormones',
    'nulscm': 'stop_hormones',
    'стелн': 'pregnant',
    'стелндо': 'long_pregnant',
    'сухост': 'stop_milking',
    'dry': 'stop_milking',
    'сух2': 'stop_milking2',
    'яловая': 'fail_birth',
    'open': 'fail_birth',
    'вес': 'weight',
    'weight': 'weight',
    'пала': 'death',
    'перевод': 'move',
    'move': 'move',
    'продана': 'sold',
    'sold': 'sold',
    'вакцин': 'vaccine',
    'ваквирус': 'vaccine',
    'расчкоп': 'footrim',
    'footrim': 'footrim',
    'дефект': 'defect',
    'профот': 'after_birth_treatment',
    'pot': 'after_birth_treatment',
    'болезнь': 'ill_other',
    'illmisc': 'ill_other',
    'хромота': 'lame',
    'lame': 'lame',
    'кетоз': 'ketos',
    'ketos': 'ketos',
    'мастит': 'mast',
    'mast': 'mast',
    'метрит': 'metr',
    'metr': 'metr',
    'парез': 'pares',
    'послед': 'placenta',
    'rp': 'placenta',
    'здорова': 'well',
    'well': 'well'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample sequences")

    # Input data settings
    parser.add_argument("--n_samples", default=5, type=int, help="Number of samples")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = Transformer.load_from_checkpoint('models/checkpoints/epoch=389-step=15600.ckpt')

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

    model = Transformer(path='data/simple_events.ftr', train_ratio=train_ratio, filter_events=filter_events,
                        n_items=n_items, batch_size=batch_size, learning_rate=1e-3,
                        embed_dim=64, ff_dim=128, attn_drop=0.1, embed_drop=0.1, label_smoothing=0, resid_drop=0.0,
                        n_layers=6)
    state_dict = torch.load('models/checkpoints/epoch=182-step=7137.ckpt', map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model = model.to(device)

    samples = model.sample(n_items=args.n_samples, device=device)

    print(f"Sampled {len(samples)} samples")
    for sample_id, sample in enumerate(samples):
        seq = []
        for item in sample:
            values = item.split('_', 1)
            name = values[0]
            for k, v in event_type_mapper.items():
                if name.startswith(v):
                    name = k
                    break
            item = '_'.join([name] + values[1:])
            seq.append(item)
        print(f"--{sample_id + 1}-- {' '.join(seq)}")
