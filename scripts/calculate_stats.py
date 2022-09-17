import re

import pandas as pd

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

col_mapper = {
    'Номер животного': 'id',
    'Номер лактации': 'lactation_number',
    'Результат отела': 'birth_result',
    'Легкость отела': 'birth_difficult',
    'Дата рождения': 'birthday',
    'Дней в сухостое предыдущей лактации': 'days_no_milking',
    'Дней стельности при событии': 'days_pregnant',
    'Номер группы животного': 'group_id',
    'Предыдущий номер группы животного': 'prev_group_id',
    'Событие': 'event',
    'Дни доения при событии': 'days_milking',
    'Дата события': 'date',
    'Примечание события': 'remark'
}

dataset_path = '../data/raw/raw.csv'

# read and drop extra columns
df = pd.read_csv(dataset_path, low_memory=False)
df.drop([col for col in df.columns if col not in col_mapper], axis=1, inplace=True)
df.columns = [col_mapper[col] for col in df.columns]

# process date columns
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y')

# sort events by date to adequate grouping in future
df.sort_values(by=['date', 'id'], ascending=True, inplace=True)

# map all events to unified format
orig_unique_events = df['event'].str.lower().value_counts().index.tolist()
for event in orig_unique_events:
    if event not in event_type_mapper:
        print(f"Found unregistered event {event}")
df['event'] = df['event'].str.lower().apply(lambda val: event_type_mapper[val])

# transform symbols
symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюя", u"abvgdeejzijklmnoprstufhzcss_y_eua")
trans = {ord(a): ord(b) for a, b in zip(*symbols)}
df['remark'] = df['remark'].str.lower().apply(lambda val: val.translate(trans))


def preprocess_protocols(protocols):
    result = []
    for p in protocols:
        values = p.split('_', 1)
        name = values[0]
        if len(values) == 1:
            result.append(name)
            continue
        value = values[1]
        value = re.sub('[^\d_-]', '', value).replace('_', '-')
        chars = set([c for c in value])
        if value == '' or ('-' in chars and len(chars) == 1):
            value = '1234'
        else:
            values = value.split('-')
            digits = []
            for val in values:
                if val == '':
                    continue
                if len(digits) > 0:
                    d = digits[-1]
                    while d < int(val[0]):
                        d += 1
                        digits.append(d)
                for d in val:
                    digits.append(int(d))
            digits = [str(d) for d in set(digits) if d >= 1 and d <= 4]
            value = ''.join(digits)
        result.append('_'.join([name, value]))
    return result


# take only mastit rows
df = df[df['event'] == 'mast']

mast_remarks = df['remark'].value_counts().index.tolist()
mast_remark_mapper = dict(
    [(remark, processed_remark) for remark, processed_remark in zip(mast_remarks, preprocess_protocols(mast_remarks))])
df['remark'] = df['remark'].apply(lambda x: mast_remark_mapper[x])

df['nipple_1'] = df['remark'].apply(lambda x: int('1' in x.split('_', 1)[1]) if len(x.split('_', 1)) > 1 else 0)
df['nipple_2'] = df['remark'].apply(lambda x: int('2' in x.split('_', 1)[1]) if len(x.split('_', 1)) > 1 else 0)
df['nipple_3'] = df['remark'].apply(lambda x: int('3' in x.split('_', 1)[1]) if len(x.split('_', 1)) > 1 else 0)
df['nipple_4'] = df['remark'].apply(lambda x: int('4' in x.split('_', 1)[1]) if len(x.split('_', 1)) > 1 else 0)

stats = []
for year in (2020, 2021, 2022):
    for month in range(12):
        if year == 2022 and month > 8:
            break
        print(f"Year {year}, month {month + 1}")
        data = [
            ('all', df[(df['event'] == 'mast') & (df['date'].dt.year == year) & (df['date'].dt.month == month + 1)][
                'id'].count()),
            ('heads', df[(df['event'] == 'mast') & (df['date'].dt.year == year) & (df['date'].dt.month == month + 1)][
                'id'].drop_duplicates().count()),
            ('left_front', df[(df['event'] == 'mast') & (df['nipple_1'] > 0) & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)][
                'id'].count()),
            ('right_front',
             df[(df['event'] == 'mast') & (df['nipple_2'] > 0) & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)][
                 'id'].count()),
            ('left_back',
             df[(df['event'] == 'mast') & (df['nipple_3'] > 0) & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)][
                 'id'].count()),
            ('right_back',
             df[(df['event'] == 'mast') & (df['nipple_4'] > 0) & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)][
                 'id'].count()),
            ('counts_mv', df[(df['event'] == 'mast') & (df['remark'] == 'mv') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km5', df[(df['event'] == 'mast') & (df['remark'] == 'km5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km7', df[(df['event'] == 'mast') & (df['remark'] == 'km7') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km1', df[(df['event'] == 'mast') & (df['remark'] == 'km1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km6', df[(df['event'] == 'mast') & (df['remark'] == 'km6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km3', df[(df['event'] == 'mast') & (df['remark'] == 'km3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km8', df[(df['event'] == 'mast') & (df['remark'] == 'km8') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km4', df[(df['event'] == 'mast') & (df['remark'] == 'km4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm1', df[(df['event'] == 'mast') & (df['remark'] == 'tm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_km2', df[(df['event'] == 'mast') & (df['remark'] == 'km2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm5', df[(df['event'] == 'mast') & (df['remark'] == 'tm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_skm2', df[(df['event'] == 'mast') & (df['remark'] == 'skm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm2', df[(df['event'] == 'mast') & (df['remark'] == 'tm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm3', df[(df['event'] == 'mast') & (df['remark'] == 'tm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_skm5', df[(df['event'] == 'mast') & (df['remark'] == 'skm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm4', df[(df['event'] == 'mast') & (df['remark'] == 'tm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_skm3', df[(df['event'] == 'mast') & (df['remark'] == 'skm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_skm4', df[(df['event'] == 'mast') & (df['remark'] == 'skm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_skm6', df[(df['event'] == 'mast') & (df['remark'] == 'skm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_skm1', df[(df['event'] == 'mast') & (df['remark'] == 'skm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm6', df[(df['event'] == 'mast') & (df['remark'] == 'tm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),
            ('counts_tm9', df[(df['event'] == 'mast') & (df['remark'] == 'tm9') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].count()),

            ('ucounts_mv', df[(df['event'] == 'mast') & (df['remark'] == 'mv') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km5', df[(df['event'] == 'mast') & (df['remark'] == 'km5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km7', df[(df['event'] == 'mast') & (df['remark'] == 'km7') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km1', df[(df['event'] == 'mast') & (df['remark'] == 'km1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km6', df[(df['event'] == 'mast') & (df['remark'] == 'km6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km3', df[(df['event'] == 'mast') & (df['remark'] == 'km3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km8', df[(df['event'] == 'mast') & (df['remark'] == 'km8') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km4', df[(df['event'] == 'mast') & (df['remark'] == 'km4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm1', df[(df['event'] == 'mast') & (df['remark'] == 'tm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_km2', df[(df['event'] == 'mast') & (df['remark'] == 'km2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm5', df[(df['event'] == 'mast') & (df['remark'] == 'tm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_skm2', df[(df['event'] == 'mast') & (df['remark'] == 'skm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm2', df[(df['event'] == 'mast') & (df['remark'] == 'tm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm3', df[(df['event'] == 'mast') & (df['remark'] == 'tm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_skm5', df[(df['event'] == 'mast') & (df['remark'] == 'skm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm4', df[(df['event'] == 'mast') & (df['remark'] == 'tm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_skm3', df[(df['event'] == 'mast') & (df['remark'] == 'skm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_skm4', df[(df['event'] == 'mast') & (df['remark'] == 'skm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_skm6', df[(df['event'] == 'mast') & (df['remark'] == 'skm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_skm1', df[(df['event'] == 'mast') & (df['remark'] == 'skm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm6', df[(df['event'] == 'mast') & (df['remark'] == 'tm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),
            ('ucounts_tm9', df[(df['event'] == 'mast') & (df['remark'] == 'tm9') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].drop_duplicates().count()),

            ('1counts_mv', (df[(df['event'] == 'mast') & (df['remark'] == 'mv') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km5', (df[(df['event'] == 'mast') & (df['remark'] == 'km5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km7', (df[(df['event'] == 'mast') & (df['remark'] == 'km7') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km1', (df[(df['event'] == 'mast') & (df['remark'] == 'km1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km6', (df[(df['event'] == 'mast') & (df['remark'] == 'km6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km3', (df[(df['event'] == 'mast') & (df['remark'] == 'km3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km8', (df[(df['event'] == 'mast') & (df['remark'] == 'km8') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km4', (df[(df['event'] == 'mast') & (df['remark'] == 'km4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm1', (df[(df['event'] == 'mast') & (df['remark'] == 'tm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_km2', (df[(df['event'] == 'mast') & (df['remark'] == 'km2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm5', (df[(df['event'] == 'mast') & (df['remark'] == 'tm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_skm2', (df[(df['event'] == 'mast') & (df['remark'] == 'skm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm2', (df[(df['event'] == 'mast') & (df['remark'] == 'tm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm3', (df[(df['event'] == 'mast') & (df['remark'] == 'tm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_skm5', (df[(df['event'] == 'mast') & (df['remark'] == 'skm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm4', (df[(df['event'] == 'mast') & (df['remark'] == 'tm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_skm3', (df[(df['event'] == 'mast') & (df['remark'] == 'skm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_skm4', (df[(df['event'] == 'mast') & (df['remark'] == 'skm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_skm6', (df[(df['event'] == 'mast') & (df['remark'] == 'skm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_skm1', (df[(df['event'] == 'mast') & (df['remark'] == 'skm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm6', (df[(df['event'] == 'mast') & (df['remark'] == 'tm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),
            ('1counts_tm9', (df[(df['event'] == 'mast') & (df['remark'] == 'tm9') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 1).sum()),

            ('2counts_mv', (df[(df['event'] == 'mast') & (df['remark'] == 'mv') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km5', (df[(df['event'] == 'mast') & (df['remark'] == 'km5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km7', (df[(df['event'] == 'mast') & (df['remark'] == 'km7') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km1', (df[(df['event'] == 'mast') & (df['remark'] == 'km1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km6', (df[(df['event'] == 'mast') & (df['remark'] == 'km6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km3', (df[(df['event'] == 'mast') & (df['remark'] == 'km3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km8', (df[(df['event'] == 'mast') & (df['remark'] == 'km8') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km4', (df[(df['event'] == 'mast') & (df['remark'] == 'km4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm1', (df[(df['event'] == 'mast') & (df['remark'] == 'tm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_km2', (df[(df['event'] == 'mast') & (df['remark'] == 'km2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm5', (df[(df['event'] == 'mast') & (df['remark'] == 'tm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_skm2', (df[(df['event'] == 'mast') & (df['remark'] == 'skm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm2', (df[(df['event'] == 'mast') & (df['remark'] == 'tm2') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm3', (df[(df['event'] == 'mast') & (df['remark'] == 'tm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_skm5', (df[(df['event'] == 'mast') & (df['remark'] == 'skm5') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm4', (df[(df['event'] == 'mast') & (df['remark'] == 'tm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_skm3', (df[(df['event'] == 'mast') & (df['remark'] == 'skm3') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_skm4', (df[(df['event'] == 'mast') & (df['remark'] == 'skm4') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_skm6', (df[(df['event'] == 'mast') & (df['remark'] == 'skm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_skm1', (df[(df['event'] == 'mast') & (df['remark'] == 'skm1') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm6', (df[(df['event'] == 'mast') & (df['remark'] == 'tm6') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),
            ('2counts_tm9', (df[(df['event'] == 'mast') & (df['remark'] == 'tm9') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() == 2).sum()),

            (
            'greater2counts_mv', (df[(df['event'] == 'mast') & (df['remark'] == 'mv') & (df['date'].dt.year == year) & (
                    df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km5',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km5') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km7',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km7') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km1',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km1') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km6',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km6') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km3',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km3') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km8',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km8') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km4',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km4') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm1',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm1') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_km2',
                (df[(df['event'] == 'mast') & (df['remark'] == 'km2') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm5',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm5') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            ('greater2counts_skm2',
             (df[(df['event'] == 'mast') & (df['remark'] == 'skm2') & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm2',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm2') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm3',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm3') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            ('greater2counts_skm5',
             (df[(df['event'] == 'mast') & (df['remark'] == 'skm5') & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm4',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm4') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            ('greater2counts_skm3',
             (df[(df['event'] == 'mast') & (df['remark'] == 'skm3') & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            ('greater2counts_skm4',
             (df[(df['event'] == 'mast') & (df['remark'] == 'skm4') & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            ('greater2counts_skm6',
             (df[(df['event'] == 'mast') & (df['remark'] == 'skm6') & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            ('greater2counts_skm1',
             (df[(df['event'] == 'mast') & (df['remark'] == 'skm1') & (df['date'].dt.year == year) & (
                     df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm6',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm6') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum()),
            (
                'greater2counts_tm9',
                (df[(df['event'] == 'mast') & (df['remark'] == 'tm9') & (df['date'].dt.year == year) & (
                        df['date'].dt.month == month + 1)]['id'].value_counts() > 2).sum())

        ]

        stats.append((year, month + 1, data))

columns = ['names'] + [f'{t[0]}_{t[1]}' for t in stats]
data = []
for feature in range(len(stats[0][2])):
    row = [stats[0][2][feature][0]]
    for ym in range(len(stats)):
        row.append(stats[ym][2][feature][1])
    data.append(row)
stats_df = pd.DataFrame(data, columns=columns)
stats_df.to_csv('stats.csv')
