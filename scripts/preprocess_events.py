import re

import numpy as np
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

categorical_events = {'vaccine', 'hormones', 'fertilized', 'well', 'footrim', 'stop_milking', 'fail_birth',
                      'after_birth_treatment', 'metr', 'not_fertilize', 'ketos', 'placenta', 'pares',
                      'defect'}
protocol_events = {'ill_other', 'mast', 'lame'}
mid_events = {'sold', 'death'}
simple_events = {'birth', 'stop_milking2', 'stop_hormones'}
number_events = {'weight', 'pregnant', 'long_pregnant'}
special_events = {'move', 'abort'}
all_events = categorical_events | protocol_events | mid_events | simple_events | special_events | number_events

kaggle = False
save_dataset = False

if kaggle:
    dataset_path = '/kaggle/input/cow-treatment/events.csv'
else:
    dataset_path = '../data/raw/raw.csv'

# read and drop extra columns
df = pd.read_csv(dataset_path, low_memory=False)
# df = df[df['Пол'] == 'F']
df.drop([col for col in df.columns if col not in col_mapper], axis=1, inplace=True)
df.columns = [col_mapper[col] for col in df.columns]

# process date columns
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y')

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

# explore events info
unique_events = df['event'].value_counts().index.tolist()
print(f"Events that we do not considered: ",
      set(unique_events).difference(all_events))


class RemarkEntry:

    def __init__(self, names, name):
        self.names = set(names)
        self.name = name

    def match(self, name):
        return name in self.names


class Parser:
    def __init__(self, event):
        self.event = event
        self.extra_symbols = re.compile('[^a-z0-9]')

    def parse(self, remark):
        return remark

    def modify_event(self, remark):
        return self.event


class CategoricalParser(Parser):

    def __init__(self, event, entries=()):
        super(CategoricalParser, self).__init__(event)
        self.entries = list(entries)

    def parse(self, remark):
        remark = self.extra_symbols.sub('', remark)
        for entry in self.entries:
            if entry.match(remark):
                return entry.name
        if remark == '' or remark == '-':
            remark = 'other'
        return remark

    def modify_event(self, remark):
        return '_'.join([self.event, remark])

    def add_entries(self, entries):
        self.entries += entries


class MidParser(CategoricalParser):

    def __init__(self, **kwargs):
        super(MidParser, self).__init__(**kwargs)

    def parse(self, remark):
        values = remark.split(';')
        if len(values) <= 1:
            return 'other'
        return super().parse(values[1])


class AbortParser(CategoricalParser):

    def __init__(self, **kwargs):
        super(AbortParser, self).__init__(**kwargs)
        self.num_extra_symbols = re.compile('[^0-9]')
        self.num_finder = re.compile('[0-9]')

    def parse(self, remark):
        m = self.num_finder.match(remark)
        if m:
            return int(self.num_extra_symbols.sub('', remark))
        else:
            return super().parse(remark)

    def modify_event(self, remark):
        if str(remark).isdigit():
            return self.event
        else:
            return '_'.join([self.event, remark])

class ProtocolParser(CategoricalParser):

    def __init__(self, unique_values, **kwargs):
        super(ProtocolParser, self).__init__(**kwargs)
        self.extra_symbols = re.compile('[^a-z0-9_]')
        self.value_mapper = {value: pvalue for value, pvalue in
                             zip(unique_values, self.preprocess_protocols(unique_values))}

    def preprocess_protocols(self, protocols):
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
            if value != '':
                result.append('_'.join([name, value]))
            else:
                result.append(name)
        return result

    def parse(self, remark):
        value = self.value_mapper[remark]
        value = self.extra_symbols.sub('', value)
        for entry in self.entries:
            if entry.match(value):
                return entry.name
        return value


class NumberParser(Parser):
    def __init__(self, **kwargs):
        super(NumberParser, self).__init__(**kwargs)
        self.extra_symbols = re.compile('[^0-9]')

    def parse(self, remark):
        value = self.extra_symbols.sub('', remark)
        return int(value)


class MoveParser(Parser):
    def __init__(self, **kwargs):
        super(MoveParser, self).__init__(**kwargs)

    def parse(self, remark):
        values = remark.split('t')
        fr = int(values[0][1:])
        to = int(values[1])
        return '_'.join([str(fr), str(to)])


print("Working with events remarks")
event_remarks = {event: df[df['event'] == event]['remark'].value_counts().index.tolist() for event in unique_events}
parsers = {
    **{event: Parser(event) for event in simple_events},
    **{event: MidParser(event=event) for event in mid_events},
    **{event: NumberParser(event=event) for event in number_events},
    **{event: ProtocolParser(event=event, unique_values=event_remarks[event]) for event in protocol_events},
    **{event: CategoricalParser(event=event) for event in categorical_events},
    'move': MoveParser(event='move'),
    'abort': AbortParser(event='abort')
}
parsers['well'].add_entries([
    RemarkEntry(['procie', 'procee', '611078', '812142', '910092', '102181', '22592', '912238'], 'other'),
    RemarkEntry(['mastit', 'matstit', 'mastit3'], 'mastit'),
    RemarkEntry(['hromota', 'hromata'], 'hromota')
])
parsers['ill_other'].add_entries([
    RemarkEntry(['tugodoj'], 'vyma')
])
parsers['death'].add_entries([
    RemarkEntry(['procee'], 'other')
])
parsers['sold'].add_entries([
    RemarkEntry(['other', 'procee'], 'other'),
    RemarkEntry(['agalaktia'], 'agalakt'),
    RemarkEntry(['kopyta'], 'foot')
])
parsers['mast'].add_entries([
    RemarkEntry(['mv', 'mb'], 'mv')
])
parsers['abort'].add_entries([
    RemarkEntry(['ds', 'dlsh', 'dlshema'], 'sh'),
    RemarkEntry(['bezjt', 'bjt'], 'bjt'),
    RemarkEntry(['brakgine', 'brak', 'brakjirn'], 'brak'),
])
parsers['defect'].add_entries([
    RemarkEntry(['vyma', 'tugodoj', '2soska', 'atr1', 'atr3', 'atr2', 'atr4', 'atp4'], 'vyma'),
    RemarkEntry(['braknogi', 'nogi'], 'foot'),
    RemarkEntry(['belmo', 'slepaa'], 'eye'),
    RemarkEntry(['abszess'], 'abs'),
])
parsers['fail_birth'].add_entries([
    RemarkEntry(['fk', 'fol', 'fkstel', 'fl'], 'fk'),
    RemarkEntry(['lk', 'lke'], 'lk'),
    RemarkEntry(['e', 'es', 'em'], 'es'),
    RemarkEntry(['jt', 'jte', 'jt3', 'jtm', 'ujt', 'jttosaa', 'jtpolik', 'jtdlsh', 'jzt'], 'jt'),
    RemarkEntry(['bezjt', 'bjt'], 'bjt'),
    RemarkEntry(['abs', 'brakabs', 'jtabs', 'brmatka', 'abmatki', 'abc'], 'abs'),
    RemarkEntry(['pk', 'polikist', 'polikis', 'jtpolik'], 'pk'),
    RemarkEntry(['gpf', 'gipofunk', 'gipof'], 'gpf'),
    RemarkEntry(['brak', 'bpak', 'hudaa', 'braktosa', 'braknogi', 'brakjirn', 'izbraka', 'jtbrak', 'brjirna', 'embr'],
                'brak'),
    RemarkEntry(['dlsh', 'dlshema', 'dlshoho', '11sh', 'jtdlsh', 'ds'], 'sh')
])
parsers['not_fertilize'].add_entries([
    RemarkEntry(['genekol', 'genikol', 'ginekol'], 'ginekol'),
    RemarkEntry(['atp', 'at', 'am', 'atb', 'amb'], 'at'),
    RemarkEntry(['brakmol', 'moloko'], 'moloko'),
    RemarkEntry(
        ['jirnaa', 'jirn', 'melkaa', 'razvitie', 'rost', 'nedorost', 'uzkijtaz', 'hudaa', 'tos', 'tossust', 'ves'],
        'weight'),
    RemarkEntry(
        ['abs', 'absmatki', 'abszes', 'brakabs', 'absmatk', 'absmatka', 'abc', 'abz', 'abzes', 'abzopuh', 'bezmatki',
         'bezmetki', 'abszess', 'avs', 'bmatki'], 'abs'),
    RemarkEntry(['agalakti', 'agalakt'], 'agalakt'),
    RemarkEntry(['mast', 'mastit'], 'mast'),
    RemarkEntry(['tugodoj', 'nadoj', 'tonksos', 'vyma', 'sustvym', 'vymatroz', 'nogvyma'], 'vyma'),
    RemarkEntry(['nogi', 'troznogi', 'braknogi', 'nogiputy', 'nogitos', 'nogi3', 'nogitroz'], 'foot'),
    RemarkEntry(['glaza', 'bezglaza', 'slepaa'], 'eye'),
    RemarkEntry(['zoobrak', 'brvypad', 'braktroz', ''], 'brak'),
    RemarkEntry(['vypac', 'vypaciv'], 'vypac')
])
parsers['stop_milking'].add_entries([
    RemarkEntry(['ceba', 'seva', 'azit', 'atr124', 'seba'], 'seva')
])
df_events = []
for event in unique_events:
    df_event = df[df['event'] == event]

    if event in parsers:
        parser = parsers[event]
        df_event['remark'] = df_event['remark'].apply(lambda x: parser.parse(x))
    df_events.append(df_event)
    unique_values = df_event['remark'].value_counts().index.tolist()
    print(f"Event - {event}")
    print("---------------- Values --------------------")
    print(unique_values)
    print("--------------------------------------------")
df = pd.concat(df_events)

# sort events by date to adequate grouping in future
df.sort_values(by=['date', 'id'], ascending=True, inplace=True)


def save_simple_events(df):
    events_df = df[['id', 'date', 'event', 'remark']]
    df_events = []
    for event in unique_events:
        df_event = events_df[df['event'] == event]
        if event in parsers:
            parser = parsers[event]
            df_event['event'] = df_event['remark'].apply(lambda x: parser.modify_event(x))
        df_events.append(df_event)
    events_df = pd.concat(df_events)
    events_df.drop(['remark'], axis=1, inplace=True)
    events_df.sort_values(by=['date', 'id'], ascending=True, inplace=True)
    events_df = events_df.reset_index().drop(['index'], axis=1)
    if kaggle:
        events_df.to_feather('simple_events.ftr')
    else:
        events_df.to_feather('../data/simple_events.ftr')


def save_cow_features(df):
    cows_df = df[['id', 'birthday', 'birth_result', 'birth_difficult']].groupby('id').agg('first').reset_index()
    if kaggle:
        cows_df.to_feather('cows.ftr')
    else:
        cows_df.to_feather('../data/cows.ftr')


def save_weighting_history(df):
    weights_df = df[df['event'] == 'weight'][['id', 'date', 'remark']].reset_index().drop(['index'], axis=1)
    weights_df['weight'] = weights_df['remark'].astype(np.uint16)
    weights_df.drop(['remark'], axis=1, inplace=True)
    if kaggle:
        weights_df.to_feather('weights.ftr')
    else:
        weights_df.to_feather('../data/weights.ftr')


if save_dataset:
    save_simple_events(df)
    save_cow_features(df)
    save_weighting_history(df)
