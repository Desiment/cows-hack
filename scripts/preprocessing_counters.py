from copy import copy
import re

import numpy as np
import pandas as pd

df = pd.read_csv("../data/raw/raw.csv", low_memory=False)

col_mapper = {
    'Номер события': 'event_id',
    'Пол': 'sex',
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

df.drop([col for col in df.columns if col not in col_mapper], axis=1, inplace=True)
df.columns = [col_mapper[col] for col in df.columns]

EventSynonims = {
    "ABORT": ["ABORT", "АБОРТ"],
    "TOSCM": ["TOSCM", "НА_СХЕМУ"],
    "NULSCM": ["NULSCM", "СО_СХЕМЫ"],
    "DNB": ["DNB", "НЕОСЕМ"],
    "BRED": ["BRED", "ОСЕМЕН"],
    "FRESH": ["FRESH", "ОТЕЛ"],
    "PREG": ["PREG", "СТЕЛН"],
    "PREGBEF": ["PREGBEF", "СТЕЛНДО"],
    "DRY": ["DRY", "СУХОСТ"],
    "DRY2": ["DRY2", "СУХ2"],
    "OPEN": ["OPEN", "ЯЛОВАЯ"],
    # -------
    "WEIGHT": ["WEIGHT", "ВЕС"],
    "DEAD": ["DEAD", "ПАЛА"],
    "MOVE": ["MOVE", "ПЕРЕВОД"],
    "SOLD": ["SOLD", "ПРОДАНА"],
    # -------
    "VAC": ["VAC", "ВАКЦИН"],
    "VACVIR": ["VACVIR", "ВАКВИРУС"],
    "FOOTRIM": ["FOOTRIM", "РАСЧКОП"],
    "HEALTH": ["HEALTH", "WELL", "ЗДОРОВА"],
    "BROKE": ["BROKE", "ДЕФЕКТ"],
    "POT": ["POT", "ПРОФОТ"],
    "ILLMISC": ["ILLMISC", "БОЛЕЗНЬ"],
    "LAME": ["LAME", "ХРОМОТА"],
    "KETOS": ["KETOS", "КЕТОЗ"],
    "MAST": ["MAST", "МАСТИТ"],
    "METR": ["METR", "МЕТРИТ"],
    "PARES": ["PARES", "ПАРЕЗ"],
    "RP": ["RP", "ПОСЛЕД"],
}

EventName = dict()
for event in EventSynonims.keys():
    for label in EventSynonims[event]:
        EventName[label] = event
EventTypes = EventSynonims.keys()
df['event'] = df['event'].apply(lambda label: EventName[label])


def remarks(event):
    return df[df['event'] == event]['remark'].unique().tolist()


for event in EventTypes:
    print(event + ':\n', remarks(event))
    print('---------------------------------------------------------------------------')


def splitter(s):
    return s.split(';')[1].strip().rstrip() if len(s.split(';')) > 1 else 'OTHER'


df['remark'] = df.apply(
    lambda row: splitter(row.remark) if (row.event == 'SOLD' or row.event == 'DEAD') else row.remark, axis=1)


def rule_name(s):
    symbols = (u'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789_', u'ABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA0123456789_')
    trans = {ord(a): ord(b) for a, b in zip(*symbols)}
    return s.translate(trans)


class ParsingRule:
    def __init__(self, match, parser, name=''):
        self.name = rule_name(name)
        self.parse = parser
        self.match = match

    def __copy__(self):
        return ParsingRule(self.match, self.parse, copy(self.name))

    def copy(self):
        return ParsingRule(self.match, self.parse, copy(self.name))

    @staticmethod
    def match_in(match_list, parse, name=''):
        return ParsingRule(lambda x: x in match_list, parse, name)

    @staticmethod
    def match_str(match_str, parse, name=''):
        return ParsingRule(lambda x: x == match_str, parse, name)

    @staticmethod
    def match_reg(reg, parse, name=''):
        return ParsingRule(lambda x: reg.match(x), parse, name)

    @staticmethod
    def bin():
        return lambda x: 1


def satisfy(s, rules):
    return any([r.match(s) for r in rules])


def uniq(names):
    return [ParsingRule.match_str(s, ParsingRule.bin(), rule_name(s)) for s in names]


def combi(rules, names):
    return rules + [ParsingRule.match_str(s, ParsingRule.bin(), rule_name(s)) for s in names if not satisfy(s, rules)]


def comp(rules, names):
    return rules + [
        ParsingRule.match_in([s for s in names if not satisfy(s, rules)], ParsingRule.bin(), rule_name('OTHER'))]


# -------

def prot_id(s):
    a = re.search(r"\d", s)
    b = re.search(r"_", s)
    if not a:
        return ''
    if b:
        return int(s[a.start():b.start():])
    return s[a.start()::]


def prot_name(s):
    a = re.search(r"\d", s)
    m = 0 if not a else a.start()
    return s[0:m:] if m else s


def prot_nums(s):
    m = []
    if s.find('_') == -1:
        return set()

    t = s[s.find('_') + 1::]
    for i in range(len(t)):
        if t[i] == '_' or t[i] == '-':
            for j in range(int(t[i - 1]) + 1, int(t[i + 1])):
                m.append(j)
        elif t[i].isdigit():
            m.append(int(t[i]))
    return set(m)


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


def prot(protocols):
    protocols = preprocess_protocols(protocols)
    prot_splited = []
    for p in protocols:
        if len(prot_nums(p)) == 0:
            prot_splited.append([prot_name(p), prot_id(p), '0'])
        else:
            for i in prot_nums(p):
                prot_splited.append([prot_name(p), prot_id(p), str(i)])
    prot_splited = [list(x) for x in set(tuple(x) for x in prot_splited)]
    return [ParsingRule.match_reg(re.compile(f'{p[0]}{p[1]}_(([0-9]*{p[2]})|{p[2]})[0-9]*'), ParsingRule.bin(),
                                  f'{p[0]}{p[1]}_{p[2]}') if p[2] != '0' else ParsingRule.match_reg(
        re.compile(f'{p[0]}{p[1]}'), ParsingRule.bin(), f'{p[0]}{p[1]}') for p in prot_splited]


# 'ABORT':
RemarksABORT = [ParsingRule.match_in(['ЖТ', 'ЖТ,'], ParsingRule.bin(), 'ЖТ'),
                ParsingRule.match_in(['ДС', 'ДЛ СХЕМА', 'ДЛ СХ'], ParsingRule.bin(), 'СХ'),
                ParsingRule.match_in(['БЖТ', 'БЕЗ ЖТ'], ParsingRule.bin(), 'БЖТ'),
                ParsingRule.match_str('ФК', ParsingRule.bin(), 'ФК'),
                ParsingRule.match_str('ЛК', ParsingRule.bin(), 'ЛК')]

# 'HEALTH':
RemarksHEALTH = [ParsingRule.match_in(['МАТСТИТ', 'МАСТИТ,', 'МАСТИТ3', 'МАСТИТ'], ParsingRule.bin(), 'МАСТИТ'),
                 ParsingRule.match_in(['ХРОМОТА', 'ХРОМАТА'], ParsingRule.bin(), 'ХРОМОТА')]

# 'BROKE':
RemarksBROKE = [ParsingRule.match_in(['НОГИ', 'БРАКНОГИ'], ParsingRule.bin(), 'НОГИ'),
                ParsingRule.match_in(['2СОСКА', '2 СОСКА', 'ТУГОДОЙ', 'ATP4', 'АТР4', 'АТР1', 'АТР2', 'АТР3', 'ВЫМЯ'],
                                     ParsingRule.bin(), 'ВЫМЯ'),
                ParsingRule.match_in(['БЕЛЬМО', 'СЛЕПАЯ'], ParsingRule.bin(), 'СЛЕПАЯ')]

# 'OPEN':
RemarksOPEN = [ParsingRule.match_in(['ФК', 'ФК, СТЕЛ', 'ФОЛ', 'ФЛ'], ParsingRule.bin(), 'ФК'),
               ParsingRule.match_in(['ЛК', 'ЛКЭ'], ParsingRule.bin(), 'ЛК'),
               ParsingRule.match_in(['ЖТ', 'ЖТ Э', 'ЖТ,', 'ЖТ3', 'ЖТМ', 'ЮЖТ', 'ЖТ,ТОЩАЯ', 'ЖТ ПОЛИК', 'ЖТ ДЛ СХ'],
                                    ParsingRule.bin(), 'ЖТ'),
               ParsingRule.match_in(['БЕЗЖТ', 'БЕЗ ЖТ'], ParsingRule.bin(), 'БЖТ'),
               ParsingRule.match_in(['АБС', 'БРАК АБС', 'БР МАТКА', 'БРАКАБС', 'АБ МАТКИ', 'ЖТ/АБС'], ParsingRule.bin(),
                                    'АБЦ'),
               ParsingRule.match_in(['ПК', 'ПОЛИКИСТ', 'ПОЛИКИС', 'ЖТ ПОЛИК'], ParsingRule.bin(), 'ПК'),
               ParsingRule.match_in(['ГПФ', 'ГИПОФУНК', 'ГИПОФ'], ParsingRule.bin(), 'ГПФ'),
               ParsingRule.match_in(
                   ['БПАК', 'БРАК', 'ХУДАЯ', 'БРАКНОГИ', 'БРАКТОЩА', 'БРАКЖИРН', 'БР ЖИРНА', 'ИЗ БРАКА'],
                   ParsingRule.bin(),
                   'БРАК'),
               ParsingRule.match_in(['ДЛ СХ', 'ДЛСХ', 'ДЛ СХЕМА', 'ДЛСХ ОХО', '11 СХ', 'ЖТ ДЛ СХ'], ParsingRule.bin(),
                                    'CХ')]

# 'DNB':
RemarksDNB = [ParsingRule.match_in(['ГИНЕКОЛ', 'ГЕНИКОЛ', 'ГЕНЕКОЛ'], ParsingRule.bin(), 'ГИН'),
              ParsingRule.match_in(['АТП', 'АТ'], ParsingRule.bin(), 'АТ'),
              ParsingRule.match_in(['БРАК МОЛ', 'МОЛОКО'], ParsingRule.bin(), 'МОЛ'),
              ParsingRule.match_in(
                  ['ЖИРНАЯ', 'ЖИРН', 'ВЕС', 'ХУДАЯ', 'РОСТ', 'НЕДОРОСТ', 'РАЗВИТИЕ', 'УЗКИЙТАЗ', 'МЕЛКАЯ'],
                  ParsingRule.bin(),
                  'ВЕС'),
              ParsingRule.match_in(
                  ['АБС', 'АБСЦЕСС', 'БРАК АБС', 'ABC', 'АБЦ', 'АБЦЕС', 'АБЦ ОПУХ', 'АБСМАТКА', 'БЕЗМАТКИ', 'БЕЗМЕТКИ'],
                  ParsingRule.bin(), 'АБЦ'),
              ParsingRule.match_in(['АГАЛАКТ', 'АГАЛАКТИ'], ParsingRule.bin(), 'АГ'),
              ParsingRule.match_in(['МАСТИТ', 'МАСТ'], ParsingRule.bin(), 'МАСТ'),
              ParsingRule.match_in(['ТУГОДОЙ', 'НАДОЙ', 'ТОНК СОС', 'ВЫМЯТРОЦ'], ParsingRule.bin(), 'ВЫМЯ'),
              ParsingRule.match_in(['БРАКНОГИ', 'ТРОЦНОГИ', 'НОГИТРОЦ', 'НОГИ ТОЩ'], ParsingRule.bin(), 'НОГИ'),
              ParsingRule.match_in(['ТОЩ', 'ТОЩ СУСТ', 'ТОЩЬ', 'НОГИ ТОЩ'], ParsingRule.bin(), 'ТОЩ'),
              ParsingRule.match_in(['СЛЕПАЯ', 'ГЛАЗА', 'БЕЗГЛАЗА'], ParsingRule.bin(), 'CЛЕПАЯ')]

# 'DRY':
RemarksDRY = [ParsingRule.match_in(['CEBA', 'СЕВА', 'СEBA', 'АТР124', 'AZIT'], ParsingRule.bin(), 'СЕВА')]

RuleBin = ParsingRule(lambda x: 1, ParsingRule.bin(), 'BIN')
RuleNum = ParsingRule.match_reg(re.compile('[0-9]*'), lambda x: int(x) if x.isdigit() else 0, 'INT')
RuleDays = ParsingRule.match_reg(re.compile('[0-9]* ((ДНИ)|(DAYS))'),
                                 lambda x: int(x[0:-4]) if re.match('[0-9]* (ДНИ)', x) else int(x[0:-5]), 'DAYS')

EventRules = {
    'ABORT': comp([RuleDays.copy()] + RemarksABORT, remarks('ABORT')),
    'TOSCM': uniq(remarks('TOSCM')),
    'NULSCM': [RuleBin.copy()],
    'DNB': comp(RemarksDNB, remarks('DNB')),
    'BRED': uniq(remarks('BRED')),
    'FRESH': [RuleBin.copy()],
    'PREG': [RuleDays.copy()],
    'PREGBEF': [RuleDays.copy()],
    'DRY': comp(RemarksDRY, ('DRY')),
    'DRY2': [RuleBin.copy()],
    'OPEN': comp(RemarksOPEN, ('OPEN')),
    # -------
    'WEIGHT': [RuleNum.copy()],
    'DEAD': uniq(remarks('DEAD')),
    'MOVE': [RuleBin.copy()],
    'SOLD': uniq(remarks('SOLD')),
    # -------
    'VAC': uniq(remarks('VAC')),
    'VACVIR': uniq(remarks('VACVIR')),
    'FOOTRIM': uniq(remarks('FOOTRIM')),
    'HEALTH': comp(RemarksHEALTH, remarks('HEALTH')),
    'BROKE': comp(RemarksBROKE, remarks('BROKE')),
    'POT': uniq(remarks('POT')),
    'ILLMISC': prot(remarks('ILLMISC')),
    'LAME': prot(remarks('LAME')),
    'KETOS': uniq(remarks('KETOS')),
    'MAST': prot(remarks('MAST')),
    'METR': uniq(remarks('METR')),
    'PARES': uniq(remarks('PARES')),
    'RP': uniq(remarks('RP')),
}

# Get All Rules:
print("Токены парсинга:")
for event in EventRules.keys():
    print(event + ':')
    for rule in EventRules[event]:
        rule.name = event + '_' + rule_name(rule.name.upper())
        print(rule.name)
    print('--------------------')

counters = []
for event in EventRules.keys():
    for rule in EventRules[event]:
        counters.append(df['remark'].apply(lambda x: rule.parse(x) if rule.match(x) else 0) * (df['event'] == event))
        counters[-1] = counters[-1].rename(rule.name)
    print(f"Finished event {event}")
df = pd.concat([df] + counters, axis=1)

df_features = [
    df['sex'].apply(lambda x: 1 if x == 'F' else 0).rename('sex'),
    df['birth_result'].apply(lambda x: str(x).count('FA')).rename('childs_fa'),
    df['birth_result'].apply(lambda x: str(x).count('MA')).rename('childs_ma'),
    df['birth_result'].apply(lambda x: str(x).count('FD')).rename('childs_fd'),
    df['birth_result'].apply(lambda x: str(x).count('MD')).rename('childs_md'),
    df['birth_result'].apply(lambda x: str(x).count('M')).rename('childs_m'),
    df['birth_result'].apply(lambda x: str(x).count('F')).rename('childs_f'),
    df['birth_result'].apply(lambda x: str(x).count('D')).rename('childs_d'),
    df['birth_result'].apply(lambda x: str(x).count('A')).rename('childs_a')
]

df = pd.concat([df] + df_features, axis=1)
df = df.drop(['birth_result'], axis=1)

df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['birthday'] = pd.to_datetime(df['birthday'], infer_datetime_format=True)
df['age'] = ((df['date'] - df['birthday']) / np.timedelta64(1, 'D')).astype(int)
df = df.drop(columns=['birthday'])

month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', ' dec']
df_features = [df['date'].apply(lambda x: 1 if x.month == i + 1 else 0).rename(m) for i, m in enumerate(month)]
df = pd.concat([df] + df_features, axis=1)

df.to_csv('events_counters.csv', header=True)
