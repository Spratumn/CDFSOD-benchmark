
SEEN_CLS_DICT = {
}

ALL_CLS_DICT = {
}

dataset = 'airship'
class_names = ['airship']

globals()[f'{dataset}_TRAIN'] = f'{dataset}_train'
globals()[f'{dataset}_TEST'] = f'{dataset}_test'
globals()[f'{dataset}_1SHOT'] = f'{dataset}_1shot'
globals()[f'{dataset}_5SHOT'] = f'{dataset}_5shot'
globals()[f'{dataset}_10SHOT'] = f'{dataset}_10shot'

SEEN_CLS_DICT[f'{dataset}_train'] = class_names
SEEN_CLS_DICT[f'{dataset}_test'] = class_names
SEEN_CLS_DICT[f'{dataset}_1shot'] = class_names
SEEN_CLS_DICT[f'{dataset}_5shot'] = class_names
SEEN_CLS_DICT[f'{dataset}_10shot'] = class_names

ALL_CLS_DICT[f'{dataset}_train'] = class_names
ALL_CLS_DICT[f'{dataset}_test'] = class_names
ALL_CLS_DICT[f'{dataset}_1shot'] = class_names
ALL_CLS_DICT[f'{dataset}_5shot'] = class_names
ALL_CLS_DICT[f'{dataset}_10shot'] = class_names

