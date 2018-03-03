import copy
from typing import List

import numpy as np

ONE_HOT = True
SCALAR = False




def all_encoding_masks(n_features: int) -> List :
    results = []
    mask = [ONE_HOT] * n_features
    helper(0, mask, results)
    return results


def helper(start: int, mask: list, mask_list: list):
    mask_list.append(copy.deepcopy(mask))
    i = start
    while i < len(mask):
        mask[i] = SCALAR
        helper(i + 1, mask, mask_list)
        mask[i] = ONE_HOT
        i = i + 1


def weekday_to_num(weekday: str) -> int:
    if weekday == 'Monday':
        return 1
    elif weekday == 'Tuesday':
        return 2
    elif weekday == 'Wednesday':
        return 3
    elif weekday == 'Thursday':
        return 4
    elif weekday == 'Friday':
        return 5
    elif weekday == 'Saturday':
        return 6
    elif weekday == 'Sunday':
        return 7
    else:
        return None    


def work_flow_id_to_num(work_flow_id: str) -> int:
    words = work_flow_id.split('_')
    for word in words:
        if word.isdigit():
            return int(word)
        
def filename_to_num(filename: str) -> int:
    words = filename.split('_')
    for word in words:
        if word.isdigit():
            return int(word)


