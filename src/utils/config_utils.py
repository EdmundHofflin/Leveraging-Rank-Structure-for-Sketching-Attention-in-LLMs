from typing import Union, Tuple, List, Optional

from ast import literal_eval
from argparse import Action, _copy_items, SUPPRESS

import numpy as np
import torch


# ===================
# InputParameterError
# ===================

def InputParameterError(Exception):
    pass



# =================
# Argparser Actions
# =================

class DictExtend(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Load existing items
        items = getattr(namespace, self.dest, None)
        # Copy items
        items = _copy_items(items)
        # Group values using { ... }
        start_idxs = []
        end_idxs = []
        for idx, value in enumerate(values):
            if value[0] == "{":
                start_idxs.append(idx)
            if value[-1] == "}":
                end_idxs.append(idx)
        assert len(start_idxs) == len(end_idxs)
        # Append dictionaries
        for i in range(len(start_idxs)):
            dict_as_str = ''.join(values[start_idxs[i]:end_idxs[i]+1])
            items.append(literal_eval(dict_as_str))
        # Parse each dict_str as dict
        setattr(namespace, self.dest, items)


class HelpPasser(Action):
    def __init__(self, option_strings, help_parser, help_intro, dest=SUPPRESS, nargs=0, default=SUPPRESS, required=False, type=None, metavar=None, help=None):
        self.help_parser = help_parser
        self.help_intro = help_intro
        if help is None:
            help = f"Help information for parameter {self.help_parser}:"
        super(HelpPasser, self).__init__(option_strings=option_strings, dest=dest, nargs=nargs, default=default, required=required, metavar=metavar, type=type, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(self.help_intro)
        out = self.help_parser.format_help()
        for i, line in enumerate(out.split('\n')):
            if i < 6:
                pass
            else:
                print(line)
        parser.exit()
