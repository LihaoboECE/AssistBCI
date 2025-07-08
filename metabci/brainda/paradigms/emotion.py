# -*- coding: utf-8 -*-
"""
Emotion Paradigm.
"""
from .base import BaseParadigm


class Emotion(BaseParadigm):
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != "emotion":
            ret = False
        return ret
