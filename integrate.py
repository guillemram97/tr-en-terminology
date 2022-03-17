import sys
import os
import time
from pathlib import Path

from translate_interactive import TranslationInter

def t_in_milliseconds(t):
    return int(round(t * 1000))


def init(bin_dir, model_dir, spm_dir, src, tgt):
    engine = TranslationInter(bin_dir, model_dir, spm_dir, src, tgt)
    return engine

def translate(engine: TranslationInter, textToTranslate: str, logger, terminologyList={}):
    t0 = time.time()
    report = engine.translate(textToTranslate, terminologyList)
    dt = time.time() - t0
    result = {
        "result": report['translation'],
        "time_taken": t_in_milliseconds(dt),
        "time_pre": t_in_milliseconds(report['preprocessing']),
        "time_translating": t_in_milliseconds(report['translating']),
        "time_post": t_in_milliseconds(report['postprocessing']),
        "error": None
    }
    return result

