# -*- coding: utf-8 -*-

import os

from xmnlp.config import path
from xmnlp.utils import load_stopword


# 模型地址配置
MODEL_DIR = os.getenv('XMNLP_MODEL', None)
ALLOW_POS = ['an', 'i', 'j', 'l', 'n', 'nr', 'ns', 'nt', 'nz',
             't', 'v', 'vd', 'vn', 'x', 'nn', 'g']
# store stopwords as a set for efficient membership checks and unions
SYS_STOPWORDS = set(load_stopword(path.stopword['corpus']['stopword']))
# allow sentence vector genres
ALLOW_SV_GENRES = ['通用', '金融', '国际']
