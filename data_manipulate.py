import pandas as pd
import numpy as np
import re

raw_meta = pd.read_excel('./raw_data/metadata.xls')
raw_micro = pd.read_csv('./raw_data/microbiome/microbial-540-1994.csv', index_col = 0)
raw_trans = pd.read_csv('./raw_data/sequence_data/mRNA-fpkm-540-56603.csv', index_col = 0)

trans_meta = raw_meta.dropna(subset = ['transcriptom_id'])

trans_meta.to_csv('./COAD/metadata.csv')
raw_micro.to_csv('./COAD/microbiomes/microbial.csv')
raw_trans.to_csv('./COAD/transcriptome/trancriptome.csv')