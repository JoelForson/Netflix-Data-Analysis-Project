#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 00:51:11 2025

@author: joelforson
"""

import pandas as pd

# Path to the dataset
file_path = '/Users/joelforson/Downloads/Final_Cleaned_Netflix_Dataset (1).csv'

# 1. Load the dataset
df = pd.read_csv(file_path)

# 2. For each column of string (object) type, remove brackets and quotes
def remove_brackets_and_quotes(value):
    if isinstance(value, str):
        # Remove [ and ], and "
        return value.replace('[','').replace(']','').replace("'",'')
    else:
        return value

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(remove_brackets_and_quotes)

# 3. Save back with the same name
df.to_csv(file_path, index=False)