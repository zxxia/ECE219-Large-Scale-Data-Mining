#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:35:26 2018

@author: zxxia
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

# Load dataset
categories = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories,
                                 shuffle=True, random_state=42)

# Convert text to matrix
vectorizer = text.CountVectorizer(min_df=3, max_df=0.3, stop_words='english')

# Filter unecessary words
X_train_counts = vectorizer.fit_transform(raw_documents=twenty_train['data'])
