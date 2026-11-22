# -*- coding: utf-8 -*-
"""Run the two torch-based views in their own process.

On Windows, importing scikit-learn before torch in the *same* process can make
``torch`` fail to load ``c10.dll`` (``OSError: [WinError 1114]``).  Rather than
police import order across the whole package, the two neural views are executed
as child processes and their probabilities are exchanged through an ``.npz``.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

MODULE_OF = {"tlora": "tlora_view", "cnn": "cnn_star"}


def score(view, seqs, device="cuda", quiet=True):
    """Probability vector for `view` ('tlora' or 'cnn') on a list of sequences."""
    module = MODULE_OF[view]
    tmp = tempfile.mkdtemp(prefix="prismaip_")
    csv = os.path.join(tmp, "seqs.csv")
    npz = os.path.join(tmp, view + ".npz")
    pd.DataFrame({"sequence": list(seqs)}).to_csv(csv, index=False)
    cmd = [sys.executable, os.path.join(HERE, module + ".py"),
           "--score", csv, "--out", npz, "--device", device]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if quiet else None)
    return np.load(npz)[view]


def train_cnn(device="cuda", seed=0, quiet=False):
    """Fit the five CNN fold models (writes weights/cnn/star_fold<k>.pt)."""
    cmd = [sys.executable, os.path.join(HERE, "cnn_star.py"),
           "--train", "--device", device, "--seed", str(seed)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL if quiet else None)
