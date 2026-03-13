"""
Metric Caller Module

Provides a unified interface for computing standard salient/camouflaged
object detection evaluation metrics: MAE, F-measure, S-measure,
E-measure, and weighted F-measure.

Original Author: Lart Pang (https://github.com/lartpang)
"""
# -*- coding: utf-8 -*-

import numpy as np
from py_sod_metrics.sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure


class CalTotalMetric(object):
    """Calculator for comprehensive segmentation evaluation metrics.

    Aggregates predictions and ground truths step by step, then computes
    MAE, F-measure, S-measure, E-measure, and weighted F-measure.
    """
    __slots__ = ["cal_mae", "cal_fm", "cal_sm", "cal_em", "cal_wfm"]

    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_path: str):
        """Process one prediction-ground truth pair.

        Args:
            pred: Prediction array (uint8).
            gt: Ground truth array (uint8).
            gt_path: Path to the ground truth file (for error reporting).
        """
        assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape, gt_path)
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        """Compute and return all metrics as a dictionary.

        Args:
            bit_width: Number of decimal places for rounding.

        Returns:
            Dictionary containing all metric results.
        """
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        def _round_w_zero_padding(_x):
            _x = str(_x.round(bit_width))
            _x += "0" * (bit_width - len(_x.split(".")[-1]))
            return _x

        results = {name: _round_w_zero_padding(metric) for name, metric in results.items()}

        return results