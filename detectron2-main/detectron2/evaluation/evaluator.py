# # Copyright (c) Facebook, Inc. and its affiliates.
# import datetime
# import logging
# import time
# from collections import OrderedDict, abc
# from contextlib import ExitStack, contextmanager
# from typing import List, Union
# import torch
# from torch import nn

# from detectron2.utils.comm import get_world_size, is_main_process
# from detectron2.utils.logger import log_every_n_seconds


# class DatasetEvaluator:
#     """
#     Base class for a dataset evaluator.

#     The function :func:`inference_on_dataset` runs the model over
#     all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

#     This class will accumulate information of the inputs/outputs (by :meth:`process`),
#     and produce evaluation results in the end (by :meth:`evaluate`).
#     """

#     def reset(self):
#         """
#         Preparation for a new round of evaluation.
#         Should be called before starting a round of evaluation.
#         """
#         pass

#     def process(self, inputs, outputs):
#         """
#         Process the pair of inputs and outputs.
#         If they contain batches, the pairs can be consumed one-by-one using `zip`:

#         .. code-block:: python

#             for input_, output in zip(inputs, outputs):
#                 # do evaluation on single input/output pair
#                 ...

#         Args:
#             inputs (list): the inputs that's used to call the model.
#             outputs (list): the return value of `model(inputs)`
#         """
#         pass

#     def evaluate(self):
#         """
#         Evaluate/summarize the performance, after processing all input/output pairs.

#         Returns:
#             dict:
#                 A new evaluator class can return a dict of arbitrary format
#                 as long as the user can process the results.
#                 In our train_net.py, we expect the following format:

#                 * key: the name of the task (e.g., bbox)
#                 * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
#         """
#         pass


# class DatasetEvaluators(DatasetEvaluator):
#     """
#     Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

#     This class dispatches every evaluation call to
#     all of its :class:`DatasetEvaluator`.
#     """

#     def __init__(self, evaluators):
#         """
#         Args:
#             evaluators (list): the evaluators to combine.
#         """
#         super().__init__()
#         self._evaluators = evaluators

#     def reset(self):
#         for evaluator in self._evaluators:
#             evaluator.reset()

#     def process(self, inputs, outputs):
#         for evaluator in self._evaluators:
#             evaluator.process(inputs, outputs)

#     def evaluate(self):
#         results = OrderedDict()
#         for evaluator in self._evaluators:
#             result = evaluator.evaluate()
#             if is_main_process() and result is not None:
#                 for k, v in result.items():
#                     assert (
#                         k not in results
#                     ), "Different evaluators produce results with the same key {}".format(k)
#                     results[k] = v
#         return results


# def inference_on_dataset(
#     model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
# ):
#     """
#     Run model on the data_loader and evaluate the metrics with evaluator.
#     Also benchmark the inference speed of `model.__call__` accurately.
#     The model will be used in eval mode.

#     Args:
#         model (callable): a callable which takes an object from
#             `data_loader` and returns some outputs.

#             If it's an nn.Module, it will be temporarily set to `eval` mode.
#             If you wish to evaluate a model in `training` mode instead, you can
#             wrap the given model and override its behavior of `.eval()` and `.train()`.
#         data_loader: an iterable object with a length.
#             The elements it generates will be the inputs to the model.
#         evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
#             but don't want to do any evaluation.

#     Returns:
#         The return value of `evaluator.evaluate()`
#     """
#     num_devices = get_world_size()
#     logger = logging.getLogger(__name__)
#     logger.info("Start inference on {} batches".format(len(data_loader)))

#     total = len(data_loader)  # inference data loader must have a fixed length
#     if evaluator is None:
#         # create a no-op evaluator
#         evaluator = DatasetEvaluators([])
#     if isinstance(evaluator, abc.MutableSequence):
#         evaluator = DatasetEvaluators(evaluator)
#     evaluator.reset()

#     num_warmup = min(5, total - 1)
#     start_time = time.perf_counter()
#     total_data_time = 0
#     total_compute_time = 0
#     total_eval_time = 0
#     with ExitStack() as stack:
#         if isinstance(model, nn.Module):
#             stack.enter_context(inference_context(model))
#         stack.enter_context(torch.no_grad())

#         start_data_time = time.perf_counter()
#         for idx, inputs in enumerate(data_loader):
#             total_data_time += time.perf_counter() - start_data_time
#             if idx == num_warmup:
#                 start_time = time.perf_counter()
#                 total_data_time = 0
#                 total_compute_time = 0
#                 total_eval_time = 0

#             start_compute_time = time.perf_counter()
#             outputs = model(inputs)
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             total_compute_time += time.perf_counter() - start_compute_time

#             start_eval_time = time.perf_counter()
#             evaluator.process(inputs, outputs)
#             total_eval_time += time.perf_counter() - start_eval_time

#             iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
#             data_seconds_per_iter = total_data_time / iters_after_start
#             compute_seconds_per_iter = total_compute_time / iters_after_start
#             eval_seconds_per_iter = total_eval_time / iters_after_start
#             total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
#             if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
#                 eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
#                 log_every_n_seconds(
#                     logging.INFO,
#                     (
#                         f"Inference done {idx + 1}/{total}. "
#                         f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
#                         f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
#                         f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
#                         f"Total: {total_seconds_per_iter:.4f} s/iter. "
#                         f"ETA={eta}"
#                     ),
#                     n=5,
#                 )
#             start_data_time = time.perf_counter()

#     # Measure the time only for this worker (before the synchronization barrier)
#     total_time = time.perf_counter() - start_time
#     total_time_str = str(datetime.timedelta(seconds=total_time))
#     # NOTE this format is parsed by grep
#     logger.info(
#         "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
#             total_time_str, total_time / (total - num_warmup), num_devices
#         )
#     )
#     total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
#     logger.info(
#         "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
#             total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
#         )
#     )

#     results = evaluator.evaluate()
#     # An evaluator may return None when not in main process.
#     # Replace it by an empty dict instead to make it easier for downstream code to handle
#     if results is None:
#         results = {}
#     return results


# @contextmanager
# def inference_context(model):
#     """
#     A context where the model is temporarily changed to eval mode,
#     and restored to previous mode afterwards.

#     Args:
#         model: a torch Module
#     """
#     training_mode = model.training
#     model.eval()
#     yield
#     model.train(training_mode)


# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from tqdm import tqdm
from numpy import *

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import numpy as np

class Metrics:
    def __init__(self):
        self.initial()

    def initial(self):
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []
        self.precision = []
        self.recall = []
        self.cnt = 0
        self.mae = []
        self.tot = []

    def update(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        assert pred.all() >= 0.0 and pred.all() <= 1.0
        assert target.all() >= 0.0 and target.all() <= 1.0
        assert pred.shape == target.shape

        ## threshold = 0.5
        def TP(prediction, true): return sum(logical_and(prediction, true))

        def TN(prediction, true): return sum(logical_and(
            logical_not(prediction), logical_not(true)))

        def FP(prediction, true): return sum(
            logical_and(logical_not(true), prediction))
        def FN(prediction, true): return sum(
            logical_and(logical_not(prediction), true))

        trueThres = 0.5
        predThres = 0.5
        self.tp.append(TP(pred >= predThres, target > trueThres))
        self.tn.append(TN(pred >= predThres, target > trueThres))
        self.fp.append(FP(pred >= predThres, target > trueThres))
        self.fn.append(FN(pred >= predThres, target > trueThres))
        self.tot.append(target.shape[0])
        assert self.tot[-1] == (self.tp[-1]+self.tn[-1] +
                                self.fn[-1]+self.fp[-1])

        # 256 precision and recall
        tmp_prec = []
        tmp_recall = []
        eps = 1e-4
        trueHard = target > 0.5
        for threshold in range(256):
            threshold = threshold / 255.
            tp = TP(pred >= threshold, trueHard)+eps
            ppositive = sum(pred >= threshold)+eps
            tpositive = sum(trueHard)+eps
            tmp_prec.append(tp/ppositive)
            tmp_recall.append(tp/tpositive)
        self.precision.append(tmp_prec)
        self.recall.append(tmp_recall)

        # mae
        self.mae.append(mean(abs(pred-target)))

        self.cnt += 1

    def compute_iou(self):
        iou = []
        n = len(self.tp)
        for i in range(n):
            iou.append(self.tp[i]/(self.tp[i]+self.fp[i]+self.fn[i]))
        return mean(iou)

    def compute_fbeta(self, beta_square=0.3):
        precision = array(self.precision).mean(axis=0)
        recall = array(self.recall).mean(axis=0)
        max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r)
                           for p, r in zip(precision, recall)])
        return max_fmeasure

    def compute_mae(self):
        return mean(self.mae)

    def compute_ber(self):
        return array([100*(1.0-0.5*(self.tp[i]/(self.tp[i]+self.fn[i]) + self.tn[i]/(self.tn[i]+self.fp[i]))) for i in range(len(self.tot))]).mean()

    def report(self):
        report = "Count:"+str(self.cnt)+"\n"
        report += f"IOU: {self.compute_iou()} FB: {self.compute_fbeta()} MAE: {self.compute_mae()} BER: {self.compute_ber()}"

        return report

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        ####################
        mets=[]
        ####################

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()

            #####################
            met = evaluator.process(inputs, outputs)
            mets.append(met)
            #####################

            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        
    #####################
    merge_metrics = Metrics()
    
    for x in tqdm(mets):
        merge_metrics.tp += x.tp
        merge_metrics.tn += x.tn
        merge_metrics.fp += x.fp
        merge_metrics.fn += x.fn
        merge_metrics.precision += x.precision
        merge_metrics.recall += x.recall
        merge_metrics.cnt += x.cnt
        merge_metrics.mae += x.mae
        merge_metrics.tot += x.tot

    synchronize()
    merge_metrics_tp_list = all_gather(merge_metrics.tp)
    merge_metrics_tn_list = all_gather(merge_metrics.tn)
    merge_metrics_fp_list = all_gather(merge_metrics.fp)
    merge_metrics_fn_list = all_gather(merge_metrics.fn)
    merge_metrics_precision_list = all_gather(merge_metrics.precision)
    merge_metrics_recall_list = all_gather(merge_metrics.recall)
    merge_metrics_cnt_list = all_gather(merge_metrics.cnt)
    merge_metrics_mae_list = all_gather(merge_metrics.mae)
    merge_metrics_tot_list = all_gather(merge_metrics.tot)

    merge_metrics.tp = []
    for merge_metrics_tp in merge_metrics_tp_list:
            merge_metrics.tp += merge_metrics_tp
    
    merge_metrics.tn = []
    for merge_metrics_tn in merge_metrics_tn_list:
            merge_metrics.tn += merge_metrics_tn

    merge_metrics.fp = []
    for merge_metrics_fp in merge_metrics_fp_list:
            merge_metrics.fp += merge_metrics_fp

    merge_metrics.fn = []
    for merge_metrics_fn in merge_metrics_fn_list:
            merge_metrics.fn += merge_metrics_fn

    merge_metrics.precision = []
    for merge_metrics_precision in merge_metrics_precision_list:
            merge_metrics.precision += merge_metrics_precision

    merge_metrics.recall = []
    for merge_metrics_recall in merge_metrics_recall_list:
            merge_metrics.recall += merge_metrics_recall

    merge_metrics.cnt = 0
    for merge_metrics_cnt in merge_metrics_cnt_list:
            merge_metrics.cnt += merge_metrics_cnt

    merge_metrics.tot = []
    for merge_metrics_tot in merge_metrics_tot_list:
            merge_metrics.tot += merge_metrics_tot

    merge_metrics.mae = []
    for merge_metrics_mae in merge_metrics_mae_list:
            merge_metrics.mae += merge_metrics_mae

    print(merge_metrics.report())
    logger.info(merge_metrics.report())
    #####################


    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
