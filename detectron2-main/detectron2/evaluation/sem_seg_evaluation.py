# # Copyright (c) Facebook, Inc. and its affiliates.
# import itertools
# import json
# import logging
# import numpy as np
# import os
# from collections import OrderedDict
# from typing import Optional, Union
# import pycocotools.mask as mask_util
# import torch
# from PIL import Image

# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.utils.comm import all_gather, is_main_process, synchronize
# from detectron2.utils.file_io import PathManager

# from .evaluator import DatasetEvaluator

# _CV2_IMPORTED = True
# try:
#     import cv2  # noqa
# except ImportError:
#     # OpenCV is an optional dependency at the moment
#     _CV2_IMPORTED = False


# def load_image_into_numpy_array(
#     filename: str,
#     copy: bool = False,
#     dtype: Optional[Union[np.dtype, str]] = None,
# ) -> np.ndarray:
#     with PathManager.open(filename, "rb") as f:
#         array = np.array(Image.open(f), copy=copy, dtype=dtype)
# #         array = np.where(array == 255, 0, 1).astype(dtype)
        
#     return array


# class SemSegEvaluator(DatasetEvaluator):
#     """
#     Evaluate semantic segmentation metrics.
#     """

#     def __init__(
#         self,
#         dataset_name,
#         distributed=True,
#         output_dir=None,
#         *,
#         sem_seg_loading_fn=load_image_into_numpy_array,
#         num_classes=None,
#         ignore_label=None,
#     ):
#         """
#         Args:
#             dataset_name (str): name of the dataset to be evaluated.
#             distributed (bool): if True, will collect results from all ranks for evaluation.
#                 Otherwise, will evaluate the results in the current process.
#             output_dir (str): an output directory to dump results.
#             sem_seg_loading_fn: function to read sem seg file and load into numpy array.
#                 Default provided, but projects can customize.
#             num_classes, ignore_label: deprecated argument
#         """
#         self._logger = logging.getLogger(__name__)
#         if num_classes is not None:
#             self._logger.warn(
#                 "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
#             )
#         if ignore_label is not None:
#             self._logger.warn(
#                 "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
#             )
#         self._dataset_name = dataset_name
#         self._distributed = distributed
#         self._output_dir = output_dir

#         self._cpu_device = torch.device("cpu")

#         self.input_file_to_gt_file = {
#             dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
#             for dataset_record in DatasetCatalog.get(dataset_name)
#         }

#         meta = MetadataCatalog.get(dataset_name)
#         # Dict that maps contiguous training ids to COCO category ids
#         try:
#             c2d = meta.stuff_dataset_id_to_contiguous_id
#             self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
#         except AttributeError:
#             self._contiguous_id_to_dataset_id = None
#         self._class_names = meta.stuff_classes
#         self.sem_seg_loading_fn = sem_seg_loading_fn
#         self._num_classes = len(meta.stuff_classes)
# #         import pdb; pdb.set_trace()
#         if num_classes is not None:
#             assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
#         self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

#         # This is because cv2.erode did not work for int datatype. Only works for uint8.
#         self._compute_boundary_iou = True
#         if not _CV2_IMPORTED:
#             self._compute_boundary_iou = False
#             self._logger.warn(
#                 """Boundary IoU calculation requires OpenCV. B-IoU metrics are
#                 not going to be computed because OpenCV is not available to import."""
#             )
#         if self._num_classes >= np.iinfo(np.uint8).max:
#             self._compute_boundary_iou = False
#             self._logger.warn(
#                 f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
#                 B-IoU metrics are not going to be computed. Max allowed value (exclusive)
#                 for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
#                 The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
#             )

#     def reset(self):
#         self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
#         self._b_conf_matrix = np.zeros(
#             (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
#         )
#         self._predictions = []

#     def process(self, inputs, outputs):
#         """
#         Args:
#             inputs: the inputs to a model.
#                 It is a list of dicts. Each dict corresponds to an image and
#                 contains keys like "height", "width", "file_name".
#             outputs: the outputs of a model. It is either list of semantic segmentation predictions
#                 (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
#                 segmentation prediction in the same format.
#         """
#         for input, output in zip(inputs, outputs):
#             output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
#             pred = np.array(output, dtype=np.int)
#             gt_filename = self.input_file_to_gt_file[input["file_name"]]
            
#             gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int)
            
#             gt[gt == self._ignore_label] = self._num_classes

#             self._conf_matrix += np.bincount(
#                 (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
#                 minlength=self._conf_matrix.size,
#             ).reshape(self._conf_matrix.shape)
# #             import pdb; pdb.set_trace()

#             if self._compute_boundary_iou:
#                 b_gt = self._mask_to_boundary(gt.astype(np.uint8))
#                 b_pred = self._mask_to_boundary(pred.astype(np.uint8))

#                 self._b_conf_matrix += np.bincount(
#                     (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
#                     minlength=self._conf_matrix.size,
#                 ).reshape(self._conf_matrix.shape)

#             self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

#     def evaluate(self):
#         """
#         Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

#         * Mean intersection-over-union averaged across classes (mIoU)
#         * Frequency Weighted IoU (fwIoU)
#         * Mean pixel accuracy averaged across classes (mACC)
#         * Pixel Accuracy (pACC)
#         """
#         if self._distributed:
#             synchronize()
#             conf_matrix_list = all_gather(self._conf_matrix)
#             b_conf_matrix_list = all_gather(self._b_conf_matrix)
#             self._predictions = all_gather(self._predictions)
#             self._predictions = list(itertools.chain(*self._predictions))
#             if not is_main_process():
#                 return

#             self._conf_matrix = np.zeros_like(self._conf_matrix)
#             for conf_matrix in conf_matrix_list:
#                 self._conf_matrix += conf_matrix

#             self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
#             for b_conf_matrix in b_conf_matrix_list:
#                 self._b_conf_matrix += b_conf_matrix

#         if self._output_dir:
#             PathManager.mkdirs(self._output_dir)
#             file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
#             with PathManager.open(file_path, "w") as f:
#                 f.write(json.dumps(self._predictions))
# #         import pdb; pdb.set_trace()
#         acc = np.full(self._num_classes, np.nan, dtype=np.float)
#         iou = np.full(self._num_classes, np.nan, dtype=np.float)
#         tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
#         pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
#         class_weights = pos_gt / np.sum(pos_gt)
#         pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
#         acc_valid = pos_gt > 0
#         acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
#         union = pos_gt + pos_pred - tp
#         iou_valid = np.logical_and(acc_valid, union > 0)
#         iou[iou_valid] = tp[iou_valid] / union[iou_valid]
#         macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
#         miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
#         fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
#         pacc = np.sum(tp) / np.sum(pos_gt)

#         if self._compute_boundary_iou:
#             b_iou = np.full(self._num_classes, np.nan, dtype=np.float)
#             b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float)
#             b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float)
#             b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float)
#             b_union = b_pos_gt + b_pos_pred - b_tp
#             b_iou_valid = b_union > 0
#             b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

#         res = {}
#         res["mIoU"] = 100 * miou
#         res["fwIoU"] = 100 * fiou
#         for i, name in enumerate(self._class_names):
#             res[f"IoU-{name}"] = 100 * iou[i]
#             if self._compute_boundary_iou:
#                 res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
#                 res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
#         res["mACC"] = 100 * macc
#         res["pACC"] = 100 * pacc
#         for i, name in enumerate(self._class_names):
#             res[f"ACC-{name}"] = 100 * acc[i]

#         if self._output_dir:
#             file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
#             with PathManager.open(file_path, "wb") as f:
#                 torch.save(res, f)
#         results = OrderedDict({"sem_seg": res})
#         self._logger.info(results)
#         return results

#     def encode_json_sem_seg(self, sem_seg, input_file_name):
#         """
#         Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
#         See http://cocodataset.org/#format-results
#         """
#         json_list = []
#         for label in np.unique(sem_seg):
#             if self._contiguous_id_to_dataset_id is not None:
#                 assert (
#                     label in self._contiguous_id_to_dataset_id
#                 ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
#                 dataset_id = self._contiguous_id_to_dataset_id[label]
#             else:
#                 dataset_id = int(label)
#             mask = (sem_seg == label).astype(np.uint8)
#             mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
#             mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
#             json_list.append(
#                 {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
#             )
#         return json_list

#     def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
#         assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
#         h, w = mask.shape
#         diag_len = np.sqrt(h**2 + w**2)
#         dilation = max(1, int(round(dilation_ratio * diag_len)))
#         kernel = np.ones((3, 3), dtype=np.uint8)

#         padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
#         eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
#         eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
#         boundary = mask - eroded_mask
#         return boundary


# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
from numpy import *
# import pydensecrf.densecrf as dcrf

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # {img, annos}: {np.array(uint8)}

    EPSILON = 1e-8

    M = 2
    tau = 1.05
    # CRF model setup

#    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')

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


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output_tmp = output
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)

            gt[gt == self._ignore_label] = self._num_classes

            ####################
            met = Metrics()
            
            image_np = self.sem_seg_loading_fn(input["file_name"], dtype=np.uint8)
            # image_np = cv2.resize(image_np, (384, 384))

            gt_tmp = gt
            pred_tmp = torch.tensor(pred)

            res = pred_tmp.unsqueeze(0)
            res = res.sigmoid()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            predict_np = (res.squeeze().cpu().data.numpy() * 255).astype(np.uint8)
            predict_np = predict_np.copy(order='C')

            crf_input = image_np.astype(np.uint8)
            crf_input = crf_input.copy(order='C')
#            predict_np = crf_refine(crf_input, predict_np)
            predict_np = np.where(predict_np<127.5, 0, 1).astype(np.uint8)

            predict_np = np.array(output, dtype=np.uint8)
            # im = Image.fromarray(predict_np)
            # print(gt.shape)
            # predict_np = im.resize(gt.shape)
            # predict_np = cv2.resize(predict_np, gt.shape)
            # print(predict_np)

            met.update(pred=predict_np, target=gt_tmp)
            ####################

            ####################
            # import torch.nn.functional as F
            # result = output_tmp["sem_seg"]
            # result = result[:, : 384, : 384].expand(1, -1, -1, -1)
            # result = F.interpolate(
            #     result, size=gt.shape, mode="bilinear", align_corners=False
            # )[0]
            # pred = np.array(result.argmax(dim=0).to(self._cpu_device), dtype=int)
            ####################

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

            return met

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
