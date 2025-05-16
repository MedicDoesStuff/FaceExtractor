import sys
import os
import time
import cv2
import numpy as np
import yaml
import pickle
from scipy.spatial import ConvexHull
from numba import jit
import onnxruntime as ort
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import torch

try:
    from DFLJPG import DFLJPG
except ImportError:
    print("DFLJPG module not found. Ensure it's in your environment.")
    sys.exit(1)

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from core.lib import facedesc as fd
from core.lib import math as lib_math
from core.lib import onnxruntime as lib_ort
from core.lib.functools import cached_method
from core.lib.image import FImage
from core.lib.math import FRectf

class YoloV7Face:
    @dataclass
    class DetectJob:
        aug_imgs : Sequence['YoloV7Face._AugmentedImage']
        minimum_confidence : float
        nms_threshold : float
        min_face_size : float

    @dataclass
    class _AugmentedImage:
        img : np.ndarray
        scale : float

    @staticmethod
    def get_available_devices() -> Sequence[lib_ort.DeviceRef]:
        return [lib_ort.get_cpu_device()] + lib_ort.get_avail_gpu_devices()

    def __init__(self, device : lib_ort.DeviceRef ):
        if device not in YoloV7Face.get_available_devices():
            raise Exception(f'device {device} is not in available devices for YoloV7Face')
        model_path = Path('/content/FaceExtractor/dist/modelhub/onnx/YoloV7Face/YoloV7Face.onnx')
        if not model_path.exists():
            raise FileNotFoundError(f"YoloV7Face.onnx not found at {model_path}")
        self._sess = lib_ort.InferenceSession_with_device(model_path, device)
        self._input_name = self._sess.get_inputs()[0].name

    def detect(self, img : FImage, resolution : int = None, pad_to_resolution=False, augment_pyramid=False,
               minimum_confidence : float = 0.3, nms_threshold=0.3, min_face_size=8) -> Sequence[fd.FFace]:
        return self.detect_job(self.prepare_job(img, resolution, pad_to_resolution, augment_pyramid,
                                               minimum_confidence, nms_threshold, min_face_size))

    def prepare_job(self, img : FImage, resolution : int = None, pad_to_resolution=False, augment_pyramid=False,
                   minimum_confidence : float = 0.3, nms_threshold=0.3, min_face_size=8) -> DetectJob:
        return self.DetectJob(
            aug_imgs=self._augment(img.bgr().u8(), grid_size=[32,32], resolution=resolution,
                                  pad_to_resolution=pad_to_resolution, augment_pyramid=augment_pyramid),
            minimum_confidence=minimum_confidence, nms_threshold=nms_threshold, min_face_size=min_face_size
        )

    def detect_job(self, job : DetectJob) -> Sequence[fd.FFace]:
        preds = []
        for aug_img in job.aug_imgs:
            pred = self._get_preds(aug_img.img.CHW()[None,...])[0]
            pred[:,0:4] *= 1/aug_img.scale
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        preds = preds[ preds[...,4] >= job.minimum_confidence ]
        preds = preds[ preds[...,2:4].min(-1) >= job.min_face_size ]
        x,y,w,h,confidence = preds.T
        w_half = w/2
        h_half = h/2
        l, t, r, b = x-w_half, y-h_half, x+w_half, y+h_half
        keep = lib_math.nms(l,t,r,b, confidence, job.nms_threshold)
        l, t, r, b, confidence = l[keep], t[keep], r[keep], b[keep], confidence[keep]
        return [ fd.FFace(FRectf(l,t,r,b)).set_confidence(confidence).set_detector_id('YoloV7Face')
                 for l,t,r,b,confidence in np.stack([l, t, r, b, confidence], -1) ]

    def _get_preds(self, img : np.ndarray):
        N,C,H,W = img.shape
        preds = self._sess.run(None, {self._input_name: img})
        pred0, pred1, pred2 = [pred.reshape((N,C,21,pred.shape[-2], pred.shape[-1])).transpose(0,1,3,4,2)[...,0:5] for pred in preds]
        pred0 = self._process_pred(pred0, W, H, anchor=[[4,5],[6,8],[10,12]]).reshape((N, -1, 5))
        pred1 = self._process_pred(pred1, W, H, anchor=[[15,19],[23,30],[39,52]]).reshape((N, -1, 5))
        pred2 = self._process_pred(pred2, W, H, anchor=[[72,97],[123,164],[209,297]]).reshape((N, -1, 5))
        return np.concatenate([pred0, pred1, pred2], 1)[...,:5]

    def _process_pred(self, pred, img_w, img_h, anchor):
        pred_h = pred.shape[-3]
        pred_w = pred.shape[-2]
        anchor = np.float32(anchor)[None,:,None,None,:]
        grid = self._get_grid(pred_w, pred_h)
        stride = (img_w // pred_w, img_h // pred_h)
        pred = YoloV7Face._sigmoid(pred)
        pred[..., 0:2] = (pred[..., 0:2]*2 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4]*2)**2 * anchor
        return pred

    @cached_method
    def _get_grid(self, W, H) -> np.ndarray:
        _xv, _yv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        return np.stack((_xv, _yv), 2).reshape((1, 1, H, W, 2))

    @staticmethod
    def _sigmoid(x : np.ndarray) -> np.ndarray:
        positives = x >= 0
        negatives = ~positives
        exp_x_neg = np.exp(x[negatives])
        y = x.copy()
        y[positives] = 1 / (1 + np.exp(-x[positives]))
        y[negatives] = exp_x_neg / (1 + exp_x_neg)
        return y

    def _augment(self, img: np.ndarray, grid_size: List[int], resolution: int = None,
                 pad_to_resolution: bool = False, augment_pyramid: bool = False) -> Sequence[_AugmentedImage]:
        aug_imgs = []
        h, w, _ = img.shape
        scale = 1.0
        if resolution:
            scale = resolution / max(h, w)
            if pad_to_resolution:
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))
                if new_h < resolution or new_w < resolution:
                    pad_h = (resolution - new_h) // 2
                    pad_w = (resolution - new_w) // 2
                    img = cv2.copyMakeBorder(img, pad_h, resolution - new_h - pad_h,
                                           pad_w, resolution - new_w - pad_w, cv2.BORDER_CONSTANT)
        aug_imgs.append(self._AugmentedImage(img=img.astype(np.float32) / 255.0, scale=scale))
        if augment_pyramid:
            for pyramid_scale in [0.5, 2.0]:
                scaled_img = cv2.resize(img, None, fx=pyramid_scale, fy=pyramid_scale)
                aug_imgs.append(self._AugmentedImage(img=scaled_img.astype(np.float32) / 255.0,
                                                  scale=scale * pyramid_scale))
        return aug_imgs

class TDDFAV3:
    @dataclass(frozen=True)
    class ExtractResult:
        anno_lmrks_ysa_range: np.ndarray
        anno_lmrks_2d68: np.ndarray
        anno_pose: Tuple[float, float, float]

    @staticmethod
    def get_available_devices() -> List[str]:
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        return providers

    def __init__(self, device: str):
        if device not in self.get_available_devices():
            raise Exception(f"Device {device} not in available devices: {self.get_available_devices()}")
        model_path = Path('/content/FaceExtractor/dist/TDDFAV3.onnx')
        if not model_path.exists():
            raise FileNotFoundError(f"TDDFAV3.onnx not found at {model_path}")
        self._sess = ort.InferenceSession(str(model_path), providers=[device])
        self._input_name = self._sess.get_inputs()[0].name
        self._camera_distance = 10.0
        self._world2view_proj = np.array([
            [1015.0, 0, 0],
            [0, 1015.0, 0],
            [112.0, 112.0, 1]
        ], dtype=np.float32)
        self._base_68_pts = np.array([
            [-7.31536686e-01,  2.18324587e-01, -5.90363741e-01],
            [-7.12017059e-01,  2.05892213e-02, -5.66208839e-01],
            [-6.73399091e-01, -1.59576654e-01, -5.48621714e-01],
            [-6.34890258e-01, -3.22125107e-01, -5.10903060e-01],
            [-5.79798579e-01, -4.96289551e-01, -4.19760287e-01],
            [-4.81546730e-01, -6.35660410e-01, -2.58796811e-01],
            [-3.65231335e-01, -7.19112933e-01, -6.17607236e-02],
            [-2.16505706e-01, -7.86682427e-01,  1.23575926e-01],
            [ 2.67814938e-03, -8.24886858e-01,  1.96671784e-01],
            [ 2.22356066e-01, -7.84747124e-01,  1.23866975e-01],
            [ 3.70815665e-01, -7.19669700e-01, -6.14449978e-02],
            [ 4.86512244e-01, -6.38140500e-01, -2.58221686e-01],
            [ 5.84084928e-01, -5.00324249e-01, -4.18844312e-01],
            [ 6.39234543e-01, -3.27089548e-01, -5.09994268e-01],
            [ 6.75870717e-01, -1.63255274e-01, -5.48786998e-01],
            [ 7.11227357e-01,  1.79298557e-02, -5.67488015e-01],
            [ 7.29938626e-01,  2.17761174e-01, -5.88844717e-01],
            [-5.71316302e-01,  4 | 34932321e-01,  3.62963080e-02],
            [-4.88676876e-01,  4.98392701e-01,  1.56470120e-01],
            [-3.82955074e-01,  5.19677103e-01,  2.39974141e-01],
            [-2.81491995e-01,  5.10921121e-01,  2.91035414e-01],
            [-1.89839065e-01,  4.85920191e-01,  3.15223455e-01],
            [ 1.87804177e-01,  4.88169491e-01,  3.14644337e-01],
            [ 2.79995173e-01,  5.13860941e-01,  2.89511085e-01],
            [ 3.81981879e-01,  5.23234010e-01,  2.37666845e-01],
            [ 4.89439726e-01,  5.01603365e-01,  1.55025303e-01],
            [ 5.72133660e-01,  4.36524928e-01,  3.49660516e-02],
            [ 1.76257803e-03,  2.96220154e-01,  3.45770597e-01],
            [ 2.21833796e-03,  1.75731063e-01,  4.35879111e-01],
            [ 3.29834060e-03,  5.63835837e-02,  5.29183626e-01],
            [ 3.25349881e-03, -4.61793244e-02,  5.52442431e-01],
            [-1.18324623e-01, -1.37969166e-01,  3.28948617e-01],
            [-6.76081181e-02, -1.47517920e-01,  3.74784112e-01],
            [ 1.47828390e-03, -1.60410553e-01,  3.95576954e-01],
            [ 6.99714422e-02, -1.47369280e-01,  3.74610901e-01],
            [ 1.20117076e-01, -1.37369201e-01,  3.28538775e-01],
            [-4.30975229e-01,  2.91965753e-01,  1.04033232e-01],
            [-3.67296219e-01,  3.31681073e-01,  1.86997056e-01],
            [-2.76482165e-01,  3.32692534e-01,  1.89165771e-01],
            [-1.91997916e-01,  2.88755804e-01,  1.63525820e-01],
            [-2.68532574e-01,  2.67138541e-01,  1.85307682e-01],
            [-3.63204300e-01,  2.62323439e-01,  1.62104011e-01],
            [ 1.87850699e-01,  2.88736135e-01,  1.60180151e-01],
            [ 2.72823513e-01,  3.33718687e-01,  1.85039103e-01],
            [ 3.65930617e-01,  3.31698567e-01,  1.84490025e-01],
            [ 4.31747049e-01,  2.90704608e-01,  1.03410363e-01],
            [ 3.62510055e-01,  2.63633072e-01,  1.61344767e-01],
            [ 2.66362846e-01,  2.67252117e-01,  1.83337152e-01],
            [-2.52169281e-01, -3.81339163e-01,  2.24057317e-01],
            [-1.62668362e-01, -3.21485281e-01,  3.36322904e-01],
            [-5.57506531e-02, -2.82682508e-01,  3.96588445e-01],
            [ 1.30601926e-03, -2.94665217e-01,  4.02246952e-01],
            [ 5.80654554e-02, -2.82752544e-01,  3.96721244e-01],
            [ 1.65142193e-01, -3.21170121e-01,  3.35649371e-01],
            [ 2.48466194e-01, -3.81282359e-01,  2.22357690e-01],
            [ 1.59650698e-01, -4.18565661e-01,  3.16695571e-01],
            [ 8.32042322e-02, -4.45940644e-01,  3.59758258e-01],
            [ 1.67942536e-03, -4.50652659e-01,  3.69605184e-01],
            [-7.98306689e-02, -4.45663661e-01,  3.60788703e-01],
            [-1.55356705e-01, -4.18663234e-01,  3.17556500e-01],
            [-2.27011800e-01, -3.75459403e-01,  2.31117964e-01],
            [-7.30654746e-02, -3.50188583e-01,  3.43461752e-01],
            [ 3.90908914e-04, -3.49111587e-01,  3.61925006e-01],
            [ 7.40414932e-02, -3.50820452e-01,  3.43598366e-01],
            [ 2.32114658e-01, -3.76261741e-01,  2.27939427e-01],
            [ 7.26732165e-02, -3.66326064e-01,  3.43837738e-01],
            [ 8.06166092e-04, -3.69369358e-01,  3.52605700e-01],
            [-7.08143190e-02, -3.65602463e-01,  3.43259454e-01]
        ], dtype=np.float32)
        self._68_nm_idxs = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45]

    @property
    def input_size(self) -> int:
        return 224

    def extract(self, img: np.ndarray) -> 'TDDFAV3.ExtractResult':
        H, W, _ = img.shape
        input_size = self.input_size
        h_scale = H / input_size
        w_scale = W / input_size
        feed_img = cv2.resize(img, (input_size, input_size))
        feed_img = np.transpose(feed_img, (2, 0, 1))[None, ...]
        w_68_pts, = self._sess.run(None, {self._input_name: feed_img})
        w_68_pts = w_68_pts[0]
        v_68_pts = self._project_w_pts(w_68_pts) * np.array([w_scale, h_scale])
        mat = self._estimate_affine(self._base_68_pts[self._68_nm_idxs], w_68_pts[self._68_nm_idxs])
        points = np.array([
            [0, -0.13, -0.1],
            [0, 0.53, -0.1],
            [0, -0.13, -0.1 + 1.0],
            [0, 0.53, -0.1 + 1.0]
        ])
        p0c, p0u, p1c, p1u = self._apply_affine(mat, points)
        p0cp0u_dist = np.linalg.norm(p0u - p0c)
        p1cp1u_dist = np.linalg.norm(p1u - p1c)
        vp0c, vp0cu, vp0u, vp1c, vp1cu, vp1u = self._project_w_pts([
            p0c, p0c + [0, -p0cp0u_dist, 0], p0u,
            p1c, p1c + [0, -p1cp1u_dist, 0], p1u
        ]) * np.array([w_scale, h_scale])
        vp0d = np.linalg.norm(vp0cu - vp0c)
        vp1d = np.linalg.norm(vp1cu - vp1c)
        vp0n = (vp0u - vp0c) / np.linalg.norm(vp0u - vp0c)
        vp1n = (vp1u - vp1c) / np.linalg.norm(vp1u - vp1c)
        ysa_range = np.array([
            vp0c + vp0n * vp0d,
            vp0c - vp0n * vp0d,
            vp1c + vp1n * vp1d,
            vp1c - vp1n * vp1d
        ])
        cp_w, zp_w = self._apply_affine(mat, np.array([[0, 0, 0], [0, 0, 1]]))
        cpzp_w = zp_w - cp_w
        d = np.linalg.norm(cpzp_w)
        pitch_rad = np.arcsin(cpzp_w[1] / d)
        yaw_rad = np.arcsin(-cpzp_w[0] / d)
        anno_pose = (pitch_rad, yaw_rad, 0.0)
        return self.ExtractResult(
            anno_lmrks_ysa_range=ysa_range,
            anno_lmrks_2d68=v_68_pts,
            anno_pose=anno_pose
        )

    def _project_w_pts(self, w_pts: np.ndarray) -> np.ndarray:
        cam_pts = w_pts * np.array([1, 1, -1])
        cam_pts += np.array([0, 0, self._camera_distance])
        view_pts = cam_pts @ self._world2view_proj
        view_pts = view_pts[..., :2] / view_pts[..., 2:]
        view_pts *= np.array([1, -1])
        view_pts += np.array([0, self.input_size])
        return view_pts

    def _estimate_affine(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        n = len(src_pts)
        A = np.zeros((3 * n, 12))
        b = np.zeros(3 * n)
        for i in range(n):
            x, y, z = src_pts[i]
            A[3 * i, 0:3] = [x, y, z]
            A[3 * i, 3] = 1
            A[3 * i + 1, 4:7] = [x, y, z]
            A[3 * i + 1, 7] = 1
            A[3 * i + 2, 8:11] = [x, y, z]
            A[3 * i + 2, 11] = 1
            b[3 * i:3 * i + 3] = dst_pts[i]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        mat = np.eye(4)
        mat[:3, :4] = x.reshape(3, 4)
        return mat

    def _apply_affine(self, mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
        transformed = pts_hom @ mat.T
        return transformed[:, :3]

def ema_smooth_all_landmarks(landmarks, alpha=0.8, scene_cut_flags=None):
    n = len(landmarks)
    landmarks_array = np.array(landmarks, dtype=np.float32)
    smoothed = np.zeros_like(landmarks_array)
    valid_mask = np.all(np.isfinite(landmarks_array), axis=(1, 2))
    prev = None
    for i in range(n):
        if not valid_mask[i]:
            smoothed[i] = np.nan
            continue
        curr = landmarks_array[i]
        if prev is None or (scene_cut_flags is not None and scene_cut_flags[i]):
            smoothed[i] = curr
            prev = curr
        else:
            smoothed[i] = alpha * prev + (1 - alpha) * curr
            prev = smoothed[i]
    return [smoothed[i] if valid_mask[i] else None for i in range(n)]

@jit(nopython=True)
def kalman_step(x, P, F, Q, H, R, meas, initialized):
    if not initialized:
        if meas is not None:
            x[0, 0] = meas[0]
            x[1, 0] = meas[1]
            initialized = True
            return meas, x, P, initialized
    x = F @ x
    P = F @ P @ F.T + Q
    if meas is not None:
        z = np.array([[meas[0]], [meas[1]]])
        yk = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ yk
        P = (np.eye(4) - K @ H) @ P
    return (x[0, 0], x[1, 0]), x, P, initialized

class Kalman2D:
    def __init__(self, q_factor=0.01, r_factor=1.0):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.Q = np.eye(4, dtype=np.float32) * q_factor
        self.R = np.eye(2, dtype=np.float32) * r_factor
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.H = np.zeros((2, 4), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.initialized = False

    def init_state(self, x, y):
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.initialized = True

    def step(self, meas):
        if not self.initialized:
            if meas is not None:
                self.init_state(*meas)
                return meas
            return None
        out, self.x, self.P, self.initialized = kalman_step(
            self.x, self.P, self.F, self.Q, self.H, self.R,
            meas if meas is not None else None, self.initialized
        )
        return out

def kalman_smooth_all_landmarks(landmarks, q_factor=0.01, r_factor=1.0, scene_cut_flags=None):
    n_frames = len(landmarks)
    if n_frames == 0:
        return []
    kfs = [Kalman2D(q_factor, r_factor) for _ in range(68)]
    smoothed = [None] * n_frames
    for i in range(n_frames):
        if landmarks[i] is None:
            smoothed[i] = None
            for k in range(68):
                kfs[k].step(None)
            continue
        if scene_cut_flags is not None and scene_cut_flags[i]:
            smoothed[i] = landmarks[i]
            for k in range(68):
                kfs[k].init_state(landmarks[i][k][0], landmarks[i][k][1])
        else:
            new_points = np.zeros((68, 2), dtype=np.float32)
            for k in range(68):
                meas = landmarks[i][k]
                out = kfs[k].step(meas)
                new_points[k] = out if out is not None else meas
            smoothed[i] = new_points
    return smoothed

def compute_histogram_diff(img1, img2):
    if img1 is None or img2 is None:
        return 1.0
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def save_checkpoint(frame_data, scene_cut_flags, config, output_folder, checkpoint_path="checkpoint.pkl"):
    with open(os.path.join(output_folder, checkpoint_path), 'wb') as f:
        pickle.dump({
            'frame_data': frame_data,
            'scene_cut_flags': scene_cut_flags,
            'config': config
        }, f)

def load_checkpoint(output_folder, checkpoint_path="checkpoint.pkl"):
    path = os.path.join(output_folder, checkpoint_path)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['frame_data'], data['scene_cut_flags'], data['config']
    return None, None, None

def main():
    input_folder = input("Enter input folder path: ").strip()
    if not os.path.isdir(input_folder):
        print("Invalid input folder.")
        return
    output_folder = input("Enter output folder path: ").strip()
    os.makedirs(output_folder, exist_ok=True)
    frame_data, scene_cut_flags, saved_config = load_checkpoint(output_folder)
    use_checkpoint_config = False
    if saved_config is not None:
        print("Found existing checkpoint with saved configuration.")
        response = input("Use checkpoint configuration? (y/n, default n): ").strip().lower()
        use_checkpoint_config = response == 'y'
    if use_checkpoint_config:
        print("Loading configuration from checkpoint.")
        config = saved_config
    else:
        resolutions = {"256": 256, "320": 320, "384": 384, "512": 512, "640": 640, "768": 768, "1024": 1024}
        print("Available resolutions: 256, 320, 384, 512, 640, 768, 1024")
        out_size = input("Enter resolution (default 512): ").strip() or "512"
        if out_size not in resolutions:
            print("Invalid resolution, using 512.")
            out_size = 512
        else:
            out_size = resolutions[out_size]
        face_types = {"1": "full_face", "2": "whole_face", "3": "head", "4": "mve_custom"}
        print("Face types: 1) Full Face, 2) Whole Face, 3) Head, 4) Custom")
        face_choice = input("Enter face type (1-4, default 2): ").strip() or "2"
        chosen_face_type = face_types.get(face_choice, "whole_face")
        margin_fraction = float(input("Enter margin fraction (default 0.15): ").strip() or 0.15)
        shift_fraction = float(input("Enter upward shift fraction (default 0.10): ").strip() or 0.10)
        if chosen_face_type == "whole_face":
            margin_fraction += 0.10
        elif chosen_face_type == "head":
            margin_fraction += 0.30
            shift_fraction += 0.05
        smoothing_enabled = input("Enable smoothing? (y/n, default n): ").strip().lower() == 'y'
        smoothing_mode = "none"
        alpha = 0.80
        kalman_q = 0.001
        kalman_r = 3.0
        if smoothing_enabled:
            print("Smoothing modes: 1) EMA, 2) Kalman")
            mode_choice = input("Enter smoothing mode (1-2, default 1): ").strip() or "1"
            smoothing_mode = "ema" if mode_choice == "1" else "kalman"
            if smoothing_mode == "ema":
                alpha = float(input("Enter EMA alpha (0-1, default 0.80): ").strip() or 0.80)
            else:
                kalman_q = float(input("Enter Kalman Q (default 0.001): ").strip() or 0.001)
                kalman_r = float(input("Enter Kalman R (default 3.0): ").strip() or 3.0)
        scene_cut_thresh = float(input("Enter scene cut threshold (0-1, default 0.7): ").strip() or 0.7)
        batch_size = int(input("Enter batch size for frame processing (default 1): ").strip() or 1)
        if batch_size < 1:
            print("Invalid batch size, using 1.")
            batch_size = 1
        downscale_enabled = input("Downscale frames for face detection? (y/n, default n): ").strip().lower() == 'y'
        downscale_factor = 1.0
        if downscale_enabled:
            try:
                downscale_factor = float(input("Enter downscale factor (e.g., 2 for 50% resolution, default 2): ").strip() or 2)
                if downscale_factor < 1.0:
                    print("Downscale factor must be >= 1. Using no downscaling.")
                    downscale_factor = 1.0
            except ValueError:
                print("Invalid downscale factor. Using no downscaling.")
                downscale_factor = 1.0
        config = {
            'input_folder': input_folder,
            'output_folder': output_folder,
            'out_size': out_size,
            'face_type': chosen_face_type,
            'margin_fraction': margin_fraction,
            'shift_fraction': shift_fraction,
            'smoothing': {
                'enabled': smoothing_enabled,
                'mode': smoothing_mode,
                'alpha': alpha,
                'kalman_q': kalman_q,
                'kalman_r': kalman_r
            },
            'scene_cut_thresh': scene_cut_thresh,
            'batch_size': batch_size,
            'downscale_factor': downscale_factor
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    ort_device = 'CUDAExecutionProvider' if device == "cuda" and 'CUDAExecutionProvider' in TDDFAV3.get_available_devices() else 'CPUExecutionProvider'
    try:
        yolo_model = YoloV7Face(lib_ort.DeviceRef(ort_device))
    except Exception as e:
        print(f"Failed to initialize YoloV7Face: {e}")
        sys.exit(1)
    try:
        tddfa_model = TDDFAV3(ort_device)
    except Exception as e:
        print(f"Failed to initialize TDDFAV3: {e}")
        sys.exit(1)
    input_folder = config['input_folder']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    image_files.sort()
    total_count = len(image_files)
    if total_count == 0:
        print("No images found.")
        return
    start_time = time.time()
    if frame_data is None:
        frame_data = [[] for _ in range(total_count)]
        scene_cut_flags = [False] * total_count
        prev_img = None
        for batch_start in range(0, total_count, config['batch_size']):
            batch_end = min(batch_start + config['batch_size'], total_count)
            batch_files = image_files[batch_start:batch_end]
            batch_images = []
            batch_indices = list(range(batch_start, batch_end))
            for fname in batch_files:
                path = os.path.join(input_folder, fname)
                img = cv2.imread(path)
                if img is None:
                    print(f"Skipping {fname}: can't read.")
                    batch_images.append(None)
                else:
                    if config['downscale_factor'] != 1.0:
                        new_height = int(img.shape[0] / config['downscale_factor'])
                        new_width = int(img.shape[1] / config['downscale_factor'])
                        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    batch_images.append(img)
            for i, (img, fname, global_idx) in enumerate(zip(batch_images, batch_files, batch_indices)):
                if img is None:
                    continue
                if prev_img is not None:
                    hist_diff = compute_histogram_diff(prev_img, img)
                    if hist_diff > config['scene_cut_thresh']:
                        scene_cut_flags[global_idx] = True
                prev_img = img
                try:
                    fimage = FImage(img)
                    faces = yolo_model.detect(fimage, minimum_confidence=0.5, nms_threshold=0.7, min_face_size=20)
                    boxes = [[int(face.rect.left), int(face.rect.top), int(face.rect.right), int(face.rect.bottom)]
                             for face in faces]
                except Exception as e:
                    print(f"Error detecting faces in {fname}: {e}")
                    boxes = []
                if not boxes:
                    print(f"No faces detected in {fname}")
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        print(f"Box too small in {fname}, skipping.")
                        continue
                    padding = int(max(x2 - x1, y2 - y1) * 0.2)
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(img.shape[1], x2 + padding)
                    crop_y2 = min(img.shape[0], y2 + padding)
                    face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    if face_crop.size == 0:
                        print(f"Invalid crop in {fname}, skipping.")
                        continue
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    try:
                        result = tddfa_model.extract(face_crop_rgb)
                        lmrk = result.anno_lmrks_2d68
                        lmrk[:, 0] += crop_x1
                        lmrk[:, 1] += crop_y1
                        if config['downscale_factor'] != 1.0:
                            lmrk *= config['downscale_factor']
                        lmrk = lmrk.astype(np.float32)
                        minx, maxx = int(np.min(lmrk[:, 0])), int(np.max(lmrk[:, 0]))
                        miny, maxy = int(np.min(lmrk[:, 1])), int(np.max(lmrk[:, 1]))
                        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                        area = (maxx - minx) * (maxy - miny)
                        frame_data[global_idx].append({
                            'landmarks': lmrk,
                            'bbox_center': (cx, cy),
                            'area': area
                        })
                    except Exception as e:
                        print(f"Error extracting landmarks in {fname}: {e}")
            print(f"Pass 1/2: {batch_end}/{total_count} frames processed.")
        save_checkpoint(frame_data, scene_cut_flags, config, output_folder)
    if config['smoothing']['enabled'] and config['smoothing']['mode'] != "none":
        for i in range(total_count):
            for face_data in frame_data[i]:
                landmarks = face_data['landmarks']
                landmarks_list = [landmarks]
                if config['smoothing']['mode'] == "ema":
                    smoothed = ema_smooth_all_landmarks(
                        landmarks_list, config['smoothing']['alpha'], [scene_cut_flags[i]]
                    )
                else:
                    smoothed = kalman_smooth_all_landmarks(
                        landmarks_list, config['smoothing']['kalman_q'], config['smoothing']['kalman_r'],
                        [scene_cut_flags[i]]
                    )
                face_data['landmarks'] = smoothed[0] if smoothed[0] is not None else landmarks
    pass2_count = 0
    for batch_start in range(0, total_count, config['batch_size']):
        batch_end = min(batch_start + config['batch_size'], total_count)
        batch_files = image_files[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        batch_images = []
        for fname in batch_files:
            path = os.path.join(input_folder, fname)
            img = cv2.imread(path)
            batch_images.append(img if img is not None else None)
        for i, (img, fname, global_idx) in enumerate(zip(batch_images, batch_files, batch_indices)):
            if img is None or not frame_data[global_idx]:
                continue
            h, w, _ = img.shape
            for face_idx, face_data in enumerate(frame_data[global_idx]):
                lmrk = face_data['landmarks']
                left_eye = np.mean(lmrk[36:42], axis=0)
                right_eye = np.mean(lmrk[42:48], axis=0)
                nose_center = np.mean(lmrk[27:36], axis=0)
                mouth_center = np.mean(lmrk[48:68], axis=0)
                face_center = (0.4 * (left_eye + right_eye) / 2.0 + 0.3 * nose_center + 0.3 * mouth_center)
                cx, cy = face_center
                try:
                    hull = ConvexHull(lmrk)
                    hull_points = lmrk[hull.vertices]
                    distances = [
                        np.hypot(hull_points[i][0] - hull_points[j][0], hull_points[i][1] - hull_points[j][1])
                        for i in range(len(hull_points)) for j in range(i + 1, len(hull_points))
                    ]
                    max_dist = max(distances) if distances else 100.0
                except:
                    max_dist = 100.0
                side = max_dist * 1.3 * (1.0 + config['margin_fraction'])
                shift_up = int(config['shift_fraction'] * side)
                cy -= shift_up
                half_side = side / 2.0
                sq_sx, sq_sy = cx - half_side, cy - half_side
                sq_ex, sq_ey = cx + half_side, cy + half_side
                sq_sx, sq_sy = max(0, sq_sx), max(0, sq_sy)
                sq_ex, sq_ey = min(w, sq_ex), min(h, sq_ey)
                if sq_ex - sq_sx < 20 or sq_ey - sq_sy < 20:
                    print(f"Box too small for face {face_idx + 1} in {fname}, skipping warp.")
                    continue
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                scale = config['out_size'] / side
                M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
                M_rot[0, 2] += (config['out_size'] / 2.0 - cx)
                M_rot[1, 2] += (config['out_size'] / 2.0 - cy)
                warped_face = cv2.warpAffine(img, M_rot, (config['out_size'], config['out_size']), flags=cv2.INTER_LANCZOS4)
                ones = np.ones((68, 1), dtype=np.float32)
                lmrk_2d = np.hstack([lmrk, ones])
                warped_2d = (lmrk_2d @ M_rot.T)[:, :2]
                base_name, _ = os.path.splitext(fname)
                out_name = f"{base_name}_face{face_idx + 1}_dfl.jpg"
                out_path = os.path.join(output_folder, out_name)
                cv2.imwrite(out_path, warped_face, [cv2.IMWRITE_JPEG_QUALITY, 100])
                dflimg = DFLJPG.load(out_path)
                if dflimg is None:
                    print(f"Failed to embed metadata in {out_name}.")
                    continue
                dflimg.set_face_type(config['face_type'])
                dflimg.set_landmarks(warped_2d.tolist())
                dflimg.set_source_rect([int(sq_sx), int(sq_sy), int(sq_ex), int(sq_ey)])
                dflimg.set_source_landmarks(lmrk.tolist())
                dflimg.set_image_to_face_mat(M_rot.flatten().tolist())
                dflimg.set_source_filename(fname)
                dflimg.save()
                pass2_count += 1
        print(f"Pass 2/2: {batch_end}/{total_count} frames processed.")
    total_elapsed = time.time() - start_time
    total_detected = sum(len(frame) for frame in frame_data)
    print(f"Done! Detected {total_detected} faces across {total_count} images, warped {pass2_count} faces.")
    print(f"Total time: {total_elapsed:.1f}s, Saved to: {output_folder}")

if __name__ == "__main__":
    main()
