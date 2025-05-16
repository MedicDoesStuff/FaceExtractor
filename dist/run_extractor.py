import sys
import os
import time
import cv2
import numpy as np
import yaml
import pickle
import requests
from scipy.spatial import ConvexHull
from torch.cuda.amp import autocast
from numba import jit
from ultralytics import YOLO
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
        model_path = Path(__file__).parent / 'TDDFAV3.onnx'
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
            [-5.71316302e-01,  4.34932321e-01,  3.62963080e-02],
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
        mat = self._estimate_affine(self._base_68_pts[self._68_nm_idxs], w_68_pts[self._68_nm_idxs])
        cp_w, zp_w = self._apply_affine(mat, np.array([[0, 0, 0], [0, 0, 1]]))
        cpzp_w = zp_w - cp_w
        d = np.linalg.norm(cpzp_w)
        yaw_rad = np.arcsin(-cpzp_w[0] / d)
        pitch_rad = np.arcsin(cpzp_w[1] / d)
        stable_indices = list(range(0, 17)) + list(range(27, 36))
        mat = self._estimate_affine(self._base_68_pts[stable_indices], w_68_pts[stable_indices])
        v_68_pts = self._project_w_pts(w_68_pts) * np.array([w_scale, h_scale])
        v_68_pts = self._refine_landmarks(v_68_pts, stable_indices)
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
        roll_rad = 0.0
        anno_pose = (pitch_rad, yaw_rad, roll_rad)
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

    def _refine_landmarks(self, landmarks: np.ndarray, stable_indices: List[int]) -> np.ndarray:
        refined = landmarks.copy()
        for i in range(68):
            if i not in stable_indices:
                distances = np.linalg.norm(landmarks[stable_indices] - landmarks[i], axis=1)
                nearest_idx = stable_indices[np.argmin(distances)]
                refined[i] = 0.75 * refined[i] + 0.25 * landmarks[nearest_idx]
        return refined

def ema_smooth_all_landmarks(landmarks, alpha=0.85, scene_cut_flags=None):
    n = len(landmarks)
    landmarks_array = np.array(landmarks, dtype=np.float32)
    smoothed = np.zeros_like(landmarks_array)
    valid_mask = np.all(np.isfinite(landmarks_array), axis=(1, 2))
    prev = None
    for i in range(n):
        if not valid_mask[i]:
            smoothed[i] = np.nan
            prev = None
            continue
        curr = landmarks_array[i]
        if prev is None or (scene_cut_flags is not None and scene_cut_flags[i]):
            smoothed[i] = curr
            prev = curr
        else:
            smoothed[i] = alpha * prev + (1 - alpha) * curr
            prev = smoothed[i]
    return [smoothed[i] if valid_mask[i] else None for i in range(n)]

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

def download_and_combine_onnx_parts(output_path="TDDFAV3.onnx"):
    urls = [
        "https://github.com/iperov/DeepixLab/raw/refs/heads/master/DeepixLab/modelhub/onnx/TDDFAV3/TDDFAV3.onnx.part0",
        "https://github.com/iperov/DeepixLab/raw/refs/heads/master/DeepixLab/modelhub/onnx/TDDFAV3/TDDFAV3.onnx.part1"
    ]
    temp_files = ["part0.onnx", "part1.onnx"]
    for url, temp_file in zip(urls, temp_files):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Failed to download {url}")
    with open(output_path, 'wb') as f_out:
        for temp_file in temp_files:
            with open(temp_file, 'rb') as f_in:
                f_out.write(f_in.read())
            os.remove(temp_file)
    print(f"Combined ONNX model saved to {output_path}")

def download_yolo_pt_model(output_path="models/yolov11n-face.pt"):
    if os.path.exists(output_path):
        print(f"YOLO PT model already exists at {output_path}")
        return True
    url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading YOLO PT model from {url}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"YOLO PT model saved to {output_path}")
            return True
        else:
            print(f"Failed to download YOLO PT model: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading YOLO PT model: {e}")
        return False

def get_adaptive_rotation_angle(landmarks, yaw_deg):
    def compute_angle(p1, p2):
        """Compute angle in degrees between two points, handling edge cases."""
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        return angle if np.isfinite(angle) else None

    def get_frontal_rotation(landmarks):
        """Compute rotation for frontal faces using eye and mouth landmarks."""
        pairs = [
            (landmarks[36], landmarks[45]),  # Outer eye corners
            (landmarks[39], landmarks[42]),  # Inner eye corners
            (landmarks[48], landmarks[54]),  # Mouth corners
        ]
        angles = []
        for p1, p2 in pairs:
            angle = compute_angle(p1, p2)
            if angle is not None and abs(angle) < 45:
                angles.append(angle)
        if not angles:
            return 0.0
        return np.mean(angles)

    def get_side_rotation(landmarks):
        """Compute rotation for side profiles using nose, jaw, and mouth."""
        pairs = [
            (landmarks[30], landmarks[8]),   # Nose to chin
            (landmarks[30], landmarks[51]),  # Nose to mouth center
            (landmarks[3], landmarks[8]),    # Jaw point to chin
            (landmarks[13], landmarks[8]),   # Other jaw point to chin
        ]
        angles = []
        for p1, p2 in pairs:
            angle = compute_angle(p1, p2)
            if angle is not None and abs(angle - 90) < 45:
                angles.append(angle - 90.0)
        if not angles:
            return 0.0
        return np.mean(angles) * 0.3

    if yaw_deg < 30:
        return get_frontal_rotation(landmarks)
    elif yaw_deg > 75:
        side_angle = get_side_rotation(landmarks)
        if abs(side_angle) > 15:
            return side_angle
        return 0.0
    else:
        t = (yaw_deg - 30) / 45
        frontal_angle = get_frontal_rotation(landmarks)
        side_angle = get_side_rotation(landmarks)
        return (1 - t) * frontal_angle + t * side_angle

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
        margin_fraction = float(input("Enter margin fraction (default 0.2): ").strip() or 0.2)
        shift_fraction = float(input("Enter upward shift fraction (default 0.10): ").strip() or 0.10)
        if chosen_face_type == "whole_face":
            margin_fraction += 0.15
        elif chosen_face_type == "head":
            margin_fraction += 0.35
            shift_fraction += 0.05
        smoothing_enabled = input("Enable smoothing? (y/n, default y): ").strip().lower() != 'n'
        alpha = 0.85
        if smoothing_enabled:
            alpha = float(input("Enter EMA alpha (0-1, default 0.85): ").strip() or 0.85)
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
        debug_visualization = input("Enable debug visualization? (y/n, default y): ").strip().lower() != 'n'
        config = {
            'input_folder': input_folder,
            'output_folder': output_folder,
            'out_size': out_size,
            'face_type': chosen_face_type,
            'margin_fraction': margin_fraction,
            'shift_fraction': shift_fraction,
            'smoothing': {
                'enabled': smoothing_enabled,
                'mode': 'ema',
                'alpha': alpha
            },
            'scene_cut_thresh': scene_cut_thresh,
            'batch_size': batch_size,
            'downscale_factor': downscale_factor,
            'debug_visualization': debug_visualization
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    yolo_pt_path = os.path.join(os.getcwd(), "models", "yolov11n-face.pt")
    if not download_yolo_pt_model(yolo_pt_path):
        print("Failed to download YOLO PT model. Exiting.")
        sys.exit(1)
    try:
        yolo_model = YOLO(yolo_pt_path)
        yolo_model.to(device)
    except Exception as e:
        print(f"Failed to initialize YOLOv11-face with PT model: {e}")
        sys.exit(1)
    onnx_path = os.path.join(os.getcwd(), "TDDFAV3.onnx")
    if not os.path.exists(onnx_path):
        try:
            download_and_combine_onnx_parts(onnx_path)
        except Exception as e:
            print(f"Failed to download or combine TDDFAV3 ONNX parts: {e}")
            sys.exit(1)
    ort_device = 'CUDAExecutionProvider' if device == "cuda" and 'CUDAExecutionProvider' in TDDFAV3.get_available_devices() else 'CPUExecutionProvider'
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
                    results = yolo_model.predict(img, conf=0.35, iou=0.7)
                    boxes = []
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(np.int32)
                            boxes.append([x1, y1, x2, y2])
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
                    if x2 - x1 < 30 or y2 - y1 < 30:
                        print(f"Box too small in {fname}, expanding box.")
                        width = x2 - x1
                        height = y2 - y1
                        padding = max(width, height) * 0.8
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(img.shape[1], x2 + padding)
                        y2 = min(img.shape[0], y2 + padding)
                    padding = int(max(x2 - x1, y2 - y1) * 0.4)
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
                            'area': area,
                            'pose': result.anno_pose
                        })
                    except Exception as e:
                        print(f"Error extracting landmarks in {fname}: {e}")
            print(f"Pass 1/2: {batch_end}/{total_count} frames processed.")
        save_checkpoint(frame_data, scene_cut_flags, config, output_folder)
    if config['smoothing']['enabled']:
        for i in range(total_count):
            for face_data in frame_data[i]:
                landmarks = face_data['landmarks']
                landmarks_list = [landmarks]
                smoothed = ema_smooth_all_landmarks(
                    landmarks_list, config['smoothing']['alpha'], [scene_cut_flags[i]]
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
                yaw = face_data['pose'][1]
                face_center = np.mean(lmrk, axis=0)
                cx, cy = face_center
                minx, maxx = np.min(lmrk[:, 0]), np.max(lmrk[:, 0])
                miny, maxy = np.min(lmrk[:, 1]), np.max(lmrk[:, 1])
                width = maxx - minx
                height = maxy - miny
                yaw_deg = abs(yaw * 180.0 / np.pi)
                if yaw_deg < 30:
                    margin_factor = config['margin_fraction'] + 0.15
                elif yaw_deg > 60:
                    margin_factor = config['margin_fraction'] + 0.25
                else:
                    t = (yaw_deg - 30) / 30
                    margin_factor = config['margin_fraction'] + 0.15 + t * (0.25 - 0.15)
                margin_factor = min(margin_factor, 0.45)
                side = max(width, height) * (1.0 + margin_factor)
                if yaw_deg < 30:
                    scale_adjust = 1.1
                elif yaw_deg > 60:
                    scale_adjust = 1.2
                else:
                    t = (yaw_deg - 30) / 30
                    scale_adjust = 1.1 + t * (1.2 - 1.1)
                side *= scale_adjust
                yaw_shift = (yaw / abs(yaw)) * (width * 0.2) if yaw != 0 else 0
                cx += yaw_shift
                shift_up = int(min(config['shift_fraction'] * side, height * 0.3))
                cy -= shift_up
                angle = get_adaptive_rotation_angle(lmrk, yaw_deg)
                scale = config['out_size'] / side
                M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
                M[0, 2] += config['out_size'] / 2.0 - cx
                M[1, 2] += config['out_size'] / 2.0 - cy
                warped_face = cv2.warpAffine(img, M, (config['out_size'], config['out_size']), flags=cv2.INTER_LANCZOS4)
                ones = np.ones((68, 1), dtype=np.float32)
                lmrk_2d = np.hstack([lmrk, ones])
                warped_2d = (lmrk_2d @ M.T)[:, :2]
                if config.get('debug_visualization', False):
                    vis_img = img.copy()
                    for pt in lmrk:
                        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                    sq_sx, sq_sy = cx - side / 2.0, cy - side / 2.0
                    sq_ex, sq_ey = cx + side / 2.0, cy + side / 2.0
                    cv2.rectangle(vis_img, (int(sq_sx), int(sq_sy)), (int(sq_ex), int(sq_ey)), (255, 0, 0), 2)
                    base_name, _ = os.path.splitext(fname)
                    cv2.imwrite(os.path.join(output_folder, f"debug_{base_name}_face{face_idx + 1}.jpg"), vis_img)
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
                dflimg.set_source_rect([int(cx - side / 2.0), int(cy - side / 2.0), int(cx + side / 2.0), int(cy + side / 2.0)])
                dflimg.set_source_landmarks(lmrk.tolist())
                dflimg.set_image_to_face_mat(M.flatten().tolist())
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
