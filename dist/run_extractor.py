import sys
import os
import time
import cv2
import numpy as np
import yaml
import pickle
from scipy.spatial import ConvexHull
from torch.cuda.amp import autocast
from numba import jit

try:
    from DFLJPG import DFLJPG
except ImportError:
    print("DFLJPG module not found. Ensure it's in your environment.")
    sys.exit(1)

try:
    import face_alignment
    import torch
except ImportError:
    print("You must install face-alignment and torch:\n"
          "  pip install face-alignment torch torchvision")
    sys.exit(1)

# EMA Smoothing (Vectorized)
def ema_smooth_all_landmarks(landmarks, alpha=0.8, scene_cut_flags=None):
    n = len(landmarks)
    landmarks_array = np.array(landmarks, dtype=np.float32)  # Shape: (n, 68, 2)
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

# Kalman Filter (Numba-accelerated)
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

# Scene Cut Detection
def compute_histogram_diff(img1, img2):
    if img1 is None or img2 is None:
        return 1.0
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Checkpointing Functions
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
    # Prompt for input folder first
    input_folder = input("Enter input folder path: ").strip()
    if not os.path.isdir(input_folder):
        print("Invalid input folder.")
        return
    
    # Prompt for output folder next
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
        # Interactive prompting (adjusted to include input_folder)
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
        
        smoothing_enabled = input("Enable smoothing? (y/n, default n): ").strip().lower() == "y"
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
        
        downscale_enabled = input("Downscale frames for face detection? (y/n, default n): ").strip().lower() == "y"
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

    # Device and face alignment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=str(device),
        face_detector='sfd'
    )

    # Load images
    input_folder = config['input_folder']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    image_files.sort()
    total_count = len(image_files)
    if total_count == 0:
        print("No images found.")
        return

    # Initialize or load frame data
    start_time = time.time()
    if frame_data is None:
        frame_data = [[] for _ in range(total_count)]
        scene_cut_flags = [False] * total_count

        # Pass 1: Face detection with mixed precision
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

                # Scene cut detection
                if prev_img is not None:
                    hist_diff = compute_histogram_diff(prev_img, img)
                    if hist_diff > config['scene_cut_thresh']:
                        scene_cut_flags[global_idx] = True
                prev_img = img

                # Face detection with mixed precision
                try:
                    with autocast():
                        results = fa.get_landmarks_from_image(img)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    results = None

                if not results:
                    print(f"No faces detected in {fname}")
                else:
                    for lmrk in results:
                        lmrk = lmrk.astype(np.float32)
                        if config['downscale_factor'] != 1.0:
                            lmrk *= config['downscale_factor']
                        minx, maxx = int(np.min(lmrk[:, 0])), int(np.max(lmrk[:, 0]))
                        miny, maxy = int(np.min(lmrk[:, 1])), int(np.max(lmrk[:, 1]))
                        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                        area = (maxx - minx) * (maxy - miny)
                        frame_data[global_idx].append({
                            'landmarks': lmrk,
                            'bbox_center': (cx, cy),
                            'area': area
                        })

            print(f"Pass 1/2: {batch_end}/{total_count} frames processed.")

        # Save checkpoint after detection
        save_checkpoint(frame_data, scene_cut_flags, config, output_folder)

    # Smoothing
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

    # Pass 2: Warp & Save
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

                # Optimized face center
                left_eye = np.mean(lmrk[36:42], axis=0)
                right_eye = np.mean(lmrk[42:48], axis=0)
                nose_center = np.mean(lmrk[27:36], axis=0)
                mouth_center = np.mean(lmrk[48:68], axis=0)
                face_center = (0.4 * (left_eye + right_eye) / 2.0 + 0.3 * nose_center + 0.3 * mouth_center)
                cx, cy = face_center

                # Face size using convex hull
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

                # Apply upward shift
                shift_up = int(config['shift_fraction'] * side)
                cy -= shift_up

                # Define square region
                half_side = side / 2.0
                sq_sx, sq_sy = cx - half_side, cy - half_side
                sq_ex, sq_ey = cx + half_side, cy + half_side
                sq_sx, sq_sy = max(0, sq_sx), max(0, sq_sy)
                sq_ex, sq_ey = min(w, sq_ex), min(h, sq_ey)

                if sq_ex - sq_sx < 20 or sq_ey - sq_sy < 20:
                    print(f"Box too small for face {face_idx + 1} in {fname}, skipping warp.")
                    continue

                # Rotation angle
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi

                # Rotation matrix
                scale = config['out_size'] / side
                M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
                M_rot[0, 2] += (config['out_size'] / 2.0 - cx)
                M_rot[1, 2] += (config['out_size'] / 2.0 - cy)

                # Warp image
                warped_face = cv2.warpAffine(img, M_rot, (config['out_size'], config['out_size']), flags=cv2.INTER_LANCZOS4)

                # Transform landmarks
                ones = np.ones((68, 1), dtype=np.float32)
                lmrk_2d = np.hstack([lmrk, ones])
                warped_2d = (lmrk_2d @ M_rot.T)[:, :2]

                # Save output
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
