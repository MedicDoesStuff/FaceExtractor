import sys
import os
import time
import cv2
import numpy as np
import yaml

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

# EMA Smoothing and Kalman Filter remain unchanged
def ema_smooth_all_landmarks(landmarks, alpha=0.8, scene_cut_flags=None):
    n = len(landmarks)
    smoothed = [None] * n
    prev = None
    for i in range(n):
        if landmarks[i] is None:
            smoothed[i] = None
            continue
        if prev is None:
            smoothed[i] = landmarks[i]
            prev = landmarks[i]
        else:
            if scene_cut_flags is not None and scene_cut_flags[i]:
                smoothed[i] = landmarks[i]
                prev = landmarks[i]
            else:
                new_points = alpha * prev + (1 - alpha) * landmarks[i]
                smoothed[i] = new_points
                prev = new_points
    return smoothed

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

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y):
        z = np.array([[x], [y]], dtype=np.float32)
        yk = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ yk
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

    def step(self, meas):
        if not self.initialized:
            if meas is not None:
                self.init_state(*meas)
                return meas
            return None
        self.predict()
        if meas is not None:
            self.update(*meas)
        return (self.x[0, 0], self.x[1, 0])

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
            new_points = landmarks[i]
            for k in range(68):
                x, y = new_points[k]
                kfs[k].init_state(x, y)
            smoothed[i] = new_points
        else:
            new_points = np.zeros((68, 2), dtype=np.float32)
            for k in range(68):
                x, y = landmarks[i][k]
                out = kfs[k].step((x, y))
                new_points[k] = out if out is not None else [x, y]
            smoothed[i] = new_points
    return smoothed

def main():
    # Input/output folders
    input_folder = input("Enter input folder path: ").strip()
    if not os.path.isdir(input_folder):
        print("Invalid input folder.")
        return
    output_folder = input("Enter output folder path: ").strip()
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Resolution selection
    resolutions = {"256": 256, "320": 320, "384": 384, "512": 512, "640": 640, "768": 768, "1024": 1024}
    print("Available resolutions: 256, 320, 384, 512, 640, 768, 1024")
    out_size = input("Enter resolution (default 512): ").strip() or "512"
    if out_size not in resolutions:
        print("Invalid resolution, using 512.")
        out_size = 512
    else:
        out_size = resolutions[out_size]

    # Face type selection
    face_types = {"1": "full_face", "2": "whole_face", "3": "head", "4": "mve_custom"}
    print("Face types: 1) Full Face, 2) Whole Face, 3) Head, 4) Custom")
    face_choice = input("Enter face type (1-4, default 2): ").strip() or "2"
    chosen_face_type = face_types.get(face_choice, "whole_face")

    # Margin and shift adjustments
    margin_fraction = float(input("Enter margin fraction (default 0.15): ").strip() or 0.15)
    shift_fraction = float(input("Enter upward shift fraction (default 0.10): ").strip() or 0.10)
    if chosen_face_type == "whole_face":
        margin_fraction += 0.10
    elif chosen_face_type == "head":
        margin_fraction += 0.30
        shift_fraction += 0.05

    # Smoothing options
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

    # Scene cut threshold
    scene_cut_thresh = float(input("Enter scene cut threshold (default 50): ").strip() or 50.0)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize face alignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=str(device),
        face_detector='sfd'
    )

    # Load images
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    image_files.sort()
    total_count = len(image_files)
    if total_count == 0:
        print("No images found.")
        return

    # Store multiple faces per frame
    frame_data = [[] for _ in range(total_count)]
    scene_cut_flags = [False] * total_count

    start_time = time.time()

    # Pass 1: Detection of multiple faces
    for i, fname in enumerate(image_files):
        path = os.path.join(input_folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping {fname}: can't read.")
            continue

        try:
            results = fa.get_landmarks_from_image(img)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results = None

        if not results:
            print(f"No faces detected in {fname}")
        else:
            for lmrk in results:
                lmrk = lmrk.astype(np.float32)
                minx, maxx = int(np.min(lmrk[:, 0])), int(np.max(lmrk[:, 0]))
                miny, maxy = int(np.min(lmrk[:, 1])), int(np.max(lmrk[:, 1]))
                cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                area = (maxx - minx) * (maxy - miny)
                frame_data[i].append({
                    'landmarks': lmrk,
                    'bbox_center': (cx, cy),
                    'area': area
                })

        # Scene cut detection
        if i > 0 and frame_data[i] and frame_data[i - 1]:
            prev_face = max(frame_data[i - 1], key=lambda x: x['area']) if frame_data[i - 1] else None
            curr_face = max(frame_data[i], key=lambda x: x['area']) if frame_data[i] else None
            if prev_face and curr_face:
                px, py = prev_face['bbox_center']
                cx, cy = curr_face['bbox_center']
                dist = np.hypot(cx - px, cy - py)
                if dist > scene_cut_thresh:
                    scene_cut_flags[i] = True

        print(f"Pass 1/2: {i + 1}/{total_count} frames processed.")

    # Smoothing
    if smoothing_enabled and smoothing_mode != "none":
        for i in range(total_count):
            for face_data in frame_data[i]:
                landmarks = face_data['landmarks']
                landmarks_list = [landmarks]
                if smoothing_mode == "ema":
                    smoothed = ema_smooth_all_landmarks(landmarks_list, alpha, [scene_cut_flags[i]])
                else:
                    smoothed = kalman_smooth_all_landmarks(landmarks_list, kalman_q, kalman_r, [scene_cut_flags[i]])
                face_data['landmarks'] = smoothed[0] if smoothed[0] is not None else landmarks

    # Pass 2: Warp & Save multiple faces with centering on eyes, nose, mouth
    pass2_count = 0
    for i, fname in enumerate(image_files):
        if not frame_data[i]:
            continue

        path = os.path.join(input_folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping {fname}: can't re-read.")
            continue

        h, w, _ = img.shape
        for face_idx, face_data in enumerate(frame_data[i]):
            lmrk = face_data['landmarks']

            # Compute face center as the average of eyes, nose, and mouth
            left_eye = np.mean(lmrk[36:42], axis=0)  # Left eye landmarks (36-41)
            right_eye = np.mean(lmrk[42:48], axis=0)  # Right eye landmarks (42-47)
            eyes_center = (left_eye + right_eye) / 2.0
            nose_center = np.mean(lmrk[27:36], axis=0)  # Nose landmarks (27-35)
            mouth_center = np.mean(lmrk[48:68], axis=0)  # Mouth landmarks (48-67)
            face_center = (eyes_center + nose_center + mouth_center) / 3.0  # Average of the three
            cx, cy = face_center

            # Estimate face size
            distances = []
            for j in range(len(lmrk)):
                for k in range(j + 1, len(lmrk)):
                    dist = np.hypot(lmrk[j, 0] - lmrk[k, 0], lmrk[j, 1] - lmrk[k, 1])
                    distances.append(dist)
            max_dist = max(distances) if distances else 100.0
            side = max_dist * 1.3 * (1.0 + margin_fraction)

            # Apply upward shift
            shift_up = int(shift_fraction * side)
            cy -= shift_up

            # Define square region centered at face centroid
            half_side = side / 2.0
            sq_sx, sq_sy = cx - half_side, cy - half_side
            sq_ex, sq_ey = cx + half_side, cy + half_side

            # Ensure region is within image bounds
            sq_sx, sq_sy = max(0, sq_sx), max(0, sq_sy)
            sq_ex, sq_ey = min(w, sq_ex), min(h, sq_ey)

            if sq_ex - sq_sx < 20 or sq_ey - sq_sy < 20:
                print(f"Box too small for face {face_idx + 1} in {fname}, skipping warp.")
                continue

            # Rotation angle using eyes
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi

            # Create rotation matrix centered at the refined face centroid
            scale = out_size / side
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)

            # Adjust translation to center the face in the output image
            M_rot[0, 2] += (out_size / 2.0 - cx)
            M_rot[1, 2] += (out_size / 2.0 - cy)

            # Warp the image
            warped_face = cv2.warpAffine(img, M_rot, (out_size, out_size), flags=cv2.INTER_LANCZOS4)

            # Transform landmarks
            ones = np.ones((68, 1), dtype=np.float32)
            lmrk_2d = np.hstack([lmrk, ones])
            warped_2d = (lmrk_2d @ M_rot.T)[:, :2]

            # Define source rectangle
            src_rect = [int(sq_sx), int(sq_sy), int(sq_ex), int(sq_ey)]

            # Save output
            base_name, _ = os.path.splitext(fname)
            out_name = f"{base_name}_face{face_idx + 1}_dfl.jpg"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, warped_face, [cv2.IMWRITE_JPEG_QUALITY, 95])

            dflimg = DFLJPG.load(out_path)
            if dflimg is None:
                print(f"Failed to embed metadata in {out_name}.")
                continue

            dflimg.set_face_type(chosen_face_type)
            dflimg.set_landmarks(warped_2d.tolist())
            dflimg.set_source_rect(src_rect)
            dflimg.set_source_landmarks(lmrk.tolist())
            dflimg.set_image_to_face_mat(M_rot.flatten().tolist())
            dflimg.set_source_filename(fname)
            dflimg.save()

            pass2_count += 1

        print(f"Pass 2/2: {i + 1}/{total_count} frames processed.")

    total_elapsed = time.time() - start_time
    total_detected = sum(len(frame) for frame in frame_data)
    print(f"Done! Detected {total_detected} faces across {total_count} images, warped {pass2_count} faces.")
    print(f"Total time: {total_elapsed:.1f}s, Saved to: {output_folder}")

if __name__ == "__main__":
    main()
