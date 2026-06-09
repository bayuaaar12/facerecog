import argparse
import base64
import json
import platform
import re
import time
from pathlib import Path
from urllib import error, request

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "face_ref.xml"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/customers/register-face"
DEFAULT_DETECTION_API_URL = "http://127.0.0.1:8000/api/customers/detect-member"
COLOR_PRIMARY = (238, 112, 35)
COLOR_PRIMARY_SOFT = (255, 241, 229)
COLOR_ACCENT = (75, 167, 88)
COLOR_ACCENT_SOFT = (226, 247, 231)
COLOR_TEXT = (28, 31, 36)
COLOR_MUTED = (116, 124, 138)
COLOR_BORDER = (220, 226, 235)
COLOR_BG = (246, 248, 251)
COLOR_PANEL = (255, 255, 255)
COLOR_SURFACE = (236, 240, 246)
COLOR_BLUE = COLOR_PRIMARY
COLOR_BLUE_SOFT = COLOR_PRIMARY_SOFT

face_ref = cv2.CascadeClassifier(str(CASCADE_PATH))
orb = cv2.ORB_create(nfeatures=700)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
LAST_MEMBER_NOTIFICATION = {"label": None, "sent_at": 0.0}

FACE_IMAGE_SIZE = (200, 200)
LOW_LIGHT_MEAN_LIMIT = 95
LOW_LIGHT_TARGET_MEAN = 125
LOW_LIGHT_BRIGHTNESS_BONUS = 18
ORB_DISTANCE_LIMIT = 60
ORB_RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 18
MIN_MATCH_MARGIN = 8
MIN_MATCH_RATIO = 0.18
LOW_FEATURE_DESCRIPTOR_LIMIT = 90
LOW_FEATURE_MIN_GOOD_MATCHES = 12
LOW_FEATURE_MIN_MATCH_MARGIN = 5
LOW_FEATURE_MIN_MATCH_RATIO = 0.12
MIN_TEMPLATE_SIMILARITY = 62
MIN_TEMPLATE_MARGIN = 3
MIN_TEMPLATE_ORB_SUPPORT = 4
FACE_DETECTION_ANGLES = (0, -20, 20, -35, 35)
FACE_DETECTION_IOU_LIMIT = 0.35
REGISTER_SAMPLE_COUNT = 5
REGISTER_SAMPLE_INTERVAL = 0.35
REGISTER_SAMPLE_TIMEOUT = 6.0
AUTO_REGISTER_CAPTURE_DELAY = 1.5


def default_camera_index():
    if platform.system() == "Darwin":
        return 0
    return 0


def camera_backend_candidates(camera_index):
    system_name = platform.system()

    if system_name == "Darwin":
        return [
            (camera_index, cv2.CAP_AVFOUNDATION),
            (camera_index, cv2.CAP_ANY),
        ]

    if system_name == "Windows":
        return [
            (camera_index, cv2.CAP_DSHOW),
            (camera_index, cv2.CAP_MSMF),
            (camera_index, cv2.CAP_ANY),
        ]

    return [(camera_index, cv2.CAP_ANY)]


def open_camera(camera_index=0):
    for index, backend in camera_backend_candidates(camera_index):
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            return camera
        camera.release()

    extra_hint = ""
    if platform.system() == "Darwin":
        extra_hint = (
            " Di macOS cek System Settings > Privacy & Security > Camera, "
            "pastikan Terminal/Python diberi izin kamera. Jika kamera eksternal, "
            "coba Camera Index 1 atau 2."
        )

    raise RuntimeError(f"Camera index {camera_index} tidak bisa dibuka.{extra_hint}")


def slugify(value):
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "customer"


def face_label_from_path(image_path):
    return re.sub(r"_\d+$", "", image_path.stem)


def load_known_faces():
    known_faces = []

    KNOWN_FACES_DIR.mkdir(exist_ok=True)

    for image_path in sorted(KNOWN_FACES_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = preprocess_face(image)
        keypoints, descriptors = orb.detectAndCompute(image, None)

        known_faces.append(
            {
                "name": face_label_from_path(image_path),
                "sample": image_path.stem,
                "image": image,
                "descriptors": descriptors,
                "keypoint_count": len(keypoints),
            }
        )

    return known_faces


def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = normalize_lighting(gray_frame)
    faces = detect_faces_at_angles(gray_frame)
    return gray_frame, faces


def detect_faces_at_angles(gray_frame):
    detections = []
    height, width = gray_frame.shape[:2]
    center = (width / 2, height / 2)

    for angle in FACE_DETECTION_ANGLES:
        if angle == 0:
            rotated_frame = gray_frame
            rotation_matrix = None
        else:
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(
                gray_frame,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        faces = face_ref.detectMultiScale(
            rotated_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for x, y, w, h in faces:
            if rotation_matrix is None:
                box = (x, y, w, h)
            else:
                box = unrotate_box((x, y, w, h), rotation_matrix, width, height)
            detections.append(box)

    return merge_overlapping_boxes(detections)


def unrotate_box(box, rotation_matrix, width, height):
    x, y, w, h = box
    inverse_matrix = cv2.invertAffineTransform(rotation_matrix)
    corners = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )
    transformed = cv2.transform(np.array([corners]), inverse_matrix)[0]
    x1, y1 = np.floor(transformed.min(axis=0)).astype(int)
    x2, y2 = np.ceil(transformed.max(axis=0)).astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def box_iou(first_box, second_box):
    first_x, first_y, first_w, first_h = first_box
    second_x, second_y, second_w, second_h = second_box
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = first_w * first_h
    second_area = second_w * second_h
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def merge_overlapping_boxes(boxes):
    merged_boxes = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(box_iou(box, existing_box) < FACE_DETECTION_IOU_LIMIT for existing_box in merged_boxes):
            merged_boxes.append(box)
    return np.array(merged_boxes, dtype=np.int32)


def normalize_lighting(gray_image):
    mean_light = float(np.mean(gray_image))
    if mean_light < LOW_LIGHT_MEAN_LIMIT:
        alpha = min(2.0, LOW_LIGHT_TARGET_MEAN / max(mean_light, 1.0))
        gray_image = cv2.convertScaleAbs(
            gray_image,
            alpha=alpha,
            beta=LOW_LIGHT_BRIGHTNESS_BONUS,
        )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def preprocess_face(face_roi):
    resized_face = cv2.resize(face_roi, FACE_IMAGE_SIZE)
    return normalize_lighting(resized_face)


def count_good_matches(source_descriptors, known_descriptors):
    if source_descriptors is None:
        return 0
    if known_descriptors is None or len(known_descriptors) < 2:
        return 0

    knn_matches = matcher.knnMatch(source_descriptors, known_descriptors, k=2)
    good_matches = [
        first
        for first, second in knn_matches
        if first.distance < ORB_DISTANCE_LIMIT
        and first.distance < ORB_RATIO_TEST * second.distance
    ]
    return len(good_matches)


def template_similarity(source_face, known_face):
    correlation = cv2.matchTemplate(source_face, known_face, cv2.TM_CCOEFF_NORMED)[0][0]
    abs_similarity = 1.0 - (np.mean(cv2.absdiff(source_face, known_face)) / 255.0)
    return max(0.0, ((correlation + 1.0) / 2.0) * 100.0, abs_similarity * 100.0)


def recognize_face(face_roi):
    normalized_face = preprocess_face(face_roi)
    keypoints, descriptors = orb.detectAndCompute(normalized_face, None)

    if not KNOWN_FACES:
        return "Unknown", 0

    best_name = "Unknown"
    best_combined_score = 0
    best_orb_score = 0
    best_template_score = 0
    second_combined_score = 0
    second_template_score = 0
    descriptor_count = max(len(keypoints), 1)
    scores_by_name = {}

    for known_face in KNOWN_FACES:
        face_name = known_face["name"]
        orb_score = count_good_matches(descriptors, known_face["descriptors"])
        template_score = template_similarity(normalized_face, known_face["image"])
        combined_score = orb_score + int(max(0, template_score - 50) * 0.35)
        current_score = scores_by_name.get(
            face_name,
            {
                "combined": 0,
                "orb": 0,
                "template": 0,
            },
        )
        if combined_score > current_score["combined"]:
            scores_by_name[face_name] = {
                "combined": combined_score,
                "orb": orb_score,
                "template": template_score,
            }

    for face_name, score in scores_by_name.items():
        if score["combined"] > best_combined_score:
            second_combined_score = best_combined_score
            second_template_score = best_template_score
            best_combined_score = score["combined"]
            best_orb_score = score["orb"]
            best_template_score = score["template"]
            best_name = face_name
        elif score["combined"] > second_combined_score:
            second_combined_score = score["combined"]
            second_template_score = score["template"]

    match_ratio = best_orb_score / descriptor_count
    required_good_matches = MIN_GOOD_MATCHES
    required_match_margin = MIN_MATCH_MARGIN
    required_match_ratio = MIN_MATCH_RATIO
    if descriptor_count < LOW_FEATURE_DESCRIPTOR_LIMIT:
        required_good_matches = LOW_FEATURE_MIN_GOOD_MATCHES
        required_match_margin = LOW_FEATURE_MIN_MATCH_MARGIN
        required_match_ratio = LOW_FEATURE_MIN_MATCH_RATIO

    orb_is_confident = (
        best_orb_score >= required_good_matches
        and best_combined_score - second_combined_score >= required_match_margin
        and match_ratio >= required_match_ratio
    )
    template_is_confident = (
        best_template_score >= MIN_TEMPLATE_SIMILARITY
        and best_template_score - second_template_score >= MIN_TEMPLATE_MARGIN
        and best_orb_score >= MIN_TEMPLATE_ORB_SUPPORT
    )

    if not orb_is_confident and not template_is_confident:
        return "Unknown", best_combined_score

    return best_name, best_combined_score


KNOWN_FACES = load_known_faces()


def known_person_count():
    return len({known_face["name"] for known_face in KNOWN_FACES})


def save_face_locally(face_image, face_label):
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    target_path = KNOWN_FACES_DIR / f"{face_label}.jpg"
    cv2.imwrite(str(target_path), face_image)
    return target_path


def next_face_sample_path(face_label):
    existing_numbers = []
    for image_path in list_known_face_files():
        if face_label_from_path(image_path) != face_label:
            continue

        match = re.search(r"_(\d+)$", image_path.stem)
        existing_numbers.append(int(match.group(1)) if match else 1)

    next_number = max(existing_numbers, default=0) + 1
    return KNOWN_FACES_DIR / f"{face_label}_{next_number}.jpg"


def save_face_sample(face_image, face_label):
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    target_path = next_face_sample_path(face_label)
    cv2.imwrite(str(target_path), face_image)
    return target_path


def collect_face_samples(camera, initial_gray_frame, initial_faces, face_label):
    samples = []
    started_at = time.time()
    last_capture_at = 0.0

    selected_face = find_largest_face(initial_faces)
    if selected_face is not None and initial_gray_frame is not None:
        x, y, w, h = selected_face
        samples.append(preprocess_face(initial_gray_frame[y : y + h, x : x + w]))
        last_capture_at = time.time()

    while len(samples) < REGISTER_SAMPLE_COUNT:
        if time.time() - started_at > REGISTER_SAMPLE_TIMEOUT:
            break

        success, frame = camera.read()
        if not success:
            break

        if time.time() - last_capture_at < REGISTER_SAMPLE_INTERVAL:
            cv2.waitKey(1)
            continue

        frame = cv2.flip(frame, 1)
        gray_frame, faces = face_detection(frame)
        selected_face = find_largest_face(faces)
        if selected_face is None:
            continue

        x, y, w, h = selected_face
        samples.append(preprocess_face(gray_frame[y : y + h, x : x + w]))
        last_capture_at = time.time()

    saved_paths = [save_face_sample(sample, face_label) for sample in samples]
    return samples, saved_paths


def list_known_face_files():
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png"}
    return [
        image_path
        for image_path in sorted(KNOWN_FACES_DIR.iterdir())
        if image_path.suffix.lower() in image_extensions
    ]


def list_known_face_entries():
    entries_by_label = {}
    for image_path in list_known_face_files():
        label = face_label_from_path(image_path)
        entries_by_label.setdefault(label, []).append(image_path)

    return [
        {"label": label, "files": files}
        for label, files in sorted(entries_by_label.items())
    ]


def display_face_label(face_label):
    return face_label.replace("_", " ").title()


def delete_known_face(face_entry):
    global KNOWN_FACES

    for face_path in face_entry["files"]:
        try:
            face_path.unlink()
        except FileNotFoundError:
            pass

    KNOWN_FACES = load_known_faces()


def image_to_base64(face_image):
    success, buffer = cv2.imencode(".jpg", face_image)
    if not success:
        raise RuntimeError("Gagal mengubah gambar wajah ke JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def post_json(url, payload, timeout=10):
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Laravel API error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Tidak bisa terhubung ke Laravel API: {exc.reason}") from exc


def notify_member_detected(face_label, score, api_url=DEFAULT_DETECTION_API_URL):
    global LAST_MEMBER_NOTIFICATION

    now = time.time()
    same_label = LAST_MEMBER_NOTIFICATION["label"] == face_label
    sent_recently = now - LAST_MEMBER_NOTIFICATION["sent_at"] < 15

    if same_label and sent_recently:
        return

    LAST_MEMBER_NOTIFICATION = {"label": face_label, "sent_at": now}

    try:
        post_json(
            api_url,
            {
                "face_label": face_label,
                "score": int(score),
            },
            timeout=1,
        )
    except RuntimeError as exc:
        print(f"Gagal kirim notif member ke Laravel: {exc}")


def find_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda item: item[2] * item[3])


def register_customer(name, phone, discount_percent, api_url, camera_index=0):
    camera = open_camera(camera_index)
    face_label = slugify(name)
    window_name = "Register Customer Face"
    response_payload = None
    frame_count = 0
    gray_frame = None
    faces = []
    face_seen_since = None
    action = {"value": None}
    buttons = [
        {
            "label": "Capture",
            "value": "capture",
            "rect": (350, 505, 485, 555),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Balik",
            "value": "back",
            "rect": (500, 505, 620, 555),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            action["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 3 == 1:
            gray_frame, faces = face_detection(frame)
            if len(faces):
                if face_seen_since is None:
                    face_seen_since = time.time()
            else:
                face_seen_since = None

        frame = cv2.copyMakeBorder(
            frame,
            0,
            90,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=COLOR_BG,
        )
        draw_round_rect(frame, (14, 492, 626, 568), COLOR_PANEL, 18, -1, COLOR_BORDER)
        chip_color = COLOR_ACCENT if len(faces) else COLOR_MUTED
        chip_soft = COLOR_ACCENT_SOFT if len(faces) else COLOR_SURFACE
        draw_status_chip(frame, f"{len(faces)} wajah terdeteksi", (22, 524, 208, 560), chip_color, chip_soft)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_PRIMARY, 2)

        auto_status = "Auto save: cari wajah"
        if face_seen_since is not None:
            remaining = max(0.0, AUTO_REGISTER_CAPTURE_DELAY - (time.time() - face_seen_since))
            auto_status = f"Auto save {remaining:.1f}s"

        cv2.putText(
            frame,
            "Register Wajah",
            (22, 516),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            auto_status,
            (225, 526),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            COLOR_ACCENT if face_seen_since is not None else COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{name} | Diskon {discount_percent}%",
            (225, 548),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        for button in buttons:
            draw_menu_button(frame, button)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or action["value"] == "back":
            break

        auto_capture_ready = (
            face_seen_since is not None
            and time.time() - face_seen_since >= AUTO_REGISTER_CAPTURE_DELAY
        )

        if key == ord("c") or action["value"] == "capture" or auto_capture_ready:
            action["value"] = None
            selected_face = find_largest_face(faces)
            if selected_face is None or gray_frame is None:
                print("Wajah belum terdeteksi. Coba hadapkan wajah ke kamera.")
                continue

            if auto_capture_ready:
                print("Wajah terdeteksi stabil. Auto save ke lokal...")

            face_samples, saved_paths = collect_face_samples(camera, gray_frame, faces, face_label)
            if not face_samples:
                print("Gagal mengambil sampel wajah. Coba ulangi dengan posisi wajah lebih jelas.")
                continue

            global KNOWN_FACES
            KNOWN_FACES = load_known_faces()
            face_roi = face_samples[0]

            try:
                response_payload = post_json(
                    api_url,
                    {
                        "name": name,
                        "phone": phone,
                        "discount_percent": discount_percent,
                        "face_label": face_label,
                        "face_image_base64": image_to_base64(face_roi),
                    },
                    timeout=3,
                )
                print("Customer berhasil dikirim ke Laravel.")
                print(f"{len(saved_paths)} sampel wajah tersimpan lokal.")
                print(json.dumps(response_payload, indent=2))
            except RuntimeError as exc:
                print(f"{len(saved_paths)} sampel wajah tersimpan lokal, tapi gagal kirim ke Laravel.")
                print(exc)
            break

    close_window(camera)
    return response_payload


def run_recognition(camera_index=0):
    camera = open_camera(camera_index)
    window_name = "Pingkal Face Recognition"
    frame_count = 0
    detections = []
    action = {"value": None}
    buttons = [
        {
            "label": "Balik",
            "value": "back",
            "rect": (510, 505, 620, 555),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            action["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 4 == 1:
            gray_frame, faces = face_detection(frame)
            detections = []

            for x, y, w, h in faces:
                face_roi = gray_frame[y : y + h, x : x + w]
                label, score = recognize_face(face_roi)
                detections.append((x, y, w, h, label, score))

        for x, y, w, h, label, score in detections:
            color = COLOR_BLUE if label != "Unknown" else (90, 90, 96)
            text = f"{label} ({score})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            if label != "Unknown":
                notify_member_detected(label, score)

        frame = cv2.copyMakeBorder(
            frame,
            0,
            90,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=COLOR_BG,
        )

        if not KNOWN_FACES:
            draw_round_rect(frame, (14, 492, 626, 568), COLOR_PANEL, 18, -1, COLOR_BORDER)
            draw_status_chip(frame, "Belum ada data", (22, 524, 170, 560), COLOR_MUTED, COLOR_SURFACE)
            cv2.putText(
                frame,
                "Folder known_faces masih kosong",
                (22, 516),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_MUTED,
                1,
                cv2.LINE_AA,
            )
        else:
            known_label = f"{known_person_count()} orang / {len(KNOWN_FACES)} sampel"
            recognized_count = sum(1 for item in detections if item[4] != "Unknown")
            status_text = "Member dikenali" if recognized_count else "Scanning aktif"
            status_color = COLOR_ACCENT if recognized_count else COLOR_PRIMARY
            status_soft = COLOR_ACCENT_SOFT if recognized_count else COLOR_PRIMARY_SOFT
            draw_round_rect(frame, (14, 492, 626, 568), COLOR_PANEL, 18, -1, COLOR_BORDER)
            draw_status_chip(frame, status_text, (22, 524, 182, 560), status_color, status_soft)
            cv2.putText(
                frame,
                "Face Recognition",
                (22, 516),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                known_label,
                (205, 548),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_MUTED,
                1,
                cv2.LINE_AA,
            )

        for button in buttons:
            draw_menu_button(frame, button)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or action["value"] == "back":
            break

    close_window(camera)


def close_window(camera):
    camera.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    default_index = default_camera_index()
    parser.add_argument(
        "--mode",
        choices=["recognize", "register", "manage"],
        help="recognize untuk deteksi wajah, register untuk simpan customer, manage untuk list/hapus data",
    )
    parser.add_argument("--name", help="Nama customer saat mode register")
    parser.add_argument("--phone", default="", help="Nomor telepon customer")
    parser.add_argument("--discount", type=int, default=0, help="Diskon member persen")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL API Laravel")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=default_index,
        help=f"Index kamera OpenCV, default di perangkat ini: {default_index}",
    )
    return parser.parse_args()


def prompt_int(label, default_value=0):
    raw_value = input(f"{label} [{default_value}]: ").strip()
    if not raw_value:
        return default_value

    try:
        return int(raw_value)
    except ValueError:
        print(f"{label} harus angka. Dipakai default {default_value}.")
        return default_value


def draw_centered_text(canvas, text, center, font_scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(center[0] - text_size[0] / 2)
    y = int(center[1] + text_size[1] / 2)
    cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_round_rect(canvas, rect, color, radius=14, thickness=-1, border_color=None):
    x1, y1, x2, y2 = rect
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, -1)
        if border_color:
            draw_round_rect(canvas, rect, border_color, radius, 1)
        return

    cv2.line(canvas, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_header(canvas, title, subtitle, width):
    draw_round_rect(canvas, (18, 18, width - 18, 112), COLOR_PANEL, 18, -1, COLOR_BORDER)
    draw_round_rect(canvas, (36, 42, 76, 82), COLOR_PRIMARY_SOFT, 12, -1)
    cv2.circle(canvas, (56, 62), 11, COLOR_PRIMARY, -1, cv2.LINE_AA)
    cv2.circle(canvas, (56, 62), 4, COLOR_PANEL, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (94, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (94, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1, cv2.LINE_AA)


def draw_status_chip(canvas, text, rect, color, soft_color):
    draw_round_rect(canvas, rect, soft_color, 12, -1)
    x1, y1, _x2, _y2 = rect
    cv2.circle(canvas, (x1 + 18, y1 + 18), 5, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x1 + 32, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def draw_menu_button(canvas, button):
    x1, y1, x2, y2 = button["rect"]
    color = button.get("color", COLOR_PANEL)
    border = button.get("border", COLOR_BORDER)
    text_color = button.get("text_color", COLOR_PRIMARY)

    draw_round_rect(canvas, (x1, y1, x2, y2), color, 15, -1, border)
    draw_centered_text(
        canvas,
        button["label"],
        ((x1 + x2) // 2, (y1 + y2) // 2),
        button.get("font_scale", 0.58),
        text_color,
        button.get("thickness", 1),
    )


def button_at_position(buttons, x, y):
    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button["value"]
    return None


def field_at_position(fields, x, y):
    for index, field in enumerate(fields):
        x1, y1, x2, y2 = field["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return index
    return None


def draw_input_field(canvas, field, value, active=False):
    x1, y1, x2, y2 = field["rect"]
    border = COLOR_PRIMARY if active else COLOR_BORDER

    cv2.putText(
        canvas,
        field["label"],
        (x1, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_MUTED,
        1,
        cv2.LINE_AA,
    )
    draw_round_rect(canvas, (x1, y1, x2, y2), COLOR_PANEL, 13, -1, border)

    display_value = value if value else field["placeholder"]
    text_color = COLOR_TEXT if value else (170, 170, 176)
    if active:
        display_value = f"{display_value}|"

    cv2.putText(
        canvas,
        display_value[:34],
        (x1 + 14, y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        text_color,
        1,
        cv2.LINE_AA,
    )


def show_register_form(default_discount=0):
    window_name = "Data Register"
    selected = {"value": None}
    active_field = {"index": 0}
    values = {
        "name": "",
        "phone": "",
        "discount": str(default_discount),
    }
    status = {"text": "Isi nama, lalu klik Mulai Register."}
    fields = [
        {
            "key": "name",
            "label": "Nama customer",
            "placeholder": "Contoh: Budi",
            "rect": (120, 205, 600, 258),
        },
        {
            "key": "phone",
            "label": "No. telepon",
            "placeholder": "Opsional",
            "rect": (120, 285, 600, 338),
        },
        {
            "key": "discount",
            "label": "Diskon (%)",
            "placeholder": "0",
            "rect": (120, 365, 600, 418),
        },
    ]
    buttons = [
        {
            "label": "Mulai register",
            "value": "start",
            "rect": (120, 455, 420, 510),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Balik",
            "value": "back",
            "rect": (440, 455, 600, 510),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        field_index = field_at_position(fields, x, y)
        if field_index is not None:
            active_field["index"] = field_index
            return

        selected["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while selected["value"] is None:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            selected["value"] = "back"
            break

        canvas = np.full((560, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, "Register Wajah", "Simpan data member dan wajah customer.", 720)
        draw_round_rect(canvas, (92, 124, 628, 530), COLOR_PANEL, 18, -1, COLOR_BORDER)
        draw_status_chip(canvas, "Form customer", (120, 132, 270, 168), COLOR_ACCENT, COLOR_ACCENT_SOFT)
        cv2.putText(
            canvas,
            status["text"],
            (120, 176),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        for index, field in enumerate(fields):
            draw_input_field(canvas, field, values[field["key"]], active_field["index"] == index)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            selected["value"] = "back"
            break

        if key in (9, 13):
            active_field["index"] = (active_field["index"] + 1) % len(fields)
            continue

        field = fields[active_field["index"]]
        field_key = field["key"]

        if key in (8, 127):
            values[field_key] = values[field_key][:-1]
            continue

        if 32 <= key <= 126:
            char = chr(key)
            if field_key == "discount" and not char.isdigit():
                continue
            values[field_key] = (values[field_key] + char)[:40]

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    if selected["value"] != "start":
        return None

    name = values["name"].strip()
    if not name:
        print("Nama customer wajib diisi.")
        return None

    try:
        discount = int(values["discount"].strip() or "0")
    except ValueError:
        discount = default_discount

    return {
        "name": name,
        "phone": values["phone"].strip(),
        "discount": discount,
    }


def row_at_position(rows, x, y):
    for index, rect in enumerate(rows):
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            return index
    return None


def show_member_data_list():
    window_name = "Data Wajah Member"
    action = {"value": None}
    selected = {"index": 0}
    scroll = {"offset": 0}
    confirm_delete = {"value": False}
    status = {"text": "Pilih data untuk melihat atau hapus."}
    visible_rows = 7
    buttons = [
        {
            "label": "Hapus",
            "value": "delete",
            "rect": (120, 555, 280, 610),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Balik",
            "value": "back",
            "rect": (440, 555, 600, 610),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]
    row_rects = []

    def on_mouse(event, x, y, _flags, _params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        row_index = row_at_position(row_rects, x, y)
        if row_index is not None:
            selected["index"] = scroll["offset"] + row_index
            confirm_delete["value"] = False
            return

        if confirm_delete["value"] and 300 <= x <= 425 and 555 <= y <= 610:
            action["value"] = "confirm_delete"
            return

        action["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while action["value"] != "back":
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            action["value"] = "back"
            break

        face_entries = list_known_face_entries()
        if selected["index"] >= len(face_entries):
            selected["index"] = max(0, len(face_entries) - 1)
        if selected["index"] < scroll["offset"]:
            scroll["offset"] = selected["index"]
        if selected["index"] >= scroll["offset"] + visible_rows:
            scroll["offset"] = selected["index"] - visible_rows + 1
        scroll["offset"] = max(0, min(scroll["offset"], max(0, len(face_entries) - visible_rows)))

        canvas = np.full((640, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, "Data Wajah", "List member yang tersimpan lokal.", 720)
        draw_round_rect(canvas, (92, 124, 628, 625), COLOR_PANEL, 18, -1, COLOR_BORDER)
        draw_status_chip(canvas, f"{len(face_entries)} orang", (120, 132, 250, 168), COLOR_ACCENT, COLOR_ACCENT_SOFT)
        cv2.putText(
            canvas,
            status["text"],
            (120, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        row_rects = []
        if not face_entries:
            cv2.putText(
                canvas,
                "Belum ada wajah tersimpan.",
                (120, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                COLOR_MUTED,
                1,
                cv2.LINE_AA,
            )
        else:
            visible_entries = face_entries[scroll["offset"] : scroll["offset"] + visible_rows]
            for index, face_entry in enumerate(visible_entries):
                row_y = 220 + index * 45
                rect = (120, row_y, 600, row_y + 36)
                row_rects.append(rect)
                absolute_index = scroll["offset"] + index
                is_selected = absolute_index == selected["index"]
                row_color = COLOR_PRIMARY_SOFT if is_selected else COLOR_PANEL
                border_color = COLOR_PRIMARY if is_selected else COLOR_BORDER
                text_color = COLOR_PRIMARY if is_selected else COLOR_TEXT
                draw_round_rect(canvas, rect, row_color, 10, -1, border_color)
                cv2.putText(
                    canvas,
                    f"{absolute_index + 1}. {display_face_label(face_entry['label'])}",
                    (138, row_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"{len(face_entry['files'])} sampel",
                    (435, row_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    COLOR_MUTED,
                    1,
                    cv2.LINE_AA,
                )

        if confirm_delete["value"] and face_entries:
            selected_name = display_face_label(face_entries[selected["index"]]["label"])
            draw_round_rect(canvas, (300, 555, 425, 610), COLOR_ACCENT, 15, -1, COLOR_ACCENT)
            draw_centered_text(canvas, "Yakin", (362, 582), 0.58, (255, 255, 255), 1)
            cv2.putText(
                canvas,
                f"Hapus {selected_name[:18]}?",
                (120, 535),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_PRIMARY,
                1,
                cv2.LINE_AA,
            )
        else:
            draw_round_rect(canvas, (300, 555, 425, 610), COLOR_SURFACE, 15, -1, COLOR_BORDER)
            draw_centered_text(canvas, "Pilih", (362, 582), 0.58, COLOR_MUTED, 1)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            action["value"] = "back"
            break
        if key in (82, ord("w")) and face_entries:
            selected["index"] = max(0, selected["index"] - 1)
            confirm_delete["value"] = False
            continue
        if key in (84, ord("s")) and face_entries:
            selected["index"] = min(len(face_entries) - 1, selected["index"] + 1)
            confirm_delete["value"] = False
            continue
        if key in (ord("d"), 127, 8):
            action["value"] = "delete"
        if key in (13, ord("y")) and confirm_delete["value"]:
            action["value"] = "confirm_delete"

        if action["value"] == "delete":
            action["value"] = None
            if not face_entries:
                status["text"] = "Tidak ada data yang bisa dihapus."
                continue
            confirm_delete["value"] = not confirm_delete["value"]
            status["text"] = "Klik Yakin atau tekan Enter untuk hapus." if confirm_delete["value"] else "Hapus dibatalkan."
            continue

        if action["value"] == "confirm_delete":
            action["value"] = None
            if confirm_delete["value"] and face_entries:
                deleted_name = display_face_label(face_entries[selected["index"]]["label"])
                delete_known_face(face_entries[selected["index"]])
                confirm_delete["value"] = False
                status["text"] = f"{deleted_name} sudah dihapus."
            continue

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def show_visual_menu():
    window_name = "Pingkal Face Menu"
    selected = {"value": None}
    buttons = [
        {
            "label": "Recognize",
            "value": "1",
            "rect": (120, 150, 600, 205),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
        {
            "label": "Register Wajah",
            "value": "2",
            "rect": (120, 225, 600, 280),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "List Data",
            "value": "3",
            "rect": (120, 300, 600, 355),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
        {
            "label": "Exit",
            "value": "0",
            "rect": (120, 375, 600, 430),
            "border": COLOR_BORDER,
            "text_color": COLOR_MUTED,
        },
    ]

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["value"] = button_at_position(buttons, x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while selected["value"] is None:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            selected["value"] = "0"
            break

        canvas = np.full((515, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, "Pingkal Face", "Kasir member recognition", 720)
        draw_round_rect(canvas, (92, 124, 628, 450), COLOR_PANEL, 18, -1, COLOR_BORDER)
        draw_status_chip(canvas, f"{known_person_count()} orang", (120, 466, 250, 502), COLOR_ACCENT, COLOR_ACCENT_SOFT)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.putText(
            canvas,
            "Kamera default: index 0",
            (452, 489),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("1"), ord("2"), ord("3"), ord("0")):
            selected["value"] = chr(key)
        elif key in (ord("q"), 27):
            selected["value"] = "0"

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    return selected["value"]


def run_terminal_menu(args):
    while True:
        choice = show_visual_menu()

        if choice == "1":
            run_recognition(args.camera_index)
            continue

        if choice == "2":
            form_data = show_register_form(args.discount)
            if form_data is None:
                continue

            register_customer(
                form_data["name"],
                form_data["phone"],
                form_data["discount"],
                args.api_url,
                args.camera_index,
            )
            continue

        if choice == "3":
            show_member_data_list()
            continue

        if choice == "0":
            print("Keluar.")
            return

        print("Menu tidak valid.")


def main():
    args = parse_args()

    if args.mode is None:
        run_terminal_menu(args)
        return

    if args.mode == "register":
        if not args.name:
            raise RuntimeError("Mode register butuh --name.")
        register_customer(args.name, args.phone, args.discount, args.api_url, args.camera_index)
        return

    if args.mode == "manage":
        show_member_data_list()
        return

    run_recognition(args.camera_index)


if __name__ == "__main__":
    main()
