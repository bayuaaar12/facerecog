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
orb = cv2.ORB_create(nfeatures=400)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
LAST_MEMBER_NOTIFICATION = {"label": None, "sent_at": 0.0}


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


def load_known_faces():
    known_faces = []

    KNOWN_FACES_DIR.mkdir(exist_ok=True)

    for image_path in sorted(KNOWN_FACES_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 10:
            continue

        known_faces.append({"name": image_path.stem, "descriptors": descriptors})

    return known_faces


KNOWN_FACES = load_known_faces()


def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    return gray_frame, faces


def recognize_face(face_roi):
    resized_face = cv2.resize(face_roi, (200, 200))
    _, descriptors = orb.detectAndCompute(resized_face, None)

    if descriptors is None or not KNOWN_FACES:
        return "Unknown", 0

    best_name = "Unknown"
    best_score = 0

    for known_face in KNOWN_FACES:
        matches = matcher.match(descriptors, known_face["descriptors"])
        good_matches = [match for match in matches if match.distance < 55]
        score = len(good_matches)

        if score > best_score:
            best_score = score
            best_name = known_face["name"]

    if best_score < 20:
        return "Unknown", best_score

    return best_name, best_score


def save_face_locally(face_image, face_label):
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    target_path = KNOWN_FACES_DIR / f"{face_label}.jpg"
    cv2.imwrite(str(target_path), face_image)
    return target_path


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

        if key == ord("c") or action["value"] == "capture":
            action["value"] = None
            selected_face = find_largest_face(faces)
            if selected_face is None or gray_frame is None:
                print("Wajah belum terdeteksi. Coba hadapkan wajah ke kamera.")
                continue

            x, y, w, h = selected_face
            face_roi = gray_frame[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            save_face_locally(face_roi, face_label)
            global KNOWN_FACES
            KNOWN_FACES = load_known_faces()

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
                print(json.dumps(response_payload, indent=2))
            except RuntimeError as exc:
                print("Wajah tersimpan lokal, tapi gagal kirim ke Laravel.")
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
            known_label = f"{len(KNOWN_FACES)} wajah terdaftar"
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
        choices=["recognize", "register"],
        help="recognize untuk deteksi wajah, register untuk simpan customer ke Laravel",
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


def show_visual_menu():
    window_name = "Pingkal Face Menu"
    selected = {"value": None}
    buttons = [
        {
            "label": "Recognize",
            "value": "1",
            "rect": (120, 170, 600, 230),
            "border": COLOR_BORDER,
            "text_color": COLOR_PRIMARY,
        },
        {
            "label": "Register Wajah",
            "value": "2",
            "rect": (120, 250, 600, 310),
            "color": COLOR_PRIMARY,
            "border": COLOR_PRIMARY,
            "text_color": (255, 255, 255),
        },
        {
            "label": "Exit",
            "value": "0",
            "rect": (120, 330, 600, 390),
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

        canvas = np.full((475, 720, 3), COLOR_BG, dtype=np.uint8)
        draw_header(canvas, "Pingkal Face", "Kasir member recognition", 720)
        draw_round_rect(canvas, (92, 138, 628, 414), COLOR_PANEL, 18, -1, COLOR_BORDER)
        draw_status_chip(canvas, f"{len(KNOWN_FACES)} wajah terdaftar", (120, 426, 310, 462), COLOR_ACCENT, COLOR_ACCENT_SOFT)

        for button in buttons:
            draw_menu_button(canvas, button)

        cv2.putText(
            canvas,
            "Kamera default: index 0",
            (452, 449),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("1"), ord("2"), ord("0")):
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

    run_recognition(args.camera_index)


if __name__ == "__main__":
    main()
