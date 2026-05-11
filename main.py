import argparse
import base64
import json
import platform
import re
import tkinter as tk
import time
from pathlib import Path
from tkinter import messagebox, ttk
from urllib import error, request

import cv2


BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "face_ref.xml"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/customers/register-face"
DEFAULT_DETECTION_API_URL = "http://127.0.0.1:8000/api/customers/detect-member"

face_ref = cv2.CascadeClassifier(str(CASCADE_PATH))
orb = cv2.ORB_create(nfeatures=700)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
LAST_MEMBER_NOTIFICATION = {"label": None, "sent_at": 0.0}


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


def post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=10) as response:
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

    try:
        post_json(
            api_url,
            {
                "face_label": face_label,
                "score": int(score),
            },
        )
        LAST_MEMBER_NOTIFICATION = {"label": face_label, "sent_at": now}
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

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray_frame, faces = face_detection(frame)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 3)

        cv2.putText(
            frame,
            "Tekan C untuk capture, Q untuk keluar",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Customer: {name} | Diskon: {discount_percent}%",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("c"):
            selected_face = find_largest_face(faces)
            if selected_face is None:
                print("Wajah belum terdeteksi. Coba hadapkan wajah ke kamera.")
                continue

            x, y, w, h = selected_face
            face_roi = gray_frame[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            save_face_locally(face_roi, face_label)

            response_payload = post_json(
                api_url,
                {
                    "name": name,
                    "phone": phone,
                    "discount_percent": discount_percent,
                    "face_image_base64": image_to_base64(face_roi),
                },
            )
            print("Customer berhasil dikirim ke Laravel.")
            print(json.dumps(response_payload, indent=2))
            break

    close_window(camera)
    return response_payload


def run_recognition(camera_index=0):
    camera = open_camera(camera_index)
    window_name = "Pingkal Face Recognition"

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray_frame, faces = face_detection(frame)

        for x, y, w, h in faces:
            face_roi = gray_frame[y : y + h, x : x + w]
            label, score = recognize_face(face_roi)
            color = (0, 200, 0) if label != "Unknown" else (0, 0, 255)
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

        if not KNOWN_FACES:
            cv2.putText(
                frame,
                "Folder known_faces masih kosong",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    close_window(camera)


def close_window(camera):
    camera.release()
    cv2.destroyAllWindows()


def frame_to_tk_image(frame):
    height, width = frame.shape[:2]
    if width > 640:
        scale = 640 / width
        frame = cv2.resize(frame, (640, int(height * scale)))

    success, buffer = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("Tidak bisa render frame kamera.")
    image_data = base64.b64encode(buffer.tobytes()).decode("ascii")
    return tk.PhotoImage(data=image_data, format="png")


def read_camera_frame(camera, attempts=8):
    for _ in range(attempts):
        success, frame = camera.read()
        if success and frame is not None:
            return frame
        time.sleep(0.04)
    return None


def launch_register_camera(parent, name, phone, discount_percent, api_url, camera_index, status_var):
    try:
        camera = open_camera(camera_index)
    except Exception as exc:
        status_var.set("Register gagal.")
        messagebox.showerror("Register gagal", str(exc))
        return

    face_label = slugify(name)
    window = tk.Toplevel(parent)
    window.title("Register Customer Face")
    window.configure(bg="#f5f7fb")
    window.resizable(False, False)

    header = ttk.Frame(window, padding=(18, 16, 18, 8), style="App.TFrame")
    header.pack(fill="x")
    ttk.Label(header, text="Register Face", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header,
        text=f"{name} | Discount {discount_percent}% | Camera {camera_index}",
        style="Muted.TLabel",
    ).pack(anchor="w", pady=(3, 0))

    preview_frame = ttk.Frame(window, padding=10, style="Surface.TFrame")
    preview_frame.pack(padx=18, pady=(6, 10))
    preview_label = ttk.Label(preview_frame, style="Preview.TLabel")
    preview_label.pack()

    camera_status = tk.StringVar(value="Arahkan wajah ke kamera, lalu klik Capture.")
    ttk.Label(window, textvariable=camera_status, style="Status.TLabel").pack(anchor="w", padx=18)

    state = {"after_id": None, "gray": None, "faces": [], "closed": False}

    def close(status_message="Register dibatalkan."):
        if state["closed"]:
            return
        state["closed"] = True
        if state["after_id"] is not None:
            window.after_cancel(state["after_id"])
        close_window(camera)
        window.destroy()
        if status_message:
            status_var.set(status_message)

    def update_preview():
        if state["closed"]:
            return

        try:
            frame = read_camera_frame(camera, attempts=1)
            if frame is None:
                camera_status.set("Frame kamera belum terbaca. Coba Camera Index lain.")
                state["after_id"] = window.after(120, update_preview)
                return

            frame = cv2.flip(frame, 1)
            gray_frame, faces = face_detection(frame)
            state["gray"] = gray_frame
            state["faces"] = faces

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 3)

            cv2.putText(
                frame,
                f"Customer: {name} | Diskon: {discount_percent}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            image = frame_to_tk_image(frame)
        except Exception as exc:
            camera_status.set("Frame kamera tidak terbaca.")
            status_var.set("Preview kamera gagal.")
            messagebox.showerror("Preview kamera gagal", str(exc))
            close("Preview kamera gagal.")
            return

        preview_label.configure(image=image)
        preview_label.image = image
        camera_status.set(f"Wajah terdeteksi: {len(faces)}")
        state["after_id"] = window.after(15, update_preview)

    def capture():
        selected_face = find_largest_face(state["faces"])
        gray_frame = state["gray"]

        if selected_face is None or gray_frame is None:
            camera_status.set("Wajah belum terdeteksi. Coba hadapkan wajah ke kamera.")
            return

        x, y, w, h = selected_face
        face_roi = gray_frame[y : y + h, x : x + w]
        face_roi = cv2.resize(face_roi, (200, 200))

        try:
            save_face_locally(face_roi, face_label)
            post_json(
                api_url,
                {
                    "name": name,
                    "phone": phone,
                    "discount_percent": discount_percent,
                    "face_image_base64": image_to_base64(face_roi),
                },
            )
        except Exception as exc:
            status_var.set("Register gagal.")
            messagebox.showerror("Register gagal", str(exc))
            return

        status_var.set("Customer berhasil disimpan.")
        messagebox.showinfo("Berhasil", "Wajah customer berhasil disimpan ke Laravel.")
        close("Customer berhasil disimpan.")

    button_frame = ttk.Frame(window, padding=(18, 12, 18, 18), style="App.TFrame")
    button_frame.pack(fill="x")
    ttk.Button(button_frame, text="Capture", command=capture, style="Accent.TButton").pack(side="left")
    ttk.Button(button_frame, text="Batal", command=close).pack(side="right")

    window.protocol("WM_DELETE_WINDOW", close)
    window.lift()
    window.focus_force()
    camera_status.set("Menyiapkan kamera...")
    state["after_id"] = window.after(120, update_preview)


def launch_recognition_camera(parent, camera_index, status_var):
    try:
        camera = open_camera(camera_index)
    except Exception as exc:
        status_var.set("Recognize gagal.")
        messagebox.showerror("Recognize gagal", str(exc))
        return

    window = tk.Toplevel(parent)
    window.title("Pingkal Face Recognition")
    window.configure(bg="#f5f7fb")
    window.resizable(False, False)

    header = ttk.Frame(window, padding=(18, 16, 18, 8), style="App.TFrame")
    header.pack(fill="x")
    ttk.Label(header, text="Live Recognition", style="Title.TLabel").pack(anchor="w")
    ttk.Label(header, text=f"Camera {camera_index}", style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

    preview_frame = ttk.Frame(window, padding=10, style="Surface.TFrame")
    preview_frame.pack(padx=18, pady=(6, 10))
    preview_label = ttk.Label(preview_frame, style="Preview.TLabel")
    preview_label.pack()

    camera_status = tk.StringVar(value="Mode recognize berjalan.")
    ttk.Label(window, textvariable=camera_status, style="Status.TLabel").pack(anchor="w", padx=18)

    state = {"after_id": None, "closed": False, "frame_count": 0, "detections": []}

    def close():
        if state["closed"]:
            return
        state["closed"] = True
        if state["after_id"] is not None:
            window.after_cancel(state["after_id"])
        close_window(camera)
        window.destroy()
        status_var.set("Mode recognize selesai.")

    def update_preview():
        if state["closed"]:
            return

        try:
            frame = read_camera_frame(camera, attempts=1)
            if frame is None:
                camera_status.set("Frame kamera belum terbaca. Coba Camera Index lain.")
                state["after_id"] = window.after(120, update_preview)
                return

            frame = cv2.flip(frame, 1)
            state["frame_count"] += 1

            if state["frame_count"] % 3 == 1:
                gray_frame, faces = face_detection(frame)
                detections = []

                for x, y, w, h in faces:
                    face_roi = gray_frame[y : y + h, x : x + w]
                    label, score = recognize_face(face_roi)
                    detections.append((x, y, w, h, label, score))

                state["detections"] = detections

            detections = state["detections"]
            last_label = "Unknown"

            for x, y, w, h, label, score in detections:
                last_label = f"{label} ({score})"
                color = (0, 200, 0) if label != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(
                    frame,
                    last_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if label != "Unknown":
                    notify_member_detected(label, score)

            if not KNOWN_FACES:
                cv2.putText(
                    frame,
                    "Folder known_faces masih kosong",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            image = frame_to_tk_image(frame)
        except Exception as exc:
            camera_status.set("Frame kamera tidak terbaca.")
            status_var.set("Preview kamera gagal.")
            messagebox.showerror("Preview kamera gagal", str(exc))
            close()
            return

        preview_label.configure(image=image)
        preview_label.image = image
        camera_status.set(f"Wajah: {len(detections)} | {last_label}")
        state["after_id"] = window.after(25, update_preview)

    footer = ttk.Frame(window, padding=(18, 12, 18, 18), style="App.TFrame")
    footer.pack(fill="x")
    ttk.Button(footer, text="Tutup", command=close).pack(side="right")
    window.protocol("WM_DELETE_WINDOW", close)
    window.lift()
    window.focus_force()
    camera_status.set("Menyiapkan kamera...")
    state["after_id"] = window.after(120, update_preview)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["recognize", "register"],
        default="recognize",
        help="recognize untuk deteksi wajah, register untuk simpan customer ke Laravel",
    )
    parser.add_argument("--name", help="Nama customer saat mode register")
    parser.add_argument("--phone", default="", help="Nomor telepon customer")
    parser.add_argument("--discount", type=int, default=0, help="Diskon member persen")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL API Laravel")
    parser.add_argument("--camera-index", type=int, default=0, help="Index kamera OpenCV")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(__import__("sys").argv) == 1:
        launch_gui()
        return

    if args.mode == "register":
        if not args.name:
            raise RuntimeError("Mode register butuh --name.")
        register_customer(args.name, args.phone, args.discount, args.api_url, args.camera_index)
        return

    run_recognition(args.camera_index)


def configure_styles(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    font_family = "Segoe UI"
    if platform.system() == "Darwin":
        font_family = "Helvetica"

    root.configure(bg="#f5f7fb")
    style.configure(".", font=(font_family, 10))
    style.configure("App.TFrame", background="#f5f7fb")
    style.configure("Surface.TFrame", background="#ffffff", relief="flat")
    style.configure("Hero.TFrame", background="#17324d")
    style.configure("Title.TLabel", background="#f5f7fb", foreground="#152033", font=(font_family, 18, "bold"))
    style.configure("HeroTitle.TLabel", background="#17324d", foreground="#ffffff", font=(font_family, 20, "bold"))
    style.configure("HeroText.TLabel", background="#17324d", foreground="#dbeafe", font=(font_family, 10))
    style.configure("Field.TLabel", background="#ffffff", foreground="#344054", font=(font_family, 10, "bold"))
    style.configure("Muted.TLabel", background="#f5f7fb", foreground="#667085")
    style.configure("SurfaceMuted.TLabel", background="#ffffff", foreground="#667085")
    style.configure("Status.TLabel", background="#f5f7fb", foreground="#175cd3", font=(font_family, 10, "bold"))
    style.configure("Preview.TLabel", background="#101828")
    style.configure(
        "TEntry",
        fieldbackground="#ffffff",
        bordercolor="#d0d5dd",
        lightcolor="#d0d5dd",
        darkcolor="#d0d5dd",
        padding=7,
    )
    style.configure("TButton", padding=(14, 8), font=(font_family, 10, "bold"))
    style.configure("Accent.TButton", background="#0f766e", foreground="#ffffff")
    style.map(
        "Accent.TButton",
        background=[("active", "#0d9488"), ("pressed", "#115e59")],
        foreground=[("disabled", "#e5e7eb"), ("!disabled", "#ffffff")],
    )


def launch_gui():
    root = tk.Tk()
    root.title("Pingkal Face App")
    root.geometry("620x520")
    root.resizable(False, False)
    configure_styles(root)

    main_frame = ttk.Frame(root, padding=20, style="App.TFrame")
    main_frame.pack(fill="both", expand=True)

    hero_frame = ttk.Frame(main_frame, padding=18, style="Hero.TFrame")
    hero_frame.pack(fill="x")
    ttk.Label(hero_frame, text="Pingkal Face App", style="HeroTitle.TLabel").pack(anchor="w")
    ttk.Label(
        hero_frame,
        text="Register customer faces and run live member recognition from one screen.",
        style="HeroText.TLabel",
    ).pack(anchor="w", pady=(6, 0))

    form_frame = ttk.Frame(main_frame, padding=18, style="Surface.TFrame")
    form_frame.pack(fill="x", pady=(16, 12))
    form_frame.columnconfigure(0, weight=1)
    form_frame.columnconfigure(1, weight=1)

    name_var = tk.StringVar()
    phone_var = tk.StringVar()
    discount_var = tk.StringVar(value="0")
    api_var = tk.StringVar(value=DEFAULT_API_URL)
    camera_index_var = tk.StringVar(value="0")

    def add_field(row, column, label, variable, width=30):
        field_frame = ttk.Frame(form_frame, style="Surface.TFrame")
        field_frame.grid(row=row, column=column, sticky="ew", padx=(0, 10) if column == 0 else (10, 0), pady=7)
        ttk.Label(field_frame, text=label, style="Field.TLabel").pack(anchor="w")
        entry = ttk.Entry(field_frame, textvariable=variable, width=width)
        entry.pack(fill="x", pady=(5, 0))
        return entry

    add_field(0, 0, "Nama Customer", name_var)
    add_field(0, 1, "No. Telepon", phone_var)
    add_field(1, 0, "Diskon (%)", discount_var)
    add_field(1, 1, "Camera Index", camera_index_var)

    api_frame = ttk.Frame(form_frame, style="Surface.TFrame")
    api_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
    ttk.Label(api_frame, text="Laravel API", style="Field.TLabel").pack(anchor="w")
    ttk.Entry(api_frame, textvariable=api_var).pack(fill="x", pady=(5, 0))

    status_var = tk.StringVar(value="Siap dipakai.")
    status_frame = ttk.Frame(main_frame, padding=(14, 10), style="Surface.TFrame")
    status_frame.pack(fill="x", pady=(0, 12))
    ttk.Label(status_frame, text="Status", style="Field.TLabel").pack(side="left")
    ttk.Label(status_frame, textvariable=status_var, style="SurfaceMuted.TLabel").pack(side="left", padx=(12, 0))

    def on_register():
        name = name_var.get().strip()
        phone = phone_var.get().strip()
        api_url = api_var.get().strip()

        if not name:
            messagebox.showwarning("Data belum lengkap", "Nama customer wajib diisi.")
            return

        try:
            discount = int(discount_var.get().strip() or "0")
        except ValueError:
            messagebox.showwarning("Diskon tidak valid", "Diskon harus berupa angka.")
            return

        try:
            camera_index = int(camera_index_var.get().strip() or "0")
        except ValueError:
            messagebox.showwarning("Camera index tidak valid", "Camera index harus berupa angka.")
            return

        status_var.set("Membuka kamera register...")
        root.update_idletasks()
        launch_register_camera(root, name, phone, discount, api_url, camera_index, status_var)

    def on_recognize():
        try:
            camera_index = int(camera_index_var.get().strip() or "0")
        except ValueError:
            messagebox.showwarning("Camera index tidak valid", "Camera index harus berupa angka.")
            return

        status_var.set("Membuka kamera recognize...")
        root.update_idletasks()
        launch_recognition_camera(root, camera_index, status_var)

    button_frame = ttk.Frame(main_frame, style="App.TFrame")
    button_frame.pack(fill="x", pady=(2, 0))

    ttk.Button(button_frame, text="Register Wajah", command=on_register, style="Accent.TButton").pack(side="left")
    ttk.Button(button_frame, text="Mode Recognize", command=on_recognize).pack(side="left", padx=10)
    ttk.Button(button_frame, text="Keluar", command=root.destroy).pack(side="right")

    tips_frame = ttk.Frame(main_frame, padding=(14, 12), style="Surface.TFrame")
    tips_frame.pack(fill="x", pady=(16, 0))
    ttk.Label(tips_frame, text="Tips", style="Field.TLabel").pack(anchor="w")
    ttk.Label(
        tips_frame,
        text="Jalankan Laravel dulu. Untuk kamera eksternal di Mac, coba Camera Index 1 atau 2.",
        style="SurfaceMuted.TLabel",
        wraplength=540,
    ).pack(anchor="w", pady=(5, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
