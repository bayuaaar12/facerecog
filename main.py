import argparse
import base64
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from urllib import error, request

import cv2


BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "face_ref.xml"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/customers/register-face"

face_ref = cv2.CascadeClassifier(str(CASCADE_PATH))
orb = cv2.ORB_create(nfeatures=700)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


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


def find_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda item: item[2] * item[3])


def register_customer(name, phone, discount_percent, api_url):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera tidak bisa dibuka.")

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


def run_recognition():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera tidak bisa dibuka.")

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
    return parser.parse_args()


def main():
    args = parse_args()

    if len(__import__("sys").argv) == 1:
        launch_gui()
        return

    if args.mode == "register":
        if not args.name:
            raise RuntimeError("Mode register butuh --name.")
        register_customer(args.name, args.phone, args.discount, args.api_url)
        return

    run_recognition()


def launch_gui():
    root = tk.Tk()
    root.title("Pingkal Face App")
    root.geometry("430x320")
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=18)
    main_frame.pack(fill="both", expand=True)

    ttk.Label(main_frame, text="Face Registration", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(
        main_frame,
        text="Isi data customer lalu klik tombol kamera.",
        font=("Segoe UI", 10),
    ).pack(anchor="w", pady=(4, 14))

    form_frame = ttk.Frame(main_frame)
    form_frame.pack(fill="x")

    name_var = tk.StringVar()
    phone_var = tk.StringVar()
    discount_var = tk.StringVar(value="0")
    api_var = tk.StringVar(value=DEFAULT_API_URL)

    ttk.Label(form_frame, text="Nama Customer").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(form_frame, textvariable=name_var, width=38).grid(row=0, column=1, pady=4)

    ttk.Label(form_frame, text="No. Telepon").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(form_frame, textvariable=phone_var, width=38).grid(row=1, column=1, pady=4)

    ttk.Label(form_frame, text="Diskon (%)").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Entry(form_frame, textvariable=discount_var, width=38).grid(row=2, column=1, pady=4)

    ttk.Label(form_frame, text="Laravel API").grid(row=3, column=0, sticky="w", pady=4)
    ttk.Entry(form_frame, textvariable=api_var, width=38).grid(row=3, column=1, pady=4)

    status_var = tk.StringVar(value="Siap dipakai.")
    ttk.Label(main_frame, textvariable=status_var, foreground="#1f4e79").pack(anchor="w", pady=(14, 10))

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

        status_var.set("Membuka kamera register...")
        root.update_idletasks()

        try:
            response = register_customer(name, phone, discount, api_url)
        except Exception as exc:
            status_var.set("Register gagal.")
            messagebox.showerror("Register gagal", str(exc))
            return

        if response:
            status_var.set("Customer berhasil disimpan.")
            messagebox.showinfo("Berhasil", "Wajah customer berhasil disimpan ke Laravel.")
        else:
            status_var.set("Register dibatalkan.")

    def on_recognize():
        status_var.set("Membuka kamera recognize...")
        root.update_idletasks()
        try:
            run_recognition()
        except Exception as exc:
            status_var.set("Recognize gagal.")
            messagebox.showerror("Recognize gagal", str(exc))
            return

        status_var.set("Mode recognize selesai.")

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x", pady=(6, 0))

    ttk.Button(button_frame, text="Register Wajah", command=on_register).pack(side="left")
    ttk.Button(button_frame, text="Mode Recognize", command=on_recognize).pack(side="left", padx=10)
    ttk.Button(button_frame, text="Keluar", command=root.destroy).pack(side="right")

    help_text = (
        "Cara pakai:\n"
        "1. Jalankan Laravel dulu.\n"
        "2. Isi nama customer.\n"
        "3. Klik Register Wajah.\n"
        "4. Di jendela kamera tekan C untuk simpan."
    )
    ttk.Label(main_frame, text=help_text, justify="left").pack(anchor="w", pady=(16, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
