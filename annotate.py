import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==============================
# Configuration
# ==============================
PAUSE_DURATION_SEC = 3
DETECTION_MODEL = "yolov8n.pt"

# Animated ring configuration
RING_COLORS = [
    (0, 255, 255),   # Yellow
    (0, 150, 255),   # Orange
    (255, 255, 255)  # White
]

RING_COLORS = [
    (0, 100, 255),    # 🔥 Deep Orange
    (0, 50, 255),     # 🧡 Red-Orange
    (80, 80, 255)     # ❤️ Soft Red/White edge
]
ELLIPSE_FLATTEN_RATIO = 0.25
ELLIPSE_SIZE_SCALE = 2
ELLIPSE_THICKNESS = 6
GLOW_ALPHA = 1
display_scale = 0.65
ANIMATION_SPEED = 2.5   # higher = faster pulsing

# ==============================
# Globals
# ==============================
current_frame = None
current_frame_index = 0
temp_annotations = []
annotation_events = []
detected_players = []
display_size = None
window_name = "Football Annotator"
animation_phase = 0.0

model = YOLO(DETECTION_MODEL)

# ==============================
# Player Detection
# ==============================
def detect_players(frame):
    results = model.predict(frame, verbose=False)
    boxes = []
    for r in results[0].boxes:
        cls = int(r.cls)
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes

# ==============================
# Geometry
# ==============================
def get_ground_ellipse_from_box(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center = ((x1 + x2)//2, int(y2 - height * 0.05))
    axes = (
        int(width * 0.5 * ELLIPSE_SIZE_SCALE),
        int(height * 0.15 * ELLIPSE_SIZE_SCALE)
    )
    return (center, axes)

# ==============================
# Animated Multicolor Rings + Chroma Key
# ==============================
def draw_animated_rings_with_mask(frame, annotations, phase):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)

    glow_layer = np.zeros_like(frame)

    for center, axes, box in annotations:
        x1, y1, x2, y2 = box

        # estimate leg region (lower 1/3)
        leg_y_start = int(y1 + (y2 - y1) * 0.65)
        leg_y_end = y2
        leg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if leg_y_end > leg_y_start:
            leg_region = frame[leg_y_start:leg_y_end, x1:x2]
            if leg_region.size > 0:
                leg_hsv = cv2.cvtColor(leg_region, cv2.COLOR_BGR2HSV)
                leg_green = cv2.inRange(leg_hsv, lower_green, upper_green)
                non_green = cv2.bitwise_not(leg_green)
                leg_mask[leg_y_start:leg_y_end, x1:x2] = non_green

        flat_axes = (axes[0], int(axes[1] * ELLIPSE_FLATTEN_RATIO))

        # Compute pulse scaling using a sine wave
        pulse = 1.0 + 0.12 * np.sin(phase)
        for i, color in enumerate(RING_COLORS):
            scale_factor = pulse + i * 0.08
            scaled_axes = (
                int(flat_axes[0] * scale_factor),
                int(flat_axes[1] * scale_factor)
            )
            alpha = 1 - (i / len(RING_COLORS))
            col = tuple(int(c * alpha) for c in color)
            cv2.ellipse(glow_layer, center, scaled_axes, 0, 0, 360, col, ELLIPSE_THICKNESS)

        # glow_layer = cv2.GaussianBlur(glow_layer, (25, 25), 12)
        # glow_layer = cv2.GaussianBlur(glow_layer, (9, 9), 4)
        glow_layer = cv2.GaussianBlur(glow_layer, (5, 5), 2)

        # keep only grass, remove legs
        valid_area = cv2.bitwise_and(grass_mask, cv2.bitwise_not(leg_mask))
        glow_layer = cv2.bitwise_and(glow_layer, glow_layer, mask=valid_area)

    return cv2.addWeighted(frame, 1.0, glow_layer, GLOW_ALPHA, 0)

# ==============================
# Mouse + Display
# ==============================
def handle_mouse_click(event, x, y, flags, param):
    global temp_annotations, current_frame, detected_players, display_size

    if event == cv2.EVENT_LBUTTONDOWN and detected_players:
        if display_size is not None:
            disp_w, disp_h = display_size
            orig_h, orig_w = current_frame.shape[:2]
            scale_x = orig_w / disp_w
            scale_y = orig_h / disp_h
            x = int(x * scale_x)
            y = int(y * scale_y)

        nearest_box = find_nearest_player_box(x, y, detected_players)
        if nearest_box:
            ellipse = get_ground_ellipse_from_box(nearest_box)
            temp_annotations.append((*ellipse, nearest_box))

def show_scaled(frame):
    global display_size
    h, w = frame.shape[:2]
    new_w, new_h = int(w * display_scale), int(h * display_scale)
    frame_disp = cv2.resize(frame, (new_w, new_h))
    display_size = (new_w, new_h)
    cv2.imshow(window_name, frame_disp)

def find_nearest_player_box(x, y, boxes):
    min_dist = float('inf')
    nearest = None
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < min_dist:
            min_dist = dist
            nearest = (x1, y1, x2, y2)
    return nearest

# ==============================
# Save Annotated Video
# ==============================
def save_annotated_video(video_path, annotation_events, fps, width, height):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter("annotated_output.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    annotation_dict = {f: a for f, a in annotation_events}
    pause_frames = int(fps * PAUSE_DURATION_SEC)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in annotation_dict:
            for t in range(pause_frames):
                phase = t * 0.2
                animated = draw_animated_rings_with_mask(frame.copy(), annotation_dict[frame_idx], phase)
                out.write(animated)
        else:
            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

# ==============================
# Main
# ==============================
def play_and_annotate(video_path):
    global current_frame, current_frame_index, temp_annotations, annotation_events, detected_players, animation_phase

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    paused = False
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse_click)
    print_controls()

    last_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("🏁 End of video.")
                break
            current_frame = frame.copy()
            current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            detected_players = detect_players(current_frame)
        else:
            frame = current_frame.copy()
            # animate ring with smooth time-based phase
            now = time.time()
            dt = now - last_time
            animation_phase += dt * ANIMATION_SPEED * np.pi
            frame = draw_animated_rings_with_mask(frame, temp_annotations, animation_phase)
            last_time = now

        for (x1, y1, x2, y2) in detected_players:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 1)

        show_scaled(frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if paused and temp_annotations:
                annotation_events.append((current_frame_index, temp_annotations.copy()))
                temp_annotations.clear()
                print(f"✅ Stored highlights for frame {current_frame_index}")
            paused = not paused
            last_time = time.time()
        elif key == ord('u') and paused:
            if temp_annotations:
                temp_annotations.pop()
                print("↩️ Undo last highlight.")
        elif key == ord('s'):
            print("💾 Saving annotated video...")
            save_annotated_video(video_path, annotation_events, fps, width, height)
            print("✅ Saved as 'annotated_output.mp4'")

    cap.release()
    cv2.destroyAllWindows()

def print_controls():
    print("\n🎮 Controls:")
    print("Space = Pause/Play")
    print("Click = Highlight player (animated ring)")
    print("U = Undo (while paused)")
    print("S = Save annotated video")
    print("Q = Quit\n")

if __name__ == "__main__":
    video_path = input("Enter path to video file: ")
    play_and_annotate(video_path)
