import math
from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import pygame


# -------------------- SETTINGS --------------------
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 900
DOT_SPACING = 4          # Smaller = more detail, but slower
MAX_DOTS = 30000
FPS = 60
ANIMATION_SECONDS = 5.0
START_DOT_SIZE = 1
FINAL_DOT_SIZE = 2
BACKGROUND = (5, 5, 8)
OUTPUT_SUFFIX = "_dot_portrait.mp4"
# --------------------------------------------------


def choose_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose a portrait image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def fit_image(image, width, height):
    """Resize the full image to fit the canvas without stretching it."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, (x0, y0, new_w, new_h)


def make_dots(image_rgb):
    """
    Sample the whole image, not a YOLO/person mask.
    Each dot keeps the original pixel's colour and target position.
    """
    h, w = image_rgb.shape[:2]
    ys = np.arange(0, h, DOT_SPACING)
    xs = np.arange(0, w, DOT_SPACING)
    grid_x, grid_y = np.meshgrid(xs, ys)

    target_x = grid_x.reshape(-1).astype(np.float32)
    target_y = grid_y.reshape(-1).astype(np.float32)
    colors = image_rgb[target_y.astype(int), target_x.astype(int)].astype(np.uint8)

    # Ignore nearly black pixels only when the background is actually black.
    # Keep all image pixels otherwise, so facial detail is not removed.
    brightness = colors.max(axis=1)
    keep = brightness > 8
    target_x = target_x[keep]
    target_y = target_y[keep]
    colors = colors[keep]

    if len(target_x) > MAX_DOTS:
        # Deterministic even sampling preserves coverage across the entire image.
        chosen = np.linspace(0, len(target_x) - 1, MAX_DOTS).astype(np.int32)
        target_x = target_x[chosen]
        target_y = target_y[chosen]
        colors = colors[chosen]

    return target_x, target_y, colors


def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def main():
    image_path = choose_image()
    if not image_path:
        print("No image selected. Exiting.")
        return

    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        print("Could not open the selected image.")
        return

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    fitted_rgb, _ = fit_image(original_rgb, CANVAS_WIDTH, CANVAS_HEIGHT)

    target_x, target_y, colors = make_dots(fitted_rgb)
    count = len(target_x)
    if count == 0:
        print("No visible dots were found in this image.")
        return

    # Every dot starts at a random location and moves to its original-image pixel.
    random.seed(17)
    rng = np.random.default_rng(17)
    start_x = rng.uniform(0, CANVAS_WIDTH, count).astype(np.float32)
    start_y = rng.uniform(0, CANVAS_HEIGHT, count).astype(np.float32)

    pygame.init()
    screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
    pygame.display.set_caption("Dot Portrait Animation")
    clock = pygame.time.Clock()

    output_path = str(Path(image_path).with_name(
        Path(image_path).stem + OUTPUT_SUFFIX
    ))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        output_path, fourcc, FPS, (CANVAS_WIDTH, CANVAS_HEIGHT)
    )

    if not video.isOpened():
        pygame.quit()
        raise RuntimeError(
            "Could not create the MP4 file. Try another folder or check write permissions."
        )

    total_frames = int(ANIMATION_SECONDS * FPS)
    running = True
    frame_number = 0

    try:
        while running and frame_number < total_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            progress = frame_number / max(1, total_frames - 1)
            eased = ease_in_out(progress)

            # Dots start tiny and grow as they converge, revealing facial details.
            current_x = start_x + (target_x - start_x) * eased
            current_y = start_y + (target_y - start_y) * eased
            radius = START_DOT_SIZE + (FINAL_DOT_SIZE - START_DOT_SIZE) * eased

            screen.fill(BACKGROUND)

            # A short fade-in helps the face become clear instead of appearing abruptly.
            alpha = min(1.0, progress * 1.6 + 0.08)
            for i in range(count):
                r, g, b = map(int, colors[i])
                color = (
                    int(r * alpha),
                    int(g * alpha),
                    int(b * alpha),
                )
                pygame.draw.circle(
                    screen,
                    color,
                    (int(current_x[i]), int(current_y[i])),
                    max(1, int(round(radius))),
                )

            pygame.display.flip()

            # Capture the displayed frame and convert RGB -> BGR for OpenCV.
            frame_rgb = pygame.surfarray.array3d(screen)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

            frame_number += 1
            clock.tick(FPS)

    finally:
        video.release()
        pygame.quit()

    print(f"Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
