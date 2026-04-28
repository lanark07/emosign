import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel

from app import config
from app.gui.theme import SURFACE, BORDER


# Displays the live camera frame with a dimmed-outside-ROI overlay, corner-bracket
# box, and a floating letter badge. Subclasses QLabel so setPixmap() handles scaling.
class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Camera initialising…")
        self.setStyleSheet(
            f"background: {SURFACE}; color: #9ca3af; font-size: 14px;"
            "border-radius: 10px; border: 1px solid #e2e2ee;"
        )
        self.setScaledContents(False)
        self._last_letter = ""
        self._last_conf   = 0.0

    def update_frame(self, frame: np.ndarray):
        pixmap = self._to_pixmap(self._draw_overlay(frame.copy()))
        self.setPixmap(
            pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def set_asl_result(self, letter: str, conf: float):
        self._last_letter = letter
        self._last_conf   = conf

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w   = frame.shape[:2]
        half   = config.ROI_SIZE // 2
        cx, cy = w // 2, h // 2
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        # Dim everything outside the ROI so the user's eye is drawn to the hand box.
        mask  = np.zeros_like(frame)
        frame = cv2.addWeighted(mask, 0.38, frame, 0.62, 0)
        # Restore full brightness inside the ROI after the global dim.
        frame[y1:y2, x1:x2] = frame.copy()[y1:y2, x1:x2]

        letter = self._last_letter
        conf   = self._last_conf

        # Box colour gives instant visual feedback: grey = no hand, purple = confident, amber = uncertain.
        if letter in ("nothing", "unknown", ""):
            bracket_color = (160, 160, 180)
        elif conf >= config.ASL_CONF_THRESHOLD:
            bracket_color = (124, 58, 237)   # purple — confident
        else:
            bracket_color = (217, 119, 6)    # amber — uncertain

        self._draw_corner_box(frame, x1, y1, x2, y2, bracket_color, thickness=3, length=22)

        if letter in ("nothing", "unknown", ""):
            cv2.putText(frame, "Place hand inside the box",
                        (cx - 88, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 180), 1, cv2.LINE_AA)
        else:
            self._draw_letter_badge(frame, letter.upper(), conf, 0, 0, w, h)

        return frame

    @staticmethod
    def _draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, length=20):
        for corner, h_pt, v_pt in [
            ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
        ]:
            cv2.line(frame, corner, h_pt, color, thickness, cv2.LINE_AA)
            cv2.line(frame, corner, v_pt, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_letter_badge(frame, letter: str, conf: float,
                           x1: int, y1: int, x2: int, y2: int):
        pad  = 10
        single = len(letter) == 1

        # Pick font scale — shrink until text fits inside the ROI width
        max_w   = (x2 - x1) - pad * 2
        scale   = 1.6 if single else 0.72
        thick   = 3   if single else 2
        (tw, th), base = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        while tw > max_w and scale > 0.35:
            scale -= 0.05
            (tw, th), base = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)

        # Badge rect — anchored to ROI top-left corner
        bx1 = x1 + pad
        by1 = y1 + pad
        bx2 = bx1 + tw + pad * 2
        by2 = by1 + th + base + pad * 2 + 8   # +8 for conf bar

        # Clamp to stay inside ROI
        bx2 = min(bx2, x2 - pad)
        by2 = min(by2, y2 - pad)

        # Colors — light badge on the camera frame
        if conf >= config.ASL_CONF_THRESHOLD:
            bg_color   = (245, 242, 255)   # pale lavender
            accent     = (124, 58, 237)    # purple
        else:
            bg_color   = (255, 248, 235)   # pale amber
            accent     = (180, 110, 0)     # dark amber

        # Background with slight transparency feel via blend
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
        frame[:] = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)

        # Border
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), accent, 1, cv2.LINE_AA)

        # Letter text
        tx = bx1 + pad
        ty = by1 + pad + th
        cv2.putText(frame, letter, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, accent, thick, cv2.LINE_AA)

        # Confidence bar at bottom of badge
        bar_y1 = by2 - 7
        bar_y2 = by2 - 2
        bar_x1 = bx1 + 2
        bar_x2 = bx2 - 2
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (210, 210, 225), -1)
        fill_w = int((bar_x2 - bar_x1) * min(conf, 1.0))
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), accent, -1)

    @staticmethod
    def _to_pixmap(frame: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
