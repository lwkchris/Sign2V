# Application/bubble_style.py
import tkinter as tk

class RoundedFrame(tk.Canvas):
    def __init__(self, parent, bg_color, *args, **kwargs):
        tk.Canvas.__init__(self, parent, *args, **kwargs, bg="#ffffff", highlightthickness=0)
        self.bg_color = bg_color

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        radius *= 1.2
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True, splinesteps=64, joinstyle=tk.ROUND)