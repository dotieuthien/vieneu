"""Render docs/how-it-works.mp4 (+ .srt, poster) — giải thích VieNeu cho người không chuyên.

    python docs/video/make_video.py --voice http://127.0.0.1:PORT   # đọc lời bằng server lite
    python docs/video/make_video.py                                 # dùng giọng đã cache (hoặc video câm)

Cần Pillow và ffmpeg (libx264, aac). Phụ đề được vẽ thẳng vào hình và cũng xuất ra .srt.
Giọng đọc của từng câu được cache trong docs/video/.voice/ theo nội dung câu, nên chỉ
câu nào sửa mới phải gọi lại server. Thời lượng mỗi câu = độ dài giọng đọc + khoảng nghỉ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import urllib.request
import wave as wavfile
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
OUT = Path(__file__).resolve().parents[1]
VOICE_DIR = Path(__file__).resolve().parent / ".voice"
SR = 48_000
PAUSE = 0.45            # giây nghỉ sau mỗi câu
SCENE_PAUSE = 0.6       # nghỉ thêm khi chuyển cảnh
# Cách viết để TTS đọc đúng (phụ đề vẫn giữ chữ gốc).
SPOKEN = {"VieNeu": "Vi Neu", "mili giây": "mi li giây"}
FONT_DIR = Path(os.getenv("FONT_DIR", "/System/Library/Fonts/Supplemental"))


def font(size, bold=False, uni=False):
    name = "Arial Unicode.ttf" if uni else ("Arial Bold.ttf" if bold else "Arial.ttf")
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_TITLE, F_H, F_B, F_S, F_XS = font(40, True), font(30, True), font(26), font(22), font(18)
F_SUB = font(28)

BG = (247, 246, 241)
INK = (31, 35, 40)
MUTED = (110, 118, 129)
LINE = (208, 212, 218)
BLUE = (47, 111, 235)
ORANGE = (232, 131, 58)
GREEN = (45, 164, 78)
RED = (207, 34, 46)
PURPLE = (130, 80, 223)
TEAL = (26, 150, 160)
USERS = [BLUE, ORANGE, GREEN, PURPLE, TEAL, RED, (191, 135, 0), (219, 97, 162)]


# ── drawing helpers ──────────────────────────────────────────────────────────

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def seg(p, a, b):
    """Progress 0..1 of ``p`` inside [a, b]."""
    return clamp((p - a) / (b - a)) if b > a else float(p >= a)


def mix(c, a):
    """Fade colour ``c`` into the background: a=1 full colour, 0 invisible."""
    return tuple(int(BG[i] + (c[i] - BG[i]) * clamp(a)) for i in range(3))


def tint(c, k=0.18):
    return tuple(int(255 + (c[i] - 255) * k) for i in range(3))


def rbox(d, x, y, w, h, fill, outline=None, r=14, width=2):
    d.rounded_rectangle([x, y, x + w, y + h], r, fill=fill, outline=outline, width=width)


def text(d, x, y, s, f, fill=INK, anchor="la"):
    d.text((x, y), s, font=f, fill=fill, anchor=anchor)


def ctext(d, cx, cy, s, f, fill=INK):
    lines = s.split("\n")
    lh = f.size * 1.25
    y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        d.text((cx, y0 + i * lh), ln, font=f, fill=fill, anchor="mm")


def arrow(d, x1, y1, x2, y2, fill=MUTED, w=4, head=14):
    d.line([x1, y1, x2, y2], fill=fill, width=w)
    a = math.atan2(y2 - y1, x2 - x1)
    pts = [(x2, y2),
           (x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45)),
           (x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45))]
    d.polygon(pts, fill=fill)


def play_icon(d, cx, cy, s, fill):
    d.polygon([(cx - s * 0.4, cy - s * 0.5), (cx - s * 0.4, cy + s * 0.5), (cx + s * 0.55, cy)], fill=fill)


def chip(d, x, y, s, color, f=F_S):
    w = d.textlength(s, font=f) + 24
    rbox(d, x, y, w, f.size + 14, tint(color, 0.16), outline=color, r=12)
    text(d, x + 12, y + 7, s, f, color)
    return w


def wave(d, x, y, w, h, color, phase=0.0, n=40, width=3):
    pts = []
    for i in range(n + 1):
        u = i / n
        amp = math.sin(u * math.pi) * (0.55 + 0.45 * math.sin(u * 17 + phase))
        pts.append((x + u * w, y + amp * h / 2 * math.sin(u * 40 + phase * 3)))
    d.line(pts, fill=color, width=width)


def listener(d, cx, cy, color=INK):
    d.ellipse([cx - 26, cy - 60, cx + 26, cy - 8], outline=color, width=4)
    d.arc([cx - 46, cy - 72, cx + 46, cy + 20], 200, 340, fill=color, width=5)
    d.rounded_rectangle([cx - 52, cy - 34, cx - 38, cy - 4], 5, fill=color)
    d.rounded_rectangle([cx + 38, cy - 34, cx + 52, cy - 4], 5, fill=color)
    d.arc([cx - 50, cy + 2, cx + 50, cy + 90], 200, 340, fill=color, width=4)


def wrap(s, f, maxw, d):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        test = f"{cur} {wd}".strip()
        if d.textlength(test, font=f) <= maxw or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


# ── scenes ───────────────────────────────────────────────────────────────────
# Each scene: (step label, title, draw(d, p, t), [subtitle cues]). p = 0..1 through the scene.

def s_title(d, p, t):
    ctext(d, W / 2, 230, "VieNeu-TTS đọc văn bản\nvà phát tiếng ngay như thế nào?", font(48, True))
    ctext(d, W / 2, 345, "Giải thích từng bước · không cần biết kỹ thuật", F_B, MUTED)
    for i, c in enumerate([BLUE, ORANGE, GREEN]):
        wave(d, 290, 450 + i * 0, 700, 70, mix(c, 0.85), phase=t * 2 + i * 2.1, n=80)


def s_stream(d, p, t):
    x0, x1 = 300, 1120
    text(d, x0, 140, "0 giây", F_S, MUTED)
    text(d, x1, 140, "thời gian →", F_S, MUTED, anchor="ra")
    head = x0 + (x1 - x0) * ease(seg(p, 0.05, 0.95))

    def lane(y, label, gen_end, play_start, color):
        text(d, 60, y + 4, label, F_H)
        text(d, 60, y + 44, "tạo", F_S, MUTED)
        text(d, 60, y + 84, "phát", F_S, MUTED)
        ge = x0 + (x1 - x0) * gen_end
        ps = x0 + (x1 - x0) * play_start
        rbox(d, x0, y + 40, max(4, min(head, ge) - x0), 28, mix(MUTED, 0.55), r=8)
        if head > ps:
            rbox(d, ps, y + 80, head - ps, 28, color, r=8)
        d.line([ps, y + 36, ps, y + 118], fill=color, width=3)
        text(d, ps + 8, y + 118, "nghe được tiếng đầu tiên", F_XS, color)

    lane(170, "Cách thường: đợi tạo xong hết", 0.62, 0.62, RED)
    lane(350, "Streaming: xong mẩu nào gửi mẩu đó", 0.62, 0.07, GREEN)
    d.line([head, 160, head, 500], fill=INK, width=2)


def s_phoneme(d, p, t):
    a = ease(seg(p, 0.0, 0.25))
    b = ease(seg(p, 0.3, 0.6))
    rbox(d, 140, 210, 360, 150, mix((255, 255, 255), a), outline=mix(INK, a), r=18)
    ctext(d, 320, 285, "Xin chào", font(52, True), mix(INK, a))
    ctext(d, 320, 400, "chữ viết", F_B, mix(MUTED, a))
    arrow(d, 530, 285, 530 + 200 * b, 285, mix(BLUE, max(b, 0.01)), w=6, head=18)
    rbox(d, 760, 210, 380, 150, mix(tint(BLUE, 0.12), b), outline=mix(BLUE, b), r=18)
    ctext(d, 950, 272, "s·i·n   c·a·o", font(46, True), mix(BLUE, b))
    ctext(d, 950, 322, "+ thanh huyền", F_S, mix(BLUE, b))
    ctext(d, 950, 400, "cách phát âm (phoneme)", F_B, mix(MUTED, b))
    c = ease(seg(p, 0.65, 0.85))
    ctext(d, W / 2, 500, "Công cụ sea-g2p làm việc này trên CPU, rất nhanh.", F_B, mix(INK, c))


def s_ar(d, p, t):
    n, size, gap, x0, y0 = 8, 92, 34, 120, 150
    shown = 1 + int(seg(p, 0.02, 0.55) * (n - 1) + 1e-6)
    rng = random.Random(7)
    for i in range(n):
        x = x0 + i * (size + gap)
        if i < shown:
            col = USERS[i % 2 and 0 or 0]
            rbox(d, x, y0, size, size, tint(BLUE, 0.15), outline=BLUE, r=12)
            ctext(d, x + size / 2, y0 + size / 2 - 10, f"mẩu {i + 1}", F_S, BLUE)
            ctext(d, x + size / 2, y0 + size / 2 + 18, "80 ms", F_XS, MUTED)
            if i:
                d.arc([x - gap - size / 2, y0 - 46, x + size / 2, y0 + 30], 200, 340, fill=ORANGE, width=3)
                arrow(d, x + size / 2 - 14, y0 - 12, x + size / 2 - 2, y0 - 2, ORANGE, w=3, head=10)
        else:
            rbox(d, x, y0, size, size, BG, outline=LINE, r=12)
    text(d, x0, y0 + size + 18, "Tầng 1: mẩu sau được đoán dựa trên các mẩu trước (mũi tên cam).", F_S, INK)

    z = ease(seg(p, 0.5, 0.62))
    if z > 0:
        y1 = 400
        text(d, x0, y1 - 46, "Tầng 2: bên trong 1 mẩu là 16 con số, cũng điền lần lượt từng số:", F_S, mix(INK, z))
        filled = int(seg(p, 0.6, 0.95) * 16 + 1e-6)
        cw = 62
        for k in range(16):
            x = x0 + k * (cw + 6)
            on = k < filled
            rbox(d, x, y1, cw, 56, tint(ORANGE, 0.18) if on else BG, outline=ORANGE if on else LINE, r=8)
            v = rng.randint(0, 1023)
            if on:
                ctext(d, x + cw / 2, y1 + 28, str(v), F_S, ORANGE)
        ctext(d, W / 2, y1 + 100, "16 con số = bản nén của âm thanh, chưa phải tiếng. Lặp lại tới khi model báo hết câu.",
              F_S, mix(MUTED, z))


def gpu_timeline(d, x, y, w, busy_frac, t, color=BLUE, blocks=24, label=True):
    rbox(d, x, y, w, 34, (232, 234, 238), r=8)
    bw = w / blocks
    for i in range(blocks):
        bx = x + i * bw
        d.rectangle([bx + 2, y + 4, bx + 2 + max(2, bw * busy_frac - 4), y + 30], fill=color)
    if label:
        text(d, x, y + 44, "xanh = GPU đang tính · xám = GPU đứng chờ lệnh", F_XS, MUTED)


def s_problem(d, p, t):
    rbox(d, 110, 170, 300, 150, (255, 255, 255), outline=INK, r=18)
    ctext(d, 260, 230, "Python / CPU", F_H)
    ctext(d, 260, 275, "người ra lệnh", F_S, MUTED)
    rbox(d, 870, 170, 300, 150, tint(BLUE, 0.12), outline=BLUE, r=18)
    ctext(d, 1020, 230, "GPU", F_H, BLUE)
    ctext(d, 1020, 275, "máy tính siêu nhanh", F_S, MUTED)
    # small commands travelling one by one, with gaps
    for k in range(6):
        u = (t * 0.9 + k / 6) % 1.0
        x = 420 + u * 440
        d.ellipse([x - 9, 236, x + 9, 254], fill=ORANGE)
    text(d, 640, 290, "lệnh nhỏ, gửi từng cái một", F_S, ORANGE, anchor="mm")
    gpu_timeline(d, 110, 380, 1060, 0.18, t)
    a = ease(seg(p, 0.35, 0.55))
    ctext(d, W / 2, 500, "Mỗi mẩu 80 ms cần hàng trăm lệnh nhỏ  →  ≈ 55–65 ms / mẩu", F_H, mix(RED, a))
    ctext(d, W / 2, 540, "(số đo của tác giả trên RTX 3060)", F_S, mix(MUTED, a))


def s_graph(d, p, t):
    # phase A: record once
    a = seg(p, 0.0, 0.33)
    rbox(d, 110, 150, 1060, 190, (255, 255, 255), outline=LINE, r=18)
    text(d, 130, 165, "1. Ghi lại MỘT lần", F_H, INK)
    n = 10
    for i in range(n):
        k = seg(a, i / n * 0.8, i / n * 0.8 + 0.25)
        x = 140 + i * 58 + k * (760 - 140 - i * 58)
        y = 230 + k * 10
        if k < 1:
            rbox(d, x, y, 48, 40, tint(ORANGE, 0.2), outline=ORANGE, r=8)
            ctext(d, x + 24, y + 20, f"{i + 1}", F_XS, ORANGE)
    rbox(d, 760, 205, 380, 110, tint(RED, 0.08), outline=RED, r=16)
    d.ellipse([782, 248, 802, 268], fill=RED if int(t * 2) % 2 == 0 or a >= 1 else BG)
    text(d, 815, 242, "CUDA graph", F_H, RED)
    text(d, 815, 280, "chuỗi lệnh của 1 mẩu", F_S, MUTED)

    # phase B: replay
    b = seg(p, 0.33, 0.62)
    if b > 0:
        rbox(d, 110, 352, 1060, 130, (255, 255, 255), outline=LINE, r=18)
        text(d, 130, 365, "2. Mỗi mẩu sau đó: chỉ 1 lệnh “phát lại”", F_H, mix(INK, b * 3))
        u = (t * 1.2) % 1.0
        x = 160 + u * 380
        rbox(d, x, 412, 64, 44, GREEN, r=10)
        play_icon(d, x + 32, 434, 22, (255, 255, 255))
        gpu_timeline(d, 580, 417, 560, 0.85, t, label=False)
        text(d, 580, 457, "GPU gần như luôn bận → không phí thời gian chờ", F_XS, MUTED)

    c = seg(p, 0.62, 0.8)
    if c > 0:
        y = 498
        text(d, 130, y, "Trước", F_B, mix(INK, c * 2))
        rbox(d, 250, y - 2, 720 * ease(c), 30, RED, r=8)
        text(d, 250 + 720 * ease(c) + 12, y, "55–65 ms", F_B, mix(RED, c * 2))
        text(d, 130, y + 38, "Sau", F_B, mix(INK, c * 2))
        rbox(d, 250, y + 36, 110 * ease(c), 30, GREEN, r=8)
        text(d, 250 + 110 * ease(c) + 12, y + 38, "7–10 ms  (theo tác giả)", F_B, mix(GREEN, c * 2))


def seat_grid(d, p, t, x0=110, y0=160):
    rng = random.Random(3)
    sw, sh, g = 150, 74, 16
    tick = (t * 1.6) % 1.0
    swap = p > 0.55
    for i in range(16):
        r, c = divmod(i, 4)
        x, y = x0 + c * (sw + g), y0 + r * (sh + g)
        empty = i in (11, 14) or (i == 5 and 0.4 < p <= 0.55)
        if empty:
            rbox(d, x, y, sw, sh, BG, outline=LINE, r=12)
            ctext(d, x + sw / 2, y + sh / 2, "trống", F_S, MUTED)
        else:
            col = USERS[(i + (3 if (i == 5 and swap) else 0)) % len(USERS)]
            rbox(d, x, y, sw, sh, tint(col, 0.16), outline=col, r=12)
            who = "người mới" if (i == 5 and swap) else f"người {i + 1}"
            ctext(d, x + sw / 2, y + sh / 2, who, F_S, col)
    sweep = x0 + tick * (4 * sw + 3 * g)
    d.line([sweep, y0 - 10, sweep, y0 + 4 * (sh + g) - 6], fill=GREEN, width=4)
    return x0 + 4 * (sw + g) + 20


def s_slots(d, p, t):
    xr = seat_grid(d, p, t)
    text(d, xr + 10, 160, "16 ghế (slot)", F_H)
    lines = ["Vạch xanh = 1 lần “phát lại”:", "tạo 1 mẩu cho TẤT CẢ các ghế", "cùng một lúc.", "",
             "Ai xong thì rời ghế,", "người đang chờ vào ngay.", "", "Ghế trống vẫn chạy cùng,",
             "kết quả bị bỏ đi."]
    for i, ln in enumerate(lines):
        text(d, xr + 10, 210 + i * 34, ln, F_B, INK if i < 3 else MUTED)
    # waiting queue
    if p < 0.55:
        u = seg(p, 0.42, 0.55)
        x = 1180 - u * 560
        d.ellipse([x - 20, 520, x + 20, 560], fill=USERS[0])
        text(d, x, 575, "đang chờ", F_XS, MUTED, anchor="ma")


def s_codec(d, p, t):
    text(d, 110, 150, "Mẩu số mới tạo", F_B, MUTED)
    n_in = int(seg(p, 0.05, 0.45) * 4 + 1e-6)
    for k in range(4):
        x = 110 + k * 70
        rbox(d, x, 190, 58, 58, tint(BLUE, 0.15) if k < n_in else BG, outline=BLUE if k < n_in else LINE, r=10)
    text(d, 110, 262, "gom đủ 4 mẩu = 320 ms âm thanh", F_S, INK)
    m = ease(seg(p, 0.45, 0.6))
    arrow(d, 420, 220, 420 + 130 * m, 220, mix(INK, m), w=5)
    rbox(d, 580, 160, 230, 120, tint(ORANGE, 0.14), outline=ORANGE, r=18)
    ctext(d, 695, 205, "Codec", F_H, ORANGE)
    ctext(d, 695, 245, "số → âm thanh", F_S, MUTED)
    o = ease(seg(p, 0.6, 0.8))
    if o > 0:
        arrow(d, 820, 220, 820 + 110 * o, 220, mix(INK, o), w=5)
        wave(d, 950, 220, 140 * o, 70, GREEN, phase=t * 3)
        listener(d, 1150, 225, mix(INK, o))
    q = ease(seg(p, 0.15, 0.35))
    rbox(d, 110, 330, 1060, 92, (255, 255, 255), outline=mix(LINE, q), r=16)
    text(d, 134, 346, "Vì sao không gọi codec sau từng mẩu?", F_B, mix(INK, q))
    text(d, 134, 382, "Mỗi lần gọi tốn cố định ≈ 65 ms + 2,5 ms × số ghế (theo tác giả), dù ít hay nhiều mẩu.",
         F_S, mix(MUTED, q))
    r = ease(seg(p, 0.8, 0.95))
    rbox(d, 110, 440, 1060, 92, tint(GREEN, 0.08 * r + 0.001), outline=mix(GREEN, r), r=16)
    text(d, 134, 456, "Người mới vào được ưu tiên:", F_B, mix(INK, r))
    text(d, 134, 492, "gọi codec sớm sau 2 mẩu (160 ms âm thanh) để nghe được tiếng đầu nhanh hơn.", F_S, mix(MUTED, r))


def s_summary(d, p, t):
    steps = [("Chữ → cách đọc (phoneme)", INK),
             ("Tạo từng mẩu 80 ms, mẩu sau dựa mẩu trước (AR)", BLUE),
             ("CUDA graph: 1 lệnh “phát lại” thay cho hàng trăm lệnh nhỏ", RED),
             ("16 ghế dùng chung, ai xong nhường chỗ (continuous batching)", GREEN),
             ("Codec đổi số thành âm thanh, gửi ngay cho người nghe", ORANGE)]
    for i, (s, c) in enumerate(steps):
        a = ease(seg(p, i * 0.15, i * 0.15 + 0.15))
        y = 160 + i * 74
        d.ellipse([110, y, 160, y + 50], fill=mix(c, a))
        ctext(d, 135, y + 25, str(i + 1), F_H, mix((255, 255, 255), a))
        text(d, 185, y + 10, s, F_H, mix(INK, a))


SCENES = [
    ("", "", s_title, [
        "Video này giải thích cách VieNeu biến chữ thành giọng nói và phát ngay cho người nghe.",
    ]),
    ("Mục tiêu", "Phát tiếng ngay, không đợi đọc xong", s_stream, [
        "Thay vì đợi tạo xong cả đoạn, VieNeu tạo giọng nói từng mẩu nhỏ, mẩu nào xong là gửi ngay.",
    ]),
    ("Bước 1", "Đổi chữ thành cách đọc", s_phoneme, [
        "Bước một: chữ viết được đổi thành cách phát âm, gọi là phoneme.",
    ]),
    ("Bước 2", "Tạo âm thanh từng mẩu (AR)", s_ar, [
        "Bước hai: model tạo âm thanh từng mẩu 80 mili giây. Mẩu sau dựa trên các mẩu trước, gọi là tự hồi quy, hay AR.",
        "Mỗi mẩu là 16 con số, điền lần lượt từng số. Đây là bản nén của âm thanh, chưa phải tiếng.",
    ]),
    ("Vấn đề", "GPU nhanh nhưng phải chờ lệnh", s_problem, [
        "Mỗi mẩu cần hàng trăm phép tính nhỏ. Python phải ra lệnh cho GPU từng phép một, nên GPU chủ yếu đứng chờ.",
        "Vì vậy, mỗi mẩu mất khoảng 55 đến 65 mili giây.",
    ]),
    ("Giải pháp 1", "CUDA graph: ghi lại một lần, phát lại nhiều lần", s_graph, [
        "CUDA graph ghi lại toàn bộ chuỗi lệnh của một mẩu, chỉ một lần. Sau đó, mỗi mẩu chỉ cần một lệnh phát lại.",
        "Theo tác giả, mỗi mẩu còn khoảng 7 đến 10 mili giây. Đổi lại, số ghế và bộ nhớ phải cố định.",
    ]),
    ("Giải pháp 2", "Nhiều người dùng chung một lần chạy", s_slots, [
        "Server có 16 ghế. Mỗi lần phát lại tạo một mẩu cho cả 16 người cùng lúc.",
        "Ai xong thì nhường ghế, người đang chờ vào ngay.",
    ]),
    ("Bước 3", "Codec: đổi số thành âm thanh", s_codec, [
        "Bước ba: codec đổi các con số thành âm thanh. Mỗi lần gọi codec khá tốn, nên server gom 4 mẩu rồi mới gọi.",
        "Người mới vào được gọi sớm sau 2 mẩu, để nghe được tiếng đầu nhanh hơn.",
    ]),
    ("Tóm tắt", "Năm bước", s_summary, [
        "Tóm lại: chữ thành cách đọc, tạo từng mẩu bằng CUDA graph, nhiều người dùng chung, "
        "rồi codec đổi thành âm thanh và gửi đi ngay.",
    ]),
]


def spoken(s):
    for a, b in SPOKEN.items():
        s = s.replace(a, b)
    return s


def voice_path(s, voice):
    key = hashlib.sha1(json.dumps([spoken(s), voice]).encode()).hexdigest()[:16]
    return VOICE_DIR / f"{key}.wav"


def fetch_voice(base, s, voice):
    """POST one sentence to the lite server, save it as a 48 kHz mono WAV."""
    path = voice_path(s, voice)
    if path.exists():
        return
    body = json.dumps({"input": spoken(s), "voice": voice, "response_format": "pcm"}).encode()
    req = urllib.request.Request(f"{base}/v1/audio/speech", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        pcm = r.read()
    VOICE_DIR.mkdir(exist_ok=True)
    with wavfile.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def load_voice(s, voice):
    path = voice_path(s, voice)
    if not path.exists():
        return None
    with wavfile.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def cue_seconds(s):
    return max(3.0, len(s) / 15.0 + 1.0)       # ~15 ký tự / giây


def build_timeline(voice=None):
    """Scenes with absolute cue times; a cue lasts as long as its voice-over (if cached)."""
    tl, t0 = [], 0.0
    for label, title, fn, cues in SCENES:
        times, t = [], t0
        for c in cues:
            a = load_voice(c, voice)
            dur = len(a) / SR + PAUSE if a is not None else cue_seconds(c)
            times.append((t, t + dur, c))
            t += dur
        t += SCENE_PAUSE
        tl.append((t0, t, label, title, fn, times))
        t0 = t
    return tl, t0


def srt_time(x):
    ms = int(round(x * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_frame(tl, now):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for t0, t1, label, title, fn, cues in tl:
        if t0 <= now < t1 or (now >= t1 and t1 == tl[-1][1]):
            p = clamp((now - t0) / (t1 - t0))
            if title:
                x = 60
                if label:
                    x += chip(d, 60, 44, label, BLUE, F_S) + 16
                text(d, x, 46, title, F_TITLE)
            fn(d, p, now - t0)
            cue = next((c for a, b, c in cues if a <= now < b), cues[-1][2])
            lines = wrap(cue, F_SUB, W - 200, d)
            bh = 22 + 38 * len(lines)
            rbox(d, 70, H - bh - 20, W - 140, bh, (24, 27, 31), r=14)
            for i, ln in enumerate(lines):
                d.text((W / 2, H - bh - 20 + 14 + i * 38), ln, font=F_SUB, fill=(255, 255, 255), anchor="ma")
            break
    return img


def voice_track(tl, total, voice):
    """The whole voice-over as one WAV file, or None if any sentence is missing."""
    track = np.zeros(int(total * SR) + SR, dtype=np.int16)
    for *_, cues in tl:
        for a, _, c in cues:
            v = load_voice(c, voice)
            if v is None:
                return None
            i = int(a * SR)
            track[i:i + len(v)] = v
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wavfile.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track[:int(total * SR)].tobytes())
    return f.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", metavar="URL", help="lite server to read the subtitles aloud")
    ap.add_argument("--speaker", default=None, help="preset voice name (default: server default)")
    args = ap.parse_args()
    if args.voice:
        for *_, cues in SCENES:
            for c in cues:
                fetch_voice(args.voice.rstrip("/"), c, args.speaker)

    tl, total = build_timeline(args.speaker)
    audio = voice_track(tl, total, args.speaker)
    with open(OUT / "how-it-works.vi.srt", "w", encoding="utf-8") as f:
        n = 0
        for _, _, _, _, _, cues in tl:
            for a, b, c in cues:
                n += 1
                f.write(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{c}\n\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-tune", "animation",
            "-movflags", "+faststart", str(OUT / "how-it-works.mp4")]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames = int(total * FPS)
    for i in range(frames):
        ff.stdin.write(render_frame(tl, i / FPS).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    if audio:
        os.unlink(audio)
    # poster: the CUDA graph scene, mid-way
    t0, t1 = next((a, b) for a, b, lb, *_ in tl if lb == "Giải pháp 1")
    render_frame(tl, t0 + 0.72 * (t1 - t0)).save(OUT / "how-it-works.png", optimize=True)
    print(f"{total:.0f}s, {frames} frames, voice-over: {'yes' if audio else 'no'} -> {OUT / 'how-it-works.mp4'}")


if __name__ == "__main__":
    main()
