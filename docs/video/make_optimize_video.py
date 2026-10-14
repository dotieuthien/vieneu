"""Render docs/optimize.mp4 (+ .srt, poster): how the lite server was optimised, and why.

    python docs/video/make_optimize_video.py --voice http://127.0.0.1:PORT   # đọc lời bằng server lite (bản tối ưu)
    python docs/video/make_optimize_video.py                                 # dùng giọng đã cache

Same style and voice cache as make_video.py (docs/video/.voice/). Every number on screen comes
from README "Tối ưu codec (14/10/2026)" and the result files it links.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

import make_video as mv
from make_video import (BG, BLUE, F_B, F_H, F_S, F_SUB, F_TITLE, F_XS, GREEN, INK, LINE, MUTED, ORANGE,
                        PURPLE, RED, SR, TEAL, W, H, FPS, arrow, chip, ctext, ease, font, mix, play_icon, rbox, seg,
                        text, tint, wave, wrap)

OUT = Path(__file__).resolve().parents[1]
NAME = "optimize"
# Cách viết để TTS đọc đúng; phụ đề giữ chữ gốc.
mv.SPOKEN.update({"key value": "ki va liu", "megabyte": "mê ga bai", "Profiler": "Bộ đo hiệu năng",
                  "codec": "cô đéc", "model": "mô hình", "code cũ": "mã cũ"})
F_MONO = font(20, uni=True)
F_BIG = font(56, True)


def appear(p, a, b=None):
    return ease(seg(p, a, a + 0.12 if b is None else b))


def bullet_lines(d, x, y, lines, alphas, f=F_B, color=INK, gap=44, dot=BLUE):
    for i, (ln, a) in enumerate(zip(lines, alphas)):
        if a <= 0:
            continue
        cy = y + i * gap
        d.ellipse([x, cy + 9, x + 12, cy + 21], fill=mix(dot, a))
        text(d, x + 26, cy, ln, f, mix(color, a))


def hbar(d, x, y, length, h, color, a, label=None, value=None, f=F_B):
    if a <= 0:
        return
    rbox(d, x, y, max(6, length * ease(a)), h, mix(color, a), r=8)
    if value:
        text(d, x + length * ease(a) + 12, y + h / 2, value, f, mix(color, a), anchor="lm")


# ── scenes: fn(d, p, t, ci) — p = 0..1 through the scene, ci = index of the current cue ──────────

def s_title(d, p, t, ci):
    ctext(d, W / 2, 210, "Tối ưu VieNeu lite", font(56, True))
    ctext(d, W / 2, 290, "Vì sao biết phải sửa ở đâu, và tham số chọn theo logic nào", F_H, MUTED)
    ctext(d, W / 2, 335, "Lồng tiếng bằng chính bản tối ưu", F_B, BLUE)
    for i, c in enumerate([BLUE, ORANGE, GREEN]):
        wave(d, 290, 450, 700, 70, mix(c, 0.85), phase=t * 2 + i * 2.1, n=80)


def s_principle(d, p, t, ci):
    keep = ["Trọng số model", "Kiến trúc mạng", "Kiểu số fp32 của codec", "Cách chọn mã (sampling)"]
    change = ["Gom việc cho GPU", "Bỏ những lần chờ thừa", "Bỏ những lần copy thừa", "Ghi lại bằng CUDA graph"]
    a, b = appear(p, 0.02, 0.2), appear(p, 0.45, 0.62)
    rbox(d, 100, 140, 500, 330, mix(tint(GREEN, 0.10), a), outline=mix(GREEN, a), r=18)
    text(d, 130, 160, "Giữ nguyên", F_H, mix(GREEN, a))
    bullet_lines(d, 130, 220, keep, [appear(p, 0.08 + 0.06 * i) for i in range(4)], dot=GREEN)
    rbox(d, 680, 140, 500, 330, mix(tint(BLUE, 0.10), b), outline=mix(BLUE, b), r=18)
    text(d, 710, 160, "Chỉ đổi", F_H, mix(BLUE, b))
    bullet_lines(d, 710, 220, change, [appear(p, 0.5 + 0.06 * i) for i in range(4)], dot=BLUE)
    c = appear(p, 0.8)
    ctext(d, W / 2, 515, "Âm thanh phải ra y hệt: kiểm tra trước, rồi mới đo tốc độ", F_H, mix(RED, c))


def s_measure(d, p, t, ci):
    x0, scale = 400, 760 / 320            # px per ms; 320 ms of audio = the real-time budget
    text(d, 100, 140, "Mỗi 4 mẩu = 320 ms âm thanh (16 ghế, 1 người nghe, bản cũ)", F_B, MUTED)
    d.line([x0, 185, x0, 470], fill=LINE, width=2)
    bx = x0 + 320 * scale
    d.line([bx, 185, bx, 470], fill=mix(RED, appear(p, 0.1)), width=3)
    text(d, bx, 474, "320 ms = vừa kịp real time", F_XS, mix(RED, appear(p, 0.1)), anchor="ma")
    rows = [("Tạo 4 mẩu (CUDA graph)", 40, BLUE, "≈ 40 ms GPU", 0.18),
            ("Codec: CPU ra lệnh", 182, ORANGE, "182 ms", 0.42),
            ("Codec: GPU thật sự tính", 54, GREEN, "≈ 54 ms", 0.55)]
    for i, (lab, ms, col, val, at) in enumerate(rows):
        a = appear(p, at, at + 0.15)
        y = 205 + i * 85
        text(d, 100, y + 8, lab, F_B, mix(INK, a))
        hbar(d, x0, y, ms * scale, 46, col, a, value=val)
    c = appear(p, 0.78)
    ctext(d, W / 2, 530, "Nút thắt là codec, và codec đang bị CPU giữ chân (~2000 lệnh nhỏ mỗi lần gọi)", F_H, mix(RED, c))


def s_ttfa(d, p, t, ci):
    x0, scale = 100, 1080 / 170
    text(d, x0, 150, "Thời gian ra tiếng đầu (phía server, 1 người): 167 ms", F_H)
    parts = [("nạp câu (prefill)", 47, PURPLE), ("2 mẩu đầu", 19, BLUE), ("codec lần đầu", 103, ORANGE)]
    x = x0
    for i, (lab, ms, col) in enumerate(parts):
        a = appear(p, 0.08 + 0.18 * i, 0.22 + 0.18 * i)
        w = ms * scale * a
        rbox(d, x, 230, max(4, w - 4), 80, mix(col, a), r=10)
        if a > 0.5:
            ctext(d, x + w / 2, 270, f"{ms} ms", F_H, (255, 255, 255))
            ctext(d, x + ms * scale / 2, 340, lab, F_B, mix(col, a))
        x += ms * scale
    c = appear(p, 0.7)
    ctext(d, W / 2, 440, "Một nửa chỉ để chờ codec giải mã lần đầu", F_H, mix(RED, c))


def s_code(d, p, t, ci):
    # left: one frame at a time, 12 layers each, then a host sync per seat
    text(d, 100, 140, "Code cũ: giải mã từng mẩu một", F_H)
    for k in range(4):
        a = appear(p, 0.05 + 0.08 * k)
        x = 100 + k * 135
        rbox(d, x, 200, 110, 150, mix(tint(ORANGE, 0.14), a), outline=mix(ORANGE, a), r=12)
        ctext(d, x + 55, 222, f"mẩu {k + 1}", F_S, mix(ORANGE, a))
        for layer in range(12):
            d.line([x + 14, 245 + layer * 8, x + 96, 245 + layer * 8], fill=mix(ORANGE, a * 0.5), width=3)
        s = appear(p, 0.36 + 0.03 * k) if ci >= 1 else 0
        for j in range(16):
            cx = x + 8 + (j % 8) * 13
            cy = 370 + (j // 8) * 16
            d.ellipse([cx, cy, cx + 9, cy + 9], fill=mix(RED, s))
    if ci >= 1:
        text(d, 100, 415, "chấm đỏ = hỏi độ dài từng ghế: CPU đứng chờ GPU", F_S, RED)
    # right: the KV cache copied around every forward
    c = appear(p, 0.7) if ci >= 2 else 0
    rbox(d, 700, 200, 480, 220, mix((255, 255, 255), c), outline=mix(PURPLE, c), r=16)
    ctext(d, 940, 232, "Bộ nhớ key value ≈ 400 MB", F_H, mix(PURPLE, c))
    for i, s in enumerate(["nối", "cắt", "chọn", "chép lại", "…"]):
        x = 720 + i * 92
        rbox(d, x, 290, 80, 50, mix(tint(PURPLE, 0.16), c), outline=mix(PURPLE, c), r=10)
        ctext(d, x + 40, 315, s, F_S, mix(PURPLE, c))
    ctext(d, 940, 385, "mỗi lần tính, chép cả khối ~5 lượt", F_S, mix(MUTED, c))


def s_group(d, p, t, ci):
    a = appear(p, 0.02)
    text(d, 100, 140, "Trước: 4 lượt tính", F_H, mix(INK, a))
    for k in range(4):
        x = 100 + k * 150
        rbox(d, x, 190, 120, 70, mix(tint(ORANGE, 0.15), a), outline=mix(ORANGE, a), r=12)
        ctext(d, x + 60, 225, f"mẩu {k + 1}", F_S, mix(ORANGE, a))
    b = appear(p, 0.3)
    text(d, 100, 300, "Sau: 1 lượt tính cho cả 4 mẩu", F_H, mix(INK, b))
    rbox(d, 100, 350, 570, 80, mix(tint(GREEN, 0.15), b), outline=mix(GREEN, b), r=14)
    for k in range(4):
        ctext(d, 170 + k * 140, 390, f"mẩu {k + 1}", F_S, mix(GREEN, b))
    c = appear(p, 0.55) if ci >= 1 else 0
    rbox(d, 740, 170, 440, 270, mix((255, 255, 255), c), outline=mix(LINE, c), r=16)
    text(d, 765, 190, "Vì sao vẫn đúng?", F_H, mix(INK, c))
    bullet_lines(d, 765, 245, ["Chỉ nhìn về quá khứ", "Cửa sổ nhìn lại có hạn", "Ghế khác: giữ nguyên trạng thái"],
                 [c, appear(p, 0.65) if ci >= 1 else 0, appear(p, 0.75) if ci >= 1 else 0], f=F_B, dot=GREEN)
    text(d, 765, 395, "→ cùng kết quả như tính từng mẩu", F_S, mix(GREEN, c))


def s_graph(d, p, t, ci):
    a = appear(p, 0.02)
    text(d, 100, 140, "Mỗi lần gọi codec bây giờ:", F_H, mix(INK, a))
    x = 100
    for i, s in enumerate(["chép mã", "chép độ dài", "chép mặt nạ ghế"]):
        x += chip(d, x, 195, s, mix(BLUE, appear(p, 0.06 + 0.06 * i)), F_S) + 12
    rbox(d, x + 10, 188, 64, 44, mix(GREEN, appear(p, 0.25)), r=10)
    play_icon(d, x + 42, 210, 22, (255, 255, 255))
    text(d, x + 90, 198, "1 lệnh phát lại", F_B, mix(GREEN, appear(p, 0.25)))
    text(d, 100, 255, "Graph cho mỗi độ dài 1, 2, 3, 4 mẩu · CPU chỉ còn ~0,6 ms mỗi lần gọi", F_S, mix(MUTED, a))
    if ci >= 1:
        b = appear(p, 0.5)
        rbox(d, 100, 300, 1080, 230, mix((255, 255, 255), b), outline=mix(LINE, b), r=16)
        text(d, 125, 318, "Vì sao bản graph của tác giả ra âm thanh sai?", F_H, mix(INK, b))
        k = 1 + int(seg(p, 0.55, 0.9) * 3.999)            # replays so far
        rbox(d, 470, 370, 200, 70, mix(tint(ORANGE, 0.18), b), outline=mix(ORANGE, b), r=10)
        ctext(d, 570, 393, "vùng nhớ output", F_XS, mix(MUTED, b))
        ctext(d, 570, 418, f"đang chứa mẩu {k}", F_S, mix(ORANGE, b))
        for j in range(k):
            yy = 360 + j * 24
            text(d, 140, yy, f"giữ mẩu {j + 1} →", F_S, mix(RED if j < k - 1 else INK, b))
            arrow(d, 300, yy + 12, 465, 400, mix(LINE, b), w=2, head=8)
        text(d, 140, 470, "mọi con trỏ trỏ vào cùng một chỗ: lần phát sau ghi đè lần trước", F_S, mix(RED, b))
        c = appear(p, 0.82)
        rbox(d, 780, 375, 370, 60, mix(tint(GREEN, 0.15), c), outline=mix(GREEN, c), r=10)
        ctext(d, 965, 405, "bản mới: chép kết quả ra trước", F_S, mix(GREEN, c))


def s_ring(d, p, t, ci):
    cx, cy, r, n = 900, 330, 170, 28
    head = int(t * 6) % n
    for i in range(n):
        a0 = 2 * math.pi * i / n - math.pi / 2
        a1 = 2 * math.pi * (i + 1) / n - math.pi / 2
        age = (head - i) % n
        col = PURPLE if age < 4 else (tint(PURPLE, 0.35) if age < n - 4 else LINE)
        d.pieslice([cx - r, cy - r, cx + r, cy + r], math.degrees(a0), math.degrees(a1) - 1.5, fill=col)
    d.ellipse([cx - r + 46, cy - r + 46, cx + r - 46, cy + r - 46], fill=BG)
    ctext(d, cx, cy - 12, "vòng", F_H, PURPLE)
    ctext(d, cx, cy + 22, "chỉ ghi vị trí mới", F_XS, MUTED)
    a = appear(p, 0.02)
    text(d, 100, 160, "Cách cũ", F_H, mix(RED, a))
    text(d, 100, 205, "nối + cắt + chọn + chép lại cả khối", F_B, mix(INK, a))
    text(d, 100, 245, "~5 lượt × ~400 MB mỗi lần tính", F_B, mix(MUTED, a))
    b = appear(p, 0.25)
    text(d, 100, 315, "Cách mới", F_H, mix(GREEN, b))
    text(d, 100, 360, "ghi đúng vài vị trí mới vào vòng", F_B, mix(INK, b))
    if ci >= 1:
        c = appear(p, 0.6)
        rbox(d, 100, 420, 600, 100, mix(tint(PURPLE, 0.10), c), outline=mix(PURPLE, c), r=14)
        text(d, 125, 435, "Độ dài vòng = cửa sổ nhìn lại + 4 mẩu", F_B, mix(PURPLE, c))
        text(d, 125, 475, "→ không ghi đè dữ liệu còn cần dùng", F_S, mix(MUTED, c))


def s_verify(d, p, t, ci):
    a = appear(p, 0.02)
    text(d, 100, 140, "Cùng một chuỗi mã → giải mã cách cũ và cách mới", F_H, mix(INK, a))
    for k, col in enumerate([BLUE, ORANGE]):
        wave(d, 120, 260, 1040, 120, mix(col, a * (0.9 if k == 0 else 0.7)), phase=1.3, n=160, width=4 - k * 2)
    text(d, 120, 330, "xanh = SDK gốc · cam = bản mới (trùng lên nhau)", F_S, mix(MUTED, a))
    if ci >= 1:
        checks = ["Dài 13 giây: vòng quay hết một lượt", "Người vào giữa chừng, người rời đi",
                  "Số mẩu mỗi lần lẫn lộn"]
        bullet_lines(d, 120, 385, checks, [appear(p, 0.5 + 0.08 * i) for i in range(3)], f=F_B, dot=GREEN)
        c = appear(p, 0.8)
        rbox(d, 760, 385, 400, 120, mix(tint(GREEN, 0.12), c), outline=mix(GREEN, c), r=16)
        ctext(d, 960, 425, "tương quan", F_B, mix(MUTED, c))
        ctext(d, 960, 470, "0,9999993", F_BIG, mix(GREEN, c))


def s_hang(d, p, t, ci):
    a = appear(p, 0.02)
    rbox(d, 100, 160, 300, 90, mix(tint(BLUE, 0.12), a), outline=mix(BLUE, a), r=14)
    ctext(d, 250, 205, "graph sinh âm", F_H, mix(BLUE, a))
    rbox(d, 100, 300, 300, 90, mix(tint(ORANGE, 0.12), a), outline=mix(ORANGE, a), r=14)
    ctext(d, 250, 345, "graph codec", F_H, mix(ORANGE, a))
    fixed = ci >= 1 and p > 0.62
    if not fixed:
        rbox(d, 560, 230, 330, 90, mix(tint(RED, 0.12), a), outline=mix(RED, a), r=14)
        ctext(d, 725, 262, "vùng nhớ tạm chung", F_B, mix(RED, a))
        ctext(d, 725, 295, "(thư viện nhân ma trận)", F_XS, mix(MUTED, a))
        arrow(d, 400, 205, 555, 255, mix(BLUE, a), w=4)
        arrow(d, 400, 345, 555, 300, mix(ORANGE, a), w=4)
        if int(t * 2) % 2 == 0:
            ctext(d, 1030, 275, "TREO", F_BIG, mix(RED, a))
        b = appear(p, 0.3)
        rbox(d, 100, 430, 1080, 100, mix((32, 35, 40), b), r=12)
        for i, ln in enumerate(["codec thread : ... audio.cpu()          ← đứng chờ GPU",
                                "worker thread: ... torch.cuda.Event.record ← đứng chờ GPU"]):
            text(d, 125, 448 + i * 34, ln, F_MONO, mix((230, 230, 230), b))
    else:
        c = ease(seg(p, 0.62, 0.75))
        for y, col in ((205, BLUE), (345, ORANGE)):
            rbox(d, 560, y - 40, 330, 80, mix(tint(col, 0.12), c), outline=mix(col, c), r=14)
            ctext(d, 725, y, "vùng nhớ tạm riêng", F_B, mix(col, c))
            arrow(d, 400, y, 555, y, mix(col, c), w=4)
        ctext(d, 1040, 275, "hết treo", F_H, mix(GREEN, c))
        text(d, 100, 450, "Sửa: mỗi graph codec được ghi trên stream riêng của nó", F_B, mix(INK, c))


PARAMS = [
    ("DECODE_FRAMES = 4", "mỗi lần tính đọc lại bộ nhớ mọi ghế → gom 4 mẩu cho rẻ; đoạn âm thanh vẫn chỉ 320 ms"),
    ("LEADIN_FRAMES = 2", "người mới ra tiếng sau 2 mẩu; lần sớm này chỉ tính cho người mới"),
    ("MAX_STREAMS = 16 / 32", "graph tính đủ mọi ghế, codec tốn theo số ghế → đặt bằng số người dự kiến"),
    ("nhận ≤ 16 người / lượt", "một đợt 16 người vào cùng lúc chỉ cần 1 lần nạp câu"),
    ("OVERLAP_CODEC = 1", "codec chạy song song: tổng thời gian tốt hơn, tiếng đầu chậm hơn khi dồn"),
]


PARAM_ROW = [0, 0, 1, 2, 3, 4]       # cue index -> highlighted row


def s_params(d, p, t, ci):
    ci = PARAM_ROW[ci]
    for i, (name, why) in enumerate(PARAMS):
        if i > ci:                                     # rows appear with their cue
            continue
        on = i == ci
        y = 140 + i * 82
        rbox(d, 90, y, 1100, 70, tint(BLUE, 0.10) if on else (255, 255, 255), outline=BLUE if on else LINE, r=14)
        text(d, 115, y + 22, name, F_B, BLUE if on else INK)
        lines = wrap(why, F_S, 700, d)[:2]
        for k, ln in enumerate(lines):
            text(d, 470, y + (22 if len(lines) == 1 else 9 + k * 27), ln, F_S, INK if on else MUTED)


def s_results(d, p, t, ci):
    if ci == 0:
        text(d, 100, 140, "RTF (thấp hơn = tốt hơn) · RTX 3060, cùng một máy, 16 ghế", F_B, MUTED)
        groups = [("1 người", 0.40, 0.21), ("16 người", 0.53, 0.31)]
        for g, (lab, old, new) in enumerate(groups):
            y = 200 + g * 160
            text(d, 100, y + 30, lab, F_H)
            a, b = appear(p, 0.1 + 0.25 * g), appear(p, 0.2 + 0.25 * g)
            hbar(d, 320, y, old * 1500, 44, MUTED, a, value=f"cũ {old:.2f}".replace(".", ","))
            hbar(d, 320, y + 56, new * 1500, 44, GREEN, b, value=f"mới {new:.2f}".replace(".", ","))
    else:
        text(d, 100, 140, "32 ghế, 32 người dùng liên tục trong 3 phút", F_H)
        stats = [("1824", "yêu cầu"), ("0", "lỗi"), ("305 ms", "tiếng đầu (trung vị)"), ("0,74", "RTF chậm nhất 1%")]
        for i, (v, lab) in enumerate(stats):
            a = appear(p, 0.55 + 0.07 * i)
            x = 100 + i * 275
            rbox(d, x, 220, 250, 190, mix((255, 255, 255), a), outline=mix(GREEN if i != 2 else BLUE, a), r=18)
            ctext(d, x + 125, 295, v, F_BIG, mix(GREEN if i != 2 else BLUE, a))
            ctext(d, x + 125, 365, lab, F_S, mix(MUTED, a))
        ctext(d, W / 2, 470, "RTF < 1: mọi người đều nghe theo thời gian thực", F_B, mix(INK, appear(p, 0.85)))


def s_quality(d, p, t, ci):
    if ci == 0:
        text(d, 100, 140, "5 câu mẫu · bản cũ và bản mới · cùng hạt giống ngẫu nhiên", F_B, MUTED)
        for k, col in enumerate([BLUE, ORANGE]):
            wave(d, 120, 280, 1040, 150, mix(col, 0.9 if k == 0 else 0.7), phase=0.4, n=180, width=4 - k * 2)
        a = appear(p, 0.3)
        rbox(d, 330, 400, 620, 110, mix(tint(GREEN, 0.12), a), outline=mix(GREEN, a), r=16)
        ctext(d, 640, 435, "khác biệt nhỏ hơn tín hiệu ~1600 lần (SNR ≈ 64 dB)", F_B, mix(GREEN, a))
        ctext(d, 640, 475, "mẫu nghe: docs/samples/", F_S, mix(MUTED, a))
    else:
        text(d, 100, 140, "Máy nhận dạng giọng nói (PhoWhisper) chấm lỗi ký tự · trung vị", F_H)
        rows = [("Chạy riêng", "4,4 %", MUTED), ("Khi 32 người dùng cùng lúc", "1,8 %", GREEN)]
        for i, (lab, v, col) in enumerate(rows):
            a = appear(p, 0.5 + 0.12 * i)
            y = 220 + i * 100
            text(d, 120, y + 20, lab, F_H, mix(INK, a))
            text(d, 900, y + 10, v, F_BIG, mix(col, a))
        ctext(d, W / 2, 470, "Giọng trong video này: chính bản tối ưu đọc", F_H, mix(BLUE, appear(p, 0.8)))


def s_summary(d, p, t, ci):
    steps = [("Đo trước khi sửa (profiler)", INK), ("Đọc mã chỗ tốn nhất", ORANGE),
             ("Sửa mà không đụng model", BLUE), ("Chứng minh âm thanh không đổi", GREEN),
             ("Rồi mới đo tốc độ và độ ổn định", PURPLE)]
    for i, (s, c) in enumerate(steps):
        a = appear(p, i * 0.14, i * 0.14 + 0.14)
        y = 150 + i * 74
        d.ellipse([110, y, 160, y + 50], fill=mix(c, a))
        ctext(d, 135, y + 25, str(i + 1), F_H, mix((255, 255, 255), a))
        text(d, 185, y + 10, s, F_H, mix(INK, a))


SCENES = [
    ("", "", s_title, [
        "Video này kể lại cách tối ưu bản lite của VieNeu: vì sao biết phải sửa ở đâu, và các tham số được chọn theo logic nào.",
    ]),
    ("Nguyên tắc", "Không đụng vào model", s_principle, [
        "Nguyên tắc đầu tiên: không đụng vào model. Trọng số, kiến trúc, kiểu số và cách chọn mã đều giữ nguyên.",
        "Chỉ đổi cách giao việc cho GPU: gom việc, bỏ những lần chờ thừa, bỏ những lần copy thừa. Âm thanh phải ra y hệt.",
    ]),
    ("Bước 1", "Đo trước khi sửa", s_measure, [
        "Bước một là đo. Profiler ghi thời gian của từng phần, cả phía CPU lẫn phía GPU.",
        "Cứ 4 mẩu là 320 mili giây âm thanh. Phần tạo mẩu bằng CUDA graph chỉ mất khoảng 40 mili giây.",
        "Nhưng mỗi lần gọi codec, CPU mất 182 mili giây để ra lệnh, trong khi GPU chỉ thật sự tính khoảng 54 mili giây.",
        "Kết luận: nút thắt không nằm ở model sinh âm, mà ở codec, và codec đang bị CPU giữ chân.",
    ]),
    ("Bước 1", "Tiếng đầu tiên mất bao lâu", s_ttfa, [
        "Profiler còn tách thời gian ra tiếng đầu tiên: 167 mili giây, trong đó riêng lần gọi codec đầu đã mất 103 mili giây.",
    ]),
    ("Bước 2", "Đọc mã nguồn chỗ tốn nhất", s_code, [
        "Bước hai là đọc mã nguồn của codec. Hóa ra nó giải mã từng mẩu một, mỗi mẩu chạy qua mười hai lớp tính.",
        "Sau mỗi mẩu, nó còn hỏi độ dài âm thanh của từng ghế, và mỗi lần hỏi lại bắt CPU đứng chờ GPU.",
        "Thêm nữa, mỗi lần tính, nó chép cả bộ nhớ key value, khoảng 400 megabyte, qua lại nhiều lượt.",
    ]),
    ("Ý tưởng 1", "Gom cả 4 mẩu vào một lần tính", s_group, [
        "Ý tưởng một: gom cả 4 mẩu vào chung một lần tính.",
        "Cách này vẫn đúng, vì codec chỉ nhìn về quá khứ trong một cửa sổ có hạn. Các ghế không liên quan được giữ nguyên trạng thái.",
    ]),
    ("Ý tưởng 2", "CUDA graph cho codec", s_graph, [
        "Ý tưởng hai: ghi lần tính của codec thành CUDA graph. Mỗi lần gọi chỉ còn vài lệnh chép và một lệnh phát lại.",
        "Tác giả từng thử graph cho codec nhưng ra âm thanh sai, vì code cũ giữ con trỏ vào vùng nhớ mà lần phát sau ghi đè. Bản mới chép kết quả ra trước.",
    ]),
    ("Ý tưởng 3", "Bộ nhớ key value dạng vòng", s_ring, [
        "Ý tưởng ba: thay việc chép cả bộ nhớ bằng một vòng tròn. Mỗi lần chỉ ghi vào chỗ của các vị trí mới.",
        "Vòng dài bằng cửa sổ nhìn lại cộng thêm 4 mẩu, nên không bao giờ ghi đè lên dữ liệu còn cần dùng.",
    ]),
    ("Kiểm tra", "Đúng trước, nhanh sau", s_verify, [
        "Trước khi đo tốc độ, phải chứng minh âm thanh không đổi: cùng một chuỗi mã được giải theo cách cũ và cách mới.",
        "Chuỗi dài 13 giây, có người vào giữa chừng, có người rời đi. Hai kết quả tương quan gần như tuyệt đối.",
    ]),
    ("Sự cố", "GPU bị treo khi hai graph chạy song song", s_hang, [
        "Lần chạy đầu, GPU bị treo. Bộ canh treo in ra chỗ kẹt: cả hai luồng đều đứng chờ GPU.",
        "Nhiều khả năng hai graph đã dùng chung một vùng nhớ tạm. Cho graph của codec ghi trên luồng riêng thì hết treo.",
    ]),
    ("Tham số", "Logic chọn tham số", s_params, [
        "Gọi codec mỗi 4 mẩu, vì mỗi lần tính phải đọc lại bộ nhớ của mọi ghế, nên gom lại thì rẻ hơn.",
        "Mà mỗi đoạn âm thanh gửi đi vẫn chỉ dài 320 mili giây.",
        "Người mới được giải mã sớm sau 2 mẩu để tiếng đầu ra nhanh. Lần sớm này chỉ tính cho người mới, người cũ đợi tới mốc 4 mẩu.",
        "Số ghế: graph luôn tính đủ mọi ghế, kể cả ghế trống, và codec càng nhiều ghế càng tốn. Nên đặt số ghế bằng số người dùng cùng lúc dự kiến.",
        "Mỗi lượt nhận tối đa 16 người, để một đợt 16 người vào cùng lúc chỉ cần một lần nạp câu.",
        "Codec chạy song song với việc sinh âm: tổng thời gian tốt hơn, nhưng khi nhiều người vào cùng lúc, tiếng đầu có thể chậm hơn.",
    ]),
    ("Kết quả", "Nhanh hơn bao nhiêu", s_results, [
        "Trên cùng một máy, với một người, hệ số thời gian thực giảm từ 0,40 xuống 0,21. Với 16 người, từ 0,53 xuống 0,31.",
        "Với 32 ghế và 32 người dùng liên tục trong 3 phút: 1824 yêu cầu, không lỗi nào, ai cũng nghe theo thời gian thực.",
    ]),
    ("Chất lượng", "Âm thanh có đổi không?", s_quality, [
        "Năm câu mẫu đọc bằng bản cũ và bản mới, cùng hạt giống ngẫu nhiên, ra âm thanh trùng nhau: phần khác biệt nhỏ hơn tín hiệu khoảng 1600 lần.",
        "Khi tải nặng, máy nhận dạng giọng nói vẫn nghe đúng như lúc chạy riêng. Giọng trong video này cũng do chính bản tối ưu đọc.",
    ]),
    ("Tóm tắt", "Quy trình", s_summary, [
        "Tóm lại: đo trước, đọc mã chỗ tốn nhất, sửa mà không đụng model, chứng minh âm thanh không đổi, rồi mới đo tốc độ và độ ổn định.",
    ]),
]


def build_timeline(voice):
    tl, t0 = [], 0.0
    for label, title, fn, cues in SCENES:
        times, t = [], t0
        for c in cues:
            a = mv.load_voice(c, voice)
            dur = len(a) / SR + mv.PAUSE if a is not None else mv.cue_seconds(c)
            times.append((t, t + dur, c))
            t += dur
        t += mv.SCENE_PAUSE
        tl.append((t0, t, label, title, fn, times))
        t0 = t
    return tl, t0


def render_frame(tl, now):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for t0, t1, label, title, fn, cues in tl:
        if t0 <= now < t1 or (now >= t1 and t1 == tl[-1][1]):
            p = min(1.0, max(0.0, (now - t0) / (t1 - t0)))
            ci = next((i for i, (a, b, _) in enumerate(cues) if a <= now < b), len(cues) - 1)
            if title:
                x = 60
                if label:
                    x += chip(d, 60, 44, label, BLUE, F_S) + 16
                text(d, x, 46, title, F_TITLE)
            fn(d, p, now - t0, ci)
            lines = wrap(cues[ci][2], F_SUB, W - 200, d)
            bh = 22 + 38 * len(lines)
            rbox(d, 70, H - bh - 20, W - 140, bh, (24, 27, 31), r=14)
            for i, ln in enumerate(lines):
                d.text((W / 2, H - bh - 20 + 14 + i * 38), ln, font=F_SUB, fill=(255, 255, 255), anchor="ma")
            break
    return img


def voice_track(tl, total, voice):
    import tempfile
    import wave as wavfile
    import numpy as np
    track = np.zeros(int(total * SR) + SR, dtype=np.int16)
    for *_, cues in tl:
        for a, _, c in cues:
            v = mv.load_voice(c, voice)
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
    ap.add_argument("--frames", default=None, help="only save PNG stills at these comma-separated seconds")
    args = ap.parse_args()
    if args.voice:
        for *_, cues in SCENES:
            for c in cues:
                mv.fetch_voice(args.voice.rstrip("/"), c, args.speaker)

    tl, total = build_timeline(args.speaker)
    if args.frames:
        for s in args.frames.split(","):
            still = mv.VOICE_DIR.parent / ".preview" / f"{NAME}-{float(s):05.1f}.png"
            still.parent.mkdir(exist_ok=True)
            render_frame(tl, float(s)).save(still)
        return
    audio = voice_track(tl, total, args.speaker)
    with open(OUT / f"{NAME}.vi.srt", "w", encoding="utf-8") as f:
        n = 0
        for *_, cues in tl:
            for a, b, c in cues:
                n += 1
                f.write(f"{n}\n{mv.srt_time(a)} --> {mv.srt_time(b)}\n{c}\n\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-tune", "animation",
            "-movflags", "+faststart", str(OUT / f"{NAME}.mp4")]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames = int(total * FPS)
    for i in range(frames):
        ff.stdin.write(render_frame(tl, i / FPS).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    if audio:
        os.unlink(audio)
    t0, t1 = next((a, b) for a, b, lb, *_ in tl if lb == "Bước 1")
    render_frame(tl, t0 + 0.9 * (t1 - t0)).save(OUT / f"{NAME}.png", optimize=True)
    print(f"{total:.0f}s, {frames} frames, voice-over: {'yes' if audio else 'no'} -> {OUT / f'{NAME}.mp4'}")


if __name__ == "__main__":
    main()
