"""Build a short status-update slide deck PDF for R.E.A.P."""
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = landscape(A4)  # 11.69 x 8.27
BLUE = (0.12, 0.44, 0.71)
DARK = (0.15, 0.15, 0.15)
GREY = (0.35, 0.35, 0.35)
LIGHT_BG = (0.96, 0.97, 0.99)

TITLE = "R.E.A.P : Realtime Edge Agentic Physio-Buddy"

slides = [
    ("title", TITLE, [
        "B.Tech (IT) Major Project — Status Update",
        "Shishir SH (231IT067)  •  Rahul S (231IT054)  •  Ashab M (231IT011)",
        "Guide: Prof. Ram Mohana Reddy Guddeti  •  NITK Surathkal",
    ]),
    ("Problem Statement", "Problem Statement", [
        "Form degrades long before fatigue is felt.",
        "Bad form mid-set → joint / tendon injury (older adults at highest risk).",
        "No real-time self-correction without a coach.",
        "Need: instant, eyes-free audio feedback every rep.",
    ]),
    ("Objectives", "Objectives", [
        "O1 — Instant feedback: sub-second audio cues mid-rep.",
        "O2 — Injury prevention: catch velocity-loss + deviation, stop before failure.",
        "O3 — Accessibility: phone-grade kinematic tracking at home.",
    ]),
    ("Pipeline", "System Architecture", [
        "Camera → Hybrid Pose (MediaPipe + YOLO-Pose)",
        "→ Predictive Kinematics (OneEuro now, N-euro later)",
        "→ Action Recognition (ST-GCN) → Fatigue Model → Local TTS Audio",
        "Constraint: 100% on-device, zero cloud, audio-first.",
    ]),
    ("Done", "Work Done (Sept milestone)", [
        "Project scaffold: uv + pyproject, configs, modular src/reap/.",
        "Hybrid pose core: MP every frame + YOLO fallback / validator, COCO-17 fusion.",
        "Benchmark app: live MP vs YOLO vs Hybrid + FPS / latency HUD + CSV log.",
        "Kinematics + smoothing: joint angles, OneEuro filter, N-euro stub.",
        "Verified: 6/6 tests pass; MP ~32 ms/frame; hybrid fallback works.",
    ]),
    ("Pending", "Pending / Next Work", [
        "Oct–Nov: core framework + exercise plug-in packs + edge tuning.",
        "Dec: local TTS + audio cue engine.",
        "Jan–Feb: predictive fatigue (velocity-loss) modeling.",
        "Mar: real-world testing with older adults.  Apr: N-euro / ST-GCN iteration.",
    ]),
    ("thanks", "Thank You", [
        "Demo: uv run reap-bench --source 0 --mode hybrid",
        "Questions welcome.",
    ]),
]


def draw_slide(c: Canvas, kind: str, heading: str, bullets: list[str], num: int, total: int):
    # background
    c.setFillColorRGB(*LIGHT_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # header bar
    c.setFillColorRGB(*BLUE)
    c.rect(0, PAGE_H - 1.15 * inch, PAGE_W, 1.15 * inch, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 0.78 * inch, heading)
    # body
    left = 0.9 * inch
    top = PAGE_H - 1.9 * inch
    c.setFillColorRGB(*DARK)
    if kind == "title":
        c.setFont("Helvetica", 15)
        y = top
        for line in bullets:
            c.setFillColorRGB(*GREY) if "Guide" in line or "•" in line else c.setFillColorRGB(*DARK)
            c.drawCentredString(PAGE_W / 2, y, line)
            y -= 0.42 * inch
    elif kind == "thanks":
        c.setFont("Helvetica", 16)
        y = top - 0.3 * inch
        for line in bullets:
            c.setFillColorRGB(*GREY)
            c.drawCentredString(PAGE_W / 2, y, line)
            y -= 0.5 * inch
    else:
        y = top
        for b in bullets:
            c.setFillColorRGB(*BLUE)
            c.circle(left + 6, y + 5, 5, fill=1, stroke=0)
            c.setFillColorRGB(*DARK)
            c.setFont("Helvetica", 15)
            # bold lead before — : render whole line, simple
            c.drawString(left + 0.32 * inch, y, b)
            y -= 0.52 * inch
    # footer
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(*GREY)
    c.drawString(0.6 * inch, 0.4 * inch, "R.E.A.P  |  NITK Surathkal")
    c.drawRightString(PAGE_W - 0.6 * inch, 0.4 * inch, f"{num}/{total}")


def main(out: str = "REAP_Status_Update.pdf"):
    c = Canvas(out, pagesize=landscape(A4))
    for i, (kind, head, bullets) in enumerate(slides, 1):
        draw_slide(c, kind, head, bullets, i, len(slides))
        c.showPage()
    c.save()
    print(f"wrote {out} ({len(slides)} slides)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "REAP_Status_Update.pdf")
