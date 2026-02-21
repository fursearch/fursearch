"""Image annotation utilities for identification results."""

import os
import subprocess
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

# Distinct colors for bounding boxes
BOX_COLORS = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#FFFF00",  # Yellow
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#FFA500",  # Orange
    "#800080",  # Purple
    "#00FF7F",  # Spring Green
    "#FF69B4",  # Hot Pink
]

# Font paths to try
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_fonts(label_size: int = 32, watermark_size: int = 20):
    """Load fonts for annotation, with fallback to default.

    Args:
        label_size: Font size for labels
        watermark_size: Font size for watermark

    Returns:
        Tuple of (label_font, watermark_font)
    """
    font = None
    watermark_font = None
    for font_path in FONT_PATHS:
        try:
            font = ImageFont.truetype(font_path, label_size)
            watermark_font = ImageFont.truetype(font_path, watermark_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
        watermark_font = font
    return font, watermark_font


def annotate_image(
    image: Image.Image,
    results: List,
    min_confidence: float,
    watermark_text: Optional[str] = None
) -> Image.Image:
    """Draw bounding boxes, character names, and optional watermark on the image.

    Args:
        image: PIL Image to annotate
        results: List of SegmentResults from identification
        min_confidence: Minimum confidence threshold for displaying matches
        watermark_text: Custom watermark text. If None, does not show watermark.

    Returns:
        Annotated PIL Image
    """
    img_rgb = image.convert("RGB")
    font, watermark_font = _load_fonts()

    # Determine watermark text and bar height
    if watermark_text is not None:
        temp_draw = ImageDraw.Draw(img_rgb)
        wm_bbox = temp_draw.textbbox((0, 0), watermark_text, font=watermark_font)
        wm_text_h = wm_bbox[3] - wm_bbox[1]
        bar_height = wm_text_h + 16  # padding
    else:
        bar_height = 0
        watermark_text = ""
        wm_text_h = 0

    # Create new image with optional watermark bar at bottom
    annotated = Image.new("RGB", (img_rgb.width, img_rgb.height + bar_height), "black")
    annotated.paste(img_rgb, (0, 0))
    draw = ImageDraw.Draw(annotated)

    # Draw bounding boxes and labels for each segment
    multi_segment = len(results) > 1
    for i, result in enumerate(results):
        x, y, w, h = result.segment_bbox
        color = BOX_COLORS[i % len(BOX_COLORS)]

        # Filter matches by confidence
        filtered_matches = [m for m in result.matches if m.confidence >= min_confidence]

        # Get top character name, prefix with segment number if multiple characters
        if filtered_matches:
            name = filtered_matches[0].character_name or "Unknown"
            conf = filtered_matches[0].confidence
            if multi_segment:
                label = f"{i + 1}: {name} ({conf:.0%})"
            else:
                label = f"{name} ({conf:.0%})"
        else:
            label = "No match"

        # Draw rectangle with distinct color
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)

        # Draw label background and text above the box
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y - text_h - 8)
        draw.rectangle([x, label_y, x + text_w + 12, label_y + text_h + 6], fill=color)
        draw.text((x + 6, label_y + 3), label, fill="white", font=font)

    # Draw watermark text centered in the black bar
    if watermark_text is not None:
        wm_bbox = draw.textbbox((0, 0), watermark_text, font=watermark_font)
        wm_text_w = wm_bbox[2] - wm_bbox[0]
        wm_x = (annotated.width - wm_text_w) // 2
        wm_y = img_rgb.height + (bar_height - wm_text_h) // 2
        draw.text((wm_x, wm_y), watermark_text, fill="white", font=watermark_font)

    return annotated
