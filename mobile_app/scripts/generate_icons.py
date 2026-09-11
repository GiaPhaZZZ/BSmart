import os
import subprocess
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ICON_SVG = os.path.join(REPO_ROOT, "assets", "icons", "icon.svg")
if not os.path.exists(ICON_SVG):
    ICON_SVG = os.path.join(REPO_ROOT, "icon.svg")
RES_DIR = os.path.join(REPO_ROOT, "mobile_app", "android", "app", "src", "main", "res")

MIPMAPS = {
    "mipmap-mdpi": 48,
    "mipmap-hdpi": 72,
    "mipmap-xhdpi": 96,
    "mipmap-xxhdpi": 144,
    "mipmap-xxxhdpi": 192,
}

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def main():
    if not os.path.exists(ICON_SVG):
        print(f"Error: {ICON_SVG} does not exist.")
        return 1

    with open(ICON_SVG, "r", encoding="utf-8") as f:
        svg_content = f.read()

    # Create round SVG variant by replacing outer rect with circle
    round_svg_content = svg_content.replace(
        '<rect x="48" y="48" width="928" height="928" rx="250" fill="#091426"/>',
        '<circle cx="512" cy="512" r="464" fill="#091426"/>'
    )
    temp_round_svg = os.path.join(REPO_ROOT, "temp_icon_round.svg")
    with open(temp_round_svg, "w", encoding="utf-8") as f:
        f.write(round_svg_content)

    temp_square_png = os.path.join(REPO_ROOT, "temp_square_1024.png")
    temp_round_png = os.path.join(REPO_ROOT, "temp_round_1024.png")

    print("Rendering SVG to 1024x1024 PNG via Edge headless...")
    for svg_path, out_png in [(ICON_SVG, temp_square_png), (temp_round_svg, temp_round_png)]:
        cmd = [
            EDGE_PATH,
            "--headless=old",
            "--disable-gpu",
            "--default-background-color=00000000",
            "--hide-scrollbars",
            f"--screenshot={out_png}",
            "--window-size=1024,1024",
            svg_path
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0 or not os.path.exists(out_png):
            print(f"Failed to render {svg_path}: {res.stderr.decode('utf-8', errors='ignore')}")
            return 1

    img_square = Image.open(temp_square_png).convert("RGBA")
    img_round = Image.open(temp_round_png).convert("RGBA")

    print("Generating launcher icons for all Android densities...")
    for folder, size in MIPMAPS.items():
        folder_path = os.path.join(RES_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)

        # Square icon
        sq_resized = img_square.resize((size, size), Image.Resampling.LANCZOS)
        sq_path = os.path.join(folder_path, "ic_launcher.png")
        sq_resized.save(sq_path, "PNG", optimize=True)

        # Round icon
        rd_resized = img_round.resize((size, size), Image.Resampling.LANCZOS)
        rd_path = os.path.join(folder_path, "ic_launcher_round.png")
        rd_resized.save(rd_path, "PNG", optimize=True)

        print(f"  -> Generated {folder}: {size}x{size} (square & round)")

    # Clean up temporary files
    for p in [temp_round_svg, temp_square_png, temp_round_png]:
        if os.path.exists(p):
            os.remove(p)

    print("Done! All app icons successfully generated.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
