import sys
import os

from beadsketch.app import run


def smoke_test() -> None:
    """Packaged-runtime check: bundled assets, OpenCV and algorithm all load."""
    from PIL import Image, ImageDraw
    from beadsketch.engine import ConvertOptions, convert_image
    from beadsketch.palettes import load_palette

    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((14, 14, 82, 82), fill=(232, 154, 91), outline=(35, 37, 42), width=4)
    result = convert_image(image, load_palette("MARD 291"),
                           ConvertOptions(width=16, max_colors=6, profile="插画/动漫"))
    if result.indices.shape != (16, 16) or sum(n for _, n in result.counts()) != 256:
        raise RuntimeError("core smoke test produced an invalid pattern")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
        # Some GUI/runtime DLLs keep background cleanup handlers alive in frozen builds.
        # The smoke test is deliberately terminal, so exit immediately after success.
        os._exit(0)
    else:
        run()
