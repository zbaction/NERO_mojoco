from PIL import Image
import pathlib

OUT = pathlib.Path.home() / "NERO" / "out"
im = Image.open(OUT / "nero_overview.png")
c = im.crop((690, 320, 1140, 630)).resize((900, 620), Image.LANCZOS)
c.save(OUT / "zoom_basket_overview.png")
print("ok", c.size)
