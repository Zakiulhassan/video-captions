import os
os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

from moviepy.editor import TextClip, CompositeVideoClip, ColorClip

# Create a background clip (10s black screen)
background = ColorClip(size=(640, 360), color=(0, 0, 0), duration=10)

# Create a caption using ImageMagick (method='caption')
text = TextClip("Hello from MoviePy!\nImageMagick is working ✅", 
                fontsize=24, color='white', method='caption', size=(600, None))

# Position it at the bottom and match timing
text = text.set_position("bottom").set_duration(10)

# Combine both
final = CompositeVideoClip([background, text])

# Export video
final.write_videofile("test_output.mp4", fps=24)
