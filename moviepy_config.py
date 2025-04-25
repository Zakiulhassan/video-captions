from moviepy.config import change_settings

# Set the path to your ImageMagick convert.exe
# Replace this path with your actual ImageMagick installation path
IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\convert.exe"

# Apply the settings
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})
