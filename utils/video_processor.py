from moviepy.video.io.VideoFileClip import VideoFileClip
import os


def extract_audio(video_path, output_audio_path):
    """
    Extract audio from a video file and save it as a compressed MP3 file.

    Args:
        video_path (str): Path to the input video file
        output_audio_path (str): Path where the extracted audio will be saved

    Returns:
        str: Path to the extracted audio file
    """
    try:
        video = VideoFileClip(video_path)
        audio = video.audio

        # Save audio as MP3 with a lower bitrate to reduce file size
        audio.write_audiofile(
            output_audio_path, codec="mp3", bitrate="128k"
        )  # 128k bitrate for compression

        video.close()
        return output_audio_path
    except Exception as e:
        raise Exception(f"Error extracting audio: {str(e)}")
