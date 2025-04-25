from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import shutil
import logging
from utils.video_processor import extract_audio
from utils.transcription import transcribe_audio
from utils.srt_converter import convert_to_srt
from utils.content_analyzer import VideoContentAnalyzer
from utils.s3_client import MinioClient
from dotenv import load_dotenv
from typing import Optional, List, Dict
import tempfile

# Load environment variables
load_dotenv()

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Captioning and Analysis Service")

# Initialize MinIO client
s3_client = MinioClient()

# Initialize content analyzer
content_analyzer = VideoContentAnalyzer()


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Video Captioning and Analysis Service API"}


@app.post("/generate-captions/")
async def generate_captions(file: UploadFile = File(...)):
    """
    Generate captions (SRT file) from a video or audio file.

    Args:
        file: The video or audio file to process
    """
    try:
        logger.info(f"Starting caption generation for file: {file.filename}")

        # 1. Save the uploaded file to a temporary location
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")

        # Check file format
        if file_extension not in [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".mp3",
            ".wav",
            ".flac",
            ".m4a",
            ".ogg",
        ]:
            logger.error(f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.close()

        # Save uploaded file to temporary location
        with open(temp_file.name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to MinIO
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_path = f"uploads/{unique_filename}"
        s3_client.upload_file(temp_file.name, s3_path)

        # 2. Extract audio if it's a video file
        if file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
            audio_path = os.path.join(
                "audio", f"{os.path.splitext(unique_filename)[0]}.mp3"
            )
            logger.info(f"Extracting audio to: {audio_path}")
            extract_audio(temp_file.name, audio_path)

            # Upload audio to MinIO
            audio_s3_path = f"audio/{os.path.splitext(unique_filename)[0]}.mp3"
            s3_client.upload_file(audio_path, audio_s3_path)

            # Download audio for processing
            file_to_transcribe = s3_client.download_file(audio_s3_path)
        else:
            file_to_transcribe = temp_file.name

        # 3. Transcribe audio
        logger.info("Starting transcription process")
        transcription_result = transcribe_audio(file_to_transcribe)
        logger.info("Transcription completed")

        # 4. Convert to SRT
        srt_filename = f"{os.path.splitext(unique_filename)[0]}.srt"
        srt_path = os.path.join("captions", srt_filename)
        logger.info(f"Converting transcription to SRT format: {srt_path}")
        convert_to_srt(transcription_result, srt_path)

        # Upload SRT to MinIO
        srt_s3_path = f"captions/{srt_filename}"
        s3_client.upload_file(srt_path, srt_s3_path)

        # Clean up temporary files
        os.unlink(temp_file.name)
        if file_to_transcribe != temp_file.name:
            os.unlink(file_to_transcribe)

        return {
            "message": "Captions generated successfully",
            "srt_filename": srt_filename,
            "download_url": s3_client.get_file_url(srt_s3_path),
            "file_size": os.path.getsize(srt_path),
        }

    except Exception as e:
        logger.error(f"Error in generate_captions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-captioned-video/")
async def create_captioned_video(file: UploadFile = File(...)):
    """
    Create a video with embedded captions.

    Args:
        file: The video file to process
    """
    try:
        logger.info(f"Starting captioned video creation for file: {file.filename}")

        # 1. Save the video file to a temporary location
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")

        if file_extension not in [".mp4", ".avi", ".mov", ".mkv"]:
            logger.error(f"Unsupported video format: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported video format")

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.close()

        # Save uploaded file to temporary location
        with open(temp_file.name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to MinIO
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_path = f"uploads/{unique_filename}"
        s3_client.upload_file(temp_file.name, s3_path)

        # 2. Extract audio and transcribe
        audio_path = os.path.join(
            "audio", f"{os.path.splitext(unique_filename)[0]}.mp3"
        )
        logger.info(f"Extracting audio to: {audio_path}")
        extract_audio(temp_file.name, audio_path)

        # Upload audio to MinIO
        audio_s3_path = f"audio/{os.path.splitext(unique_filename)[0]}.mp3"
        s3_client.upload_file(audio_path, audio_s3_path)

        # Download audio for processing
        audio_file = s3_client.download_file(audio_s3_path)

        logger.info("Starting transcription")
        transcription_result = transcribe_audio(audio_file)
        logger.info("Transcription completed")

        # 3. Create captioned video
        captioned_filename = f"{os.path.splitext(unique_filename)[0]}_captioned.mp4"
        captioned_path = os.path.join("captioned", captioned_filename)
        logger.info(f"Creating captioned video at: {captioned_path}")

        result_path = content_analyzer.create_captioned_video(
            temp_file.name, transcription_result, captioned_path
        )

        # Upload captioned video to MinIO
        captioned_s3_path = f"captioned/{captioned_filename}"
        s3_client.upload_file(result_path, captioned_s3_path)

        # Clean up temporary files
        os.unlink(temp_file.name)
        os.unlink(audio_file)

        return {
            "message": "Captioned video created successfully",
            "video_filename": captioned_filename,
            "download_url": s3_client.get_file_url(captioned_s3_path),
            "file_size": os.path.getsize(result_path),
        }

    except Exception as e:
        logger.error(f"Error in create_captioned_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-videos/")
async def list_videos():
    """
    List all uploaded videos.
    """
    try:
        videos = s3_client.list_files(prefix="uploads/")
        return {"videos": videos}
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-captions/")
async def list_captions():
    """
    List all generated captions.
    """
    try:
        captions = s3_client.list_files(prefix="captions/")
        return {"captions": captions}
    except Exception as e:
        logger.error(f"Error listing captions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-captioned-videos/")
async def list_captioned_videos():
    """
    List all captioned videos.
    """
    try:
        videos = s3_client.list_files(prefix="captioned/")
        return {"videos": videos}
    except Exception as e:
        logger.error(f"Error listing captioned videos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """
    Download a file from MinIO.

    Args:
        file_type: Type of file (uploads, captions, captioned)
        filename: Name of the file
    """
    try:
        s3_path = f"{file_type}/{filename}"
        local_path = s3_client.download_file(s3_path)

        return FileResponse(
            local_path, media_type="application/octet-stream", filename=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
