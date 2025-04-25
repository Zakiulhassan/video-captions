import os
import json
from deepgram import DeepgramClient, PrerecordedOptions
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using the Deepgram API.

    Args:
        audio_path (str): Path to the audio file to transcribe

    Returns:
        dict: Transcription results from Deepgram
    """
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not deepgram_api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPGRAM_API_KEY not found in environment variables",
        )

    try:
        deepgram = DeepgramClient(deepgram_api_key)

        with open(audio_path, "rb") as buffer_data:
            payload = {"buffer": buffer_data}

            # Simplified options that should work with most Deepgram accounts
            options = PrerecordedOptions(
                smart_format=True,
                model="nova-2",
                language="en-US",
                punctuate=True,
                # Removed enhanced tier and other options that might be causing the 403 error
            )

            # Call the Deepgram API to transcribe the audio file
            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                payload, options
            )

            # Parse the JSON response into a dictionary
            response_dict = json.loads(response.to_json())

            return response_dict

    except Exception as e:
        print(f"Exception: {e}")
        return {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": f"Error during transcription: {str(e)}",
                                "words": [],
                            }
                        ]
                    }
                ]
            }
        }
