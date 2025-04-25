import os
import re
from typing import List, Dict, Any
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.VideoClip import TextClip
from moviepy.config import change_settings

# Configure MoviePy to use ImageMagick
IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
change_settings(
    {
        "IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY,
        "IMAGEMAGICK_BINARY_PATH": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI",
    }
)

# Set up logging
logger = logging.getLogger(__name__)


class VideoContentAnalyzer:
    def __init__(self, api_key=None):
        """Initialize the content analyzer."""
        pass  # No API key needed for rule-based approach

    def analyze_transcript(
        self, transcript_data: Dict[str, Any], max_clips: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Analyze transcript to identify exactly 2-3 important segments, no duplicates.

        Args:
            transcript_data: The transcript data from Deepgram
            max_clips: Maximum number of clips to generate (default: 3)

        Returns:
            List of 2-3 most important segments with start/end times and importance score
        """
        try:
            # Extract full transcript text and all words with timestamps
            all_words = []
            full_transcript = ""

            if (
                "results" in transcript_data
                and "channels" in transcript_data["results"]
                and "alternatives" in transcript_data["results"]["channels"][0]
            ):

                alternatives = transcript_data["results"]["channels"][0][
                    "alternatives"
                ][0]
                full_transcript = alternatives.get("transcript", "")

                if "words" in alternatives:
                    all_words = alternatives["words"]

            if not full_transcript or not all_words:
                logger.error("No transcript or word timing data found")
                return []

            # Create context-rich segments directly from transcript
            segments = self._create_meaningful_segments(all_words, full_transcript)

            # Score segments based on importance
            scored_segments = self._score_segments(segments)

            # Select exactly max_clips non-overlapping segments
            selected_segments = self._select_top_segments(scored_segments, max_clips)

            # Ensure reasonable clip durations (minimum 15 seconds for context)
            selected_segments = self._adjust_clip_durations(selected_segments)

            # Sort by timestamp
            selected_segments.sort(key=lambda x: x.get("start_time", 0))

            return selected_segments

        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            return []

    def _create_meaningful_segments(
        self, all_words: List[Dict], full_transcript: str
    ) -> List[Dict]:
        """Create context-rich segments from the transcript."""
        segments = []

        # First pass: Create paragraph-level segments based on pauses
        current_segment = {"text": "", "start": None, "end": None, "words": []}

        for i, word in enumerate(all_words):
            word_text = word.get("word", "")
            word_start = float(word.get("start", 0))
            word_end = float(word.get("end", 0))

            # Start a new segment if this is the first word
            if current_segment["text"] == "":
                current_segment["text"] = word_text
                current_segment["start"] = word_start
                current_segment["words"].append(word)
            # If there's a significant pause (> 1.5 seconds), end current segment
            elif i > 0 and word_start - float(all_words[i - 1].get("end", 0)) > 1.5:
                current_segment["end"] = float(all_words[i - 1].get("end", 0))

                # Only add segment if it has reasonable length
                if len(current_segment["words"]) > 5:
                    segments.append(current_segment)

                # Start new segment
                current_segment = {
                    "text": word_text,
                    "start": word_start,
                    "end": None,
                    "words": [word],
                }
            # Otherwise, continue current segment
            else:
                current_segment["text"] += " " + word_text
                current_segment["words"].append(word)

        # Add final segment
        if current_segment["text"] and len(current_segment["words"]) > 5:
            current_segment["end"] = float(current_segment["words"][-1].get("end", 0))
            segments.append(current_segment)

        # Second pass: Merge short segments to ensure minimal context
        merged_segments = []
        temp_segment = None

        for segment in segments:
            if not temp_segment:
                temp_segment = segment.copy()
            elif float(segment["end"]) - float(temp_segment["start"]) < 45:
                # Merge if combined segment would be less than 45 seconds
                temp_segment["text"] += " " + segment["text"]
                temp_segment["end"] = segment["end"]
                temp_segment["words"].extend(segment["words"])
            else:
                merged_segments.append(temp_segment)
                temp_segment = segment.copy()

        # Add final merged segment
        if temp_segment:
            merged_segments.append(temp_segment)

        return merged_segments

    def _score_segments(self, segments: List[Dict]) -> List[Dict]:
        """Score segments based on importance factors."""
        scored_segments = []

        # Define important indicators
        topic_keywords = [
            "main point",
            "important",
            "key",
            "critical",
            "essential",
            "significant",
            "highlight",
            "conclusion",
            "therefore",
            "summary",
            "ethics",
            "philosophy",
            "framework",
            "theory",
            "concept",
            "principle",
            "foundation",
            "structure",
            "understanding",
            "analysis",
            "explanation",
        ]

        for segment in segments:
            segment_text = segment.get("text", "").lower()
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            duration = end_time - start_time

            # Skip segments that are too short (< 10 seconds)
            if duration < 10:
                continue

            # Base score - favor segments between 15-60 seconds
            base_score = 0
            if 15 <= duration <= 60:
                base_score = 5
            elif 60 < duration <= 90:
                base_score = 3
            else:
                base_score = 1

            # Content score - check for important keywords and phrases
            content_score = sum(
                2 for keyword in topic_keywords if keyword in segment_text
            )
            content_score = min(content_score, 10)  # Cap at 10 points

            # Position score - introduction and conclusion are important
            position_score = 0
            if len(segments) > 0:
                segment_index = next(
                    (
                        i
                        for i, s in enumerate(segments)
                        if s.get("text") == segment.get("text")
                    ),
                    -1,
                )

                # First segment gets high score
                if segment_index == 0:
                    position_score = 5
                # Last segment gets high score
                elif segment_index == len(segments) - 1:
                    position_score = 5
                # Segments in the middle get scores based on relative position
                else:
                    rel_position = segment_index / len(segments)
                    # Higher scores for segments around 1/3 and 2/3 of the content
                    if 0.3 <= rel_position <= 0.4 or 0.6 <= rel_position <= 0.7:
                        position_score = 3

            total_score = base_score + content_score + position_score

            # Add reasonable segment description
            description = self._generate_segment_description(segment_text)

            scored_segments.append(
                {
                    "text": segment.get("text", ""),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "importance_score": total_score,
                    "description": description,
                }
            )

        # Sort by importance score (descending)
        scored_segments.sort(key=lambda x: x.get("importance_score", 0), reverse=True)

        return scored_segments

    def _select_top_segments(
        self, scored_segments: List[Dict], max_clips: int
    ) -> List[Dict]:
        """Select exactly the top non-overlapping segments."""
        if not scored_segments:
            return []

        if len(scored_segments) <= max_clips:
            return scored_segments

        selected = []

        # Start with the highest-scoring segment
        selected.append(scored_segments[0])

        # Add more segments, ensuring no overlap
        for segment in scored_segments[1:]:
            if len(selected) >= max_clips:
                break

            # Check for overlap with already selected segments
            has_overlap = False
            for selected_segment in selected:
                # Check if the segments overlap in time
                if (
                    segment["start_time"] < selected_segment["end_time"]
                    and segment["end_time"] > selected_segment["start_time"]
                ):
                    has_overlap = True
                    break

            if not has_overlap:
                selected.append(segment)

        # If we couldn't get enough non-overlapping segments,
        # try adjusting start/end times of highest scoring segments
        if len(selected) < max_clips and len(selected) < len(scored_segments):
            remaining_slots = min(
                max_clips - len(selected), len(scored_segments) - len(selected)
            )

            for i in range(remaining_slots):
                for segment in scored_segments:
                    if segment in selected:
                        continue

                    # Try to adjust this segment to avoid overlap
                    adjusted_segment = segment.copy()

                    # Find any overlapping selected segments
                    overlaps = []
                    for sel in selected:
                        if (
                            adjusted_segment["start_time"] < sel["end_time"]
                            and adjusted_segment["end_time"] > sel["start_time"]
                        ):
                            overlaps.append(sel)

                    if not overlaps:
                        selected.append(adjusted_segment)
                        break

                    # Try to adjust start/end times to avoid overlap
                    for overlap in overlaps:
                        # If this segment starts during an existing one, start after it
                        if (
                            adjusted_segment["start_time"] >= overlap["start_time"]
                            and adjusted_segment["start_time"] < overlap["end_time"]
                        ):
                            adjusted_segment["start_time"] = overlap["end_time"] + 1

                        # If this segment ends during an existing one, end before it
                        if (
                            adjusted_segment["end_time"] > overlap["start_time"]
                            and adjusted_segment["end_time"] <= overlap["end_time"]
                        ):
                            adjusted_segment["end_time"] = overlap["start_time"] - 1

                        # Recalculate duration
                        adjusted_segment["duration"] = (
                            adjusted_segment["end_time"]
                            - adjusted_segment["start_time"]
                        )

                    # Only add if adjustment resulted in a reasonable segment
                    if adjusted_segment["duration"] >= 15:
                        selected.append(adjusted_segment)
                        break

        return selected[:max_clips]

    def _adjust_clip_durations(self, segments: List[Dict]) -> List[Dict]:
        """Ensure clips have reasonable duration for context."""
        adjusted_segments = []

        for segment in segments:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time

            # Ensure minimum duration of 15 seconds
            if duration < 15:
                # Add padding to reach minimum duration
                padding_needed = 15 - duration
                start_time = max(0, start_time - padding_needed / 2)
                end_time = end_time + padding_needed / 2

            # Ensure maximum duration of 60 seconds
            if end_time - start_time > 60:
                # Reduce to maximum duration
                middle = (start_time + end_time) / 2
                start_time = middle - 30
                end_time = middle + 30

            adjusted = segment.copy()
            adjusted["start_time"] = start_time
            adjusted["end_time"] = end_time
            adjusted["duration"] = end_time - start_time

            adjusted_segments.append(adjusted)

        return adjusted_segments

    def _generate_segment_description(self, text: str) -> str:
        """Generate a descriptive reason for why this segment is important."""
        # Check for introduction indicators
        if any(
            phrase in text.lower()
            for phrase in ["introduce", "welcome", "begin", "start", "first"]
        ):
            return "Introduces key concepts and context"

        # Check for conclusion indicators
        if any(
            phrase in text.lower()
            for phrase in ["conclusion", "therefore", "finally", "summary", "thus"]
        ):
            return "Summarizes key points and conclusions"

        # Check for example indicators
        if any(
            phrase in text.lower()
            for phrase in [
                "example",
                "instance",
                "for instance",
                "illustrate",
                "demonstrate",
            ]
        ):
            return "Provides illustrative example of key concepts"

        # Check for key point indicators
        if any(
            phrase in text.lower()
            for phrase in [
                "important",
                "critical",
                "essential",
                "key point",
                "main point",
            ]
        ):
            return "Presents critical information and core concepts"

        # Default description
        return "Contains relevant content for understanding the topic"

    def extract_clips(
        self, video_path: str, important_segments: List[Dict]
    ) -> List[Dict]:
        """
        Extract exactly 2-3 video clips for important segments.

        Args:
            video_path: Path to the original video file
            important_segments: List of important segments with timestamps

        Returns:
            List of paths to the extracted clips and their metadata
        """
        try:
            clips_info = []

            # Create clips directory if it doesn't exist
            os.makedirs("clips", exist_ok=True)

            # Load the video
            video = VideoFileClip(video_path)

            # Verify video exists and has valid duration
            video_duration = video.duration

            # Extract each important segment as a clip
            for i, segment in enumerate(important_segments):
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", start_time + 15)

                # Ensure times are within video bounds
                start_time = max(0, min(start_time, video_duration - 1))
                end_time = max(start_time + 15, min(end_time, video_duration))

                # Create subclip
                subclip = video.subclip(start_time, end_time)

                # Generate clip filename
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                clip_filename = f"{base_filename}_clip_{i+1}.mp4"
                clip_path = os.path.join("clips", clip_filename)

                # Write clip to file
                subclip.write_videofile(clip_path, codec="libx264", audio_codec="aac")

                # Add clip info to result
                clips_info.append(
                    {
                        "clip_index": i + 1,
                        "text": segment.get("text"),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "importance_score": segment.get("importance_score"),
                        "description": segment.get("description"),
                        "clip_path": clip_path,
                        "clip_filename": clip_filename,
                    }
                )

            # Close the video to free resources
            video.close()

            return clips_info

        except Exception as e:
            logger.error(f"Error extracting clips: {str(e)}")
            return []

    def create_highlight_reel(
        self, video_path: str, important_segments: List[Dict], output_path: str = None
    ) -> str:
        """
        Create a SINGLE highlight reel from exactly 2-3 important segments.

        Args:
            video_path: Path to the original video file
            important_segments: List of important segments with timestamps
            output_path: Path to save the highlight reel (optional)

        Returns:
            Path to the created highlight reel
        """
        try:
            # Create highlights directory if it doesn't exist
            os.makedirs("highlights", exist_ok=True)

            # Generate output path if not provided
            if not output_path:
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(
                    "highlights", f"{base_filename}_highlights.mp4"
                )

            # Check if output file already exists
            if os.path.exists(output_path):
                os.remove(
                    output_path
                )  # Remove existing file to prevent multiple highlight reels

            # Load the video
            video = VideoFileClip(video_path)
            video_duration = video.duration

            # Sort segments by start time to maintain chronological order
            important_segments.sort(key=lambda x: x.get("start_time", 0))

            # Create clips for each important segment
            clips = []
            for segment in important_segments:
                start_time = segment.get("start_time", 0)
                end_time = segment.get("end_time", start_time + 15)

                # Ensure times are within video bounds
                start_time = max(0, min(start_time, video_duration - 1))
                end_time = max(start_time + 15, min(end_time, video_duration))

                # Create subclip and add to list
                subclip = video.subclip(start_time, end_time)
                clips.append(subclip)

            # Concatenate clips into a highlight reel
            if clips:
                highlight_reel = concatenate_videoclips(clips)

                # Write highlight reel to file
                highlight_reel.write_videofile(
                    output_path, codec="libx264", audio_codec="aac"
                )

                # Close the highlight reel to free resources
                highlight_reel.close()

            # Close the video to free resources
            video.close()

            return output_path if os.path.exists(output_path) else ""

        except Exception as e:
            logger.error(f"Error creating highlight reel: {str(e)}")
            return ""

    def _create_caption_segments(self, all_words: List[Dict]) -> List[Dict]:
        """
        Group words into caption segments of reasonable length.
        Each segment will contain up to 8–10 words or until a punctuation mark ends the phrase.
        """
        caption_segments = []
        current_segment = {"text": "", "start": None, "end": None, "words": []}

        max_words = 10
        word_count = 0

        for word in all_words:
            word_text = word.get("word", "")
            word_start = float(word.get("start", 0))
            word_end = float(word.get("end", 0))

            if current_segment["text"] == "":
                current_segment["start"] = word_start
                current_segment["text"] = word_text
                current_segment["words"] = [word]
            else:
                current_segment["text"] += " " + word_text
                current_segment["words"].append(word)

            word_count += 1

            # Break line if punctuation or too many words
            if word_text.endswith((".", "!", "?")) or word_count >= max_words:
                current_segment["end"] = word_end
                current_segment["text"] = self._format_caption_text(
                    current_segment["text"]
                )
                caption_segments.append(current_segment)

                # Reset segment
                current_segment = {"text": "", "start": None, "end": None, "words": []}
                word_count = 0

        # Add any remaining segment
        if current_segment["text"]:
            current_segment["end"] = current_segment["words"][-1].get("end", 0)
            current_segment["text"] = self._format_caption_text(current_segment["text"])
            caption_segments.append(current_segment)

        return caption_segments

    def _format_caption_text(self, text: str) -> str:
        """Format caption text for better readability."""
        return "\n".join(textwrap.wrap(text.strip(), width=40))

    def create_captioned_video(
        self, video_path: str, transcript_data: Dict, output_path: str = None
    ) -> str:
        """
        Create a video with embedded captions based on transcript data.
        Uses moviepy's TextClip for better text rendering.

        Args:
            video_path: Path to the original video file
            transcript_data: The transcript data from Deepgram
            output_path: Path to save the captioned video (optional)

        Returns:
            Path to the created captioned video
        """
        try:
            # Create captioned directory if it doesn't exist
            os.makedirs("captioned", exist_ok=True)

            # Generate output path if not provided
            if not output_path:
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(
                    "captioned", f"{base_filename}_captioned.mp4"
                )

            # Extract all words with timestamps
            all_words = []

            if (
                "results" in transcript_data
                and "channels" in transcript_data["results"]
                and "alternatives" in transcript_data["results"]["channels"][0]
            ):
                alternatives = transcript_data["results"]["channels"][0][
                    "alternatives"
                ][0]
                if "words" in alternatives:
                    all_words = alternatives["words"]

            if not all_words:
                logger.error("No word timing data found in transcript")
                return ""

            # Group words into caption segments
            caption_segments = self._create_caption_segments(all_words)
            logger.info(f"Created {len(caption_segments)} caption segments")

            # Load the original video
            logger.info(f"Loading video from: {video_path}")
            video = VideoFileClip(video_path)
            logger.info(
                f"Video loaded successfully. Duration: {video.duration} seconds"
            )

            # Create a list to store the clips
            clips = [video]

            # Add captions as text overlays
            for i, segment in enumerate(caption_segments):
                try:
                    caption_text = segment["text"]
                    start_time = segment["start"]
                    end_time = segment["end"]
                    duration = end_time - start_time

                    logger.info(
                        f"Creating text clip {i+1}/{len(caption_segments)}: {caption_text[:50]}..."
                    )

                    # Create a text clip with proper styling
                    txt_clip = (
                        TextClip(
                            caption_text,
                            fontsize=24,
                            color="white",
                            stroke_color="black",
                            stroke_width=1.5,
                            method="label",  # Using label method which is more reliable
                            align="center",
                            size=(video.w * 0.9, None),
                            font="Arial",  # Specify a font that's available on Windows
                            kerning=0,  # Disable kerning for better compatibility
                            interline=-1,  # Adjust line spacing
                        )
                        .set_position(("center", "bottom"))
                        .set_start(start_time)
                        .set_duration(duration)
                    )

                    clips.append(txt_clip)
                    logger.info(f"Text clip {i+1} created successfully")
                except Exception as e:
                    logger.error(f"Error creating text clip {i+1}: {str(e)}")
                    continue

            # Combine all clips
            logger.info("Combining video and text clips")
            final_video = CompositeVideoClip(clips)

            # Write the final video
            logger.info(f"Writing captioned video to: {output_path}")
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                threads=4,
                preset="medium",
                logger=None,  # Disable moviepy's progress bar
            )

            # Close video resources
            video.close()
            final_video.close()

            # Verify the output file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(
                    f"Captioned video created successfully. File size: {file_size} bytes"
                )
                return output_path
            else:
                logger.error("Output file was not created")
                return ""

        except Exception as e:
            logger.error(f"Error creating captioned video: {str(e)}")
            return ""
