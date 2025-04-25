import datetime
import os

def convert_to_srt(transcription_result, output_path):
    """
    Convert Deepgram transcription results to SRT format and save to a file.
    
    Args:
        transcription_result (dict): Transcription results from Deepgram
        output_path (str): Path where the SRT file will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    srt_content = []
    counter = 1
    
    try:
        # Check if the response contains paragraphs with sentences (as in your example)
        if ('results' in transcription_result and 
            'channels' in transcription_result['results'] and 
            'alternatives' in transcription_result['results']['channels'][0]):
            
            alternatives = transcription_result['results']['channels'][0]['alternatives'][0]
            
            # Process paragraphs with sentences structure
            if 'paragraphs' in alternatives:
                paragraphs = alternatives['paragraphs']['paragraphs']
                
                for paragraph in paragraphs:
                    if 'sentences' in paragraph:
                        for sentence in paragraph['sentences']:
                            start_time = format_timestamp(float(sentence['start']))
                            end_time = format_timestamp(float(sentence['end']))
                            text = sentence['text']
                            
                            srt_entry = f"{counter}\n{start_time} --> {end_time}\n{text}\n\n"
                            srt_content.append(srt_entry)
                            counter += 1
            
            # Fallback to words-based segmentation if paragraphs/sentences not available
            elif 'words' in alternatives:
                words = alternatives['words']
                
                # Group words into sentences (simple approach: pause-based)
                current_sentence = {'words': [], 'start': None, 'end': None}
                
                for word in words:
                    # Check if this is a new sentence (first word or long pause)
                    if current_sentence['start'] is None:
                        current_sentence['start'] = word['start']
                        current_sentence['words'].append(word['word'])
                        current_sentence['end'] = word['end']
                    elif float(word['start']) - float(current_sentence['end']) > 0.7:  # Pause threshold
                        # Complete previous sentence
                        sentence_text = ' '.join(current_sentence['words'])
                        start_time = format_timestamp(float(current_sentence['start']))
                        end_time = format_timestamp(float(current_sentence['end']))
                        
                        srt_entry = f"{counter}\n{start_time} --> {end_time}\n{sentence_text}\n\n"
                        srt_content.append(srt_entry)
                        counter += 1
                        
                        # Start new sentence
                        current_sentence = {'words': [word['word']], 'start': word['start'], 'end': word['end']}
                    else:
                        # Continue current sentence
                        current_sentence['words'].append(word['word'])
                        current_sentence['end'] = word['end']
                
                # Add the last sentence if not empty
                if current_sentence['words']:
                    sentence_text = ' '.join(current_sentence['words'])
                    start_time = format_timestamp(float(current_sentence['start']))
                    end_time = format_timestamp(float(current_sentence['end']))
                    
                    srt_entry = f"{counter}\n{start_time} --> {end_time}\n{sentence_text}\n\n"
                    srt_content.append(srt_entry)
            
            # Last resort: use the full transcript as a single caption
            else:
                transcript = alternatives.get('transcript', 'No transcription available')
                srt_entry = f"1\n00:00:00,000 --> 99:59:59,999\n{transcript}\n\n"
                srt_content.append(srt_entry)
    
    except Exception as e:
        # Create an error message if processing fails
        srt_entry = f"1\n00:00:00,000 --> 00:00:10,000\nTranscription processing error: {str(e)}\n\n"
        srt_content.append(srt_entry)
    
    # Write SRT content to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(srt_content))

def format_timestamp(seconds):
    """
    Format a timestamp in seconds to SRT format: HH:MM:SS,mmm
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timestamp
    """
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"