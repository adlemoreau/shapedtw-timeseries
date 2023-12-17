import csv
import librosa
import os
import re
from dtw import dtw
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import DerivativeShapeDescriptor

def distance_between_audios(audiofile1, audiofile2, distance):
    """Given 2 audio files, calculate the distance between them using the specified distance metric.

    Args:
        audiofile1 (str): file path to audio file 1
        audiofile2 (str): file path to audio file 2
        distance (str): DTW or shapeDTW

    Returns:
        float: distance between the 2 audio files
    """
    # Load audio files
    audio1, sr1 = librosa.load(audiofile1)
    audio2, sr2 = librosa.load(audiofile2)

    # Extract MFCC features
    mfccs1 = librosa.feature.mfcc(y=audio1, sr=sr1).T
    mfccs2 = librosa.feature.mfcc(y=audio2, sr=sr2).T

    # Flatten MFCC matrices
    mfccs1_flat = mfccs1.flatten()
    mfccs2_flat = mfccs2.flatten()
    
    if distance == "dtw":
        alignment = dtw(mfccs1_flat, mfccs2_flat, keep_internals=True)
        
        return alignment.normalizedDistance
    
    elif distance == "shapedtw":
        derivative_descriptor = DerivativeShapeDescriptor()
        alignment = shape_dtw(
            x = mfccs1_flat,
            y = mfccs2_flat,
            subsequence_width = 30,
            step_pattern = "symmetric2",
            shape_descriptor = derivative_descriptor
            )

        return alignment.normalized_distance
    
def get_most_similar_sentence(audio_file_original, speaker_to_compare, distance):
    """Given an audio file, find the most similar sentence from the same speaker and the distance.

    Args:
        audio_file_original (str): file path to audio file
        speaker_to_compare (str): name of speaker to compare (DC, JE, JK, or KL)
        distance (str): DTW or shapeDTW

    Returns:
        (str, float): file path to most similar audio file, distance between the 2 audio files
    """
    min_dist = 1000000 #init with a large number
    audio_min = ''
    
    with os.scandir('data/savee') as entries:
        files = [entry.name for entry in entries if entry.is_file()]
        files = [f"data/savee/{files[i]}" for i in range(len(files)) if files[i].startswith(speaker_to_compare)]

    for audio_to_compare in files:
        calc_dist = distance_between_audios(audio_file_original, audio_to_compare, distance)
        if calc_dist < min_dist:
            min_dist = calc_dist
            audio_min = audio_to_compare

    return audio_min, min_dist

def get_all_distance_for_one_user(speaker, distance):
    """Given a speaker, calculate the distance between all sentences from the same speaker and the other speakers.

    Args:
        speaker (str): name of speaker (DC, JE, JK, or KL)
        distance (str): DTW or shapeDTW

    Returns:
        List[dict]: list of dictionaries containing the file path to the audio file as key and a tuple (most similar audio file path, distance) as value
    """
    
    speakers = ['DC', 'JK', 'JE'] # KL is not included
    speakers_to_compare_list = [og_speaker for og_speaker in speakers if og_speaker != speaker]
    res_list = []
    
    with os.scandir('data/savee') as entries:
        files = [entry.name for entry in entries if entry.is_file()]
        files = [f"data/savee/{files[i]}" for i in range(len(files)) if files[i].startswith(speaker)]
    
    for audio in files:
        res_audio = {}
        for speaker_to_compare in speakers_to_compare_list:
            res_audio[audio] = get_most_similar_sentence(audio, speaker_to_compare, distance)
            print(res_audio)
            res_list.append(res_audio)
    
    # print(f"Results for {speaker}: {res_list}")
    return res_list

def get_metrics(speaker, data):
    """Get the number of truly and wrongly matched sentences for a speaker

    Args:
        speaker (str): speaker to compare to (DC, KL, JK, JE)
        data (list): list of dictionaries with the original sentence as key and the most similar sentence as value

    Returns:
        dict: dictionary with the number of truly and wrongly matched sentences for each speaker
    """
    pattern = r'/[A-Z]+_([a-z0-9]+)\.'
    
    true_count_DC, false_count_DC = 0, 0
    true_count_KL, false_count_KL = 0, 0
    true_count_JE, false_count_JE = 0, 0
    true_count_JK, false_count_JK = 0, 0
    res_dict = {}
    
    # Iterate through the list of dictionaries
    for entry in data:
        key, (value, _) = entry.popitem()
        
        true_match_sentence = re.search(pattern, key)
        true_sentence_nb = true_match_sentence.group(1) if true_match_sentence else None
        
        similar_match_sentence = re.search(pattern, value)
        similar_sentence_nb = similar_match_sentence.group(1) if similar_match_sentence else None
        
        speaker_code = value.split('/')[-1].split('_')[0]
        
        if true_sentence_nb == similar_sentence_nb:
            if speaker_code == 'KL':
                true_count_KL += 1
            elif speaker_code == 'JE':
                true_count_JE += 1
            elif speaker_code == 'JK':
                true_count_JK += 1
            else:
                true_count_DC += 1
        else:
            if speaker_code == 'KL':
                false_count_KL += 1
            elif speaker_code == 'JE':
                false_count_JE += 1
            elif speaker_code == 'JK':
                false_count_JK += 1
            else:
                false_count_DC += 1
    
    for speaker_ in ['JE', 'JK', 'DC']:
        res_dict[f"{speaker_} compared to {speaker}"] = {
            'Correct matches': eval(f'true_count_{speaker_}'),
            'Wrong matches': eval(f'false_count_{speaker_}')
        }
        
    return res_dict

def load_csv(file_path):
    """Load a csv file and return a list of the data

    Args:
        file_path (str): file path to csv file

    Returns:
        list[dict]: list of dictionaries containing the file path to the audio file as key and a tuple (most similar audio file path, distance) as value
    """
    data_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        # Ignore header
        next(csv_reader, None)
        for row in csv_reader:
            data_list.append(eval(row[0]))
            
    return data_list