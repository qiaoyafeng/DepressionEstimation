

import os
import struct
import numpy as np
import pandas as pd
import wave
import librosa
import tensorflow_hub as hub

from pydub import AudioSegment


def save_wav_segment(input_path, output_path, start_time_ms, end_time_ms):
    """
    截取WAV文件片段并保存。

    参数:
    - input_path: 输入WAV文件路径。
    - output_path: 输出WAV文件路径。
    - start_time_ms: 截取起始时间（毫秒）。
    - end_time_ms: 截取结束时间（毫秒）。
    """
    # 加载原始音频文件
    audio = AudioSegment.from_wav(input_path)

    # 截取片段
    segment = audio[start_time_ms:end_time_ms]

    # 保存截取的片段
    segment.export(output_path, format="wav")
    print(f"片段保存成功: {output_path}")

def load_audio(audio_path):
    wavefile = wave.open(audio_path)
    audio_sr = wavefile.getframerate()
    n_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)

    return signal.astype(float), audio_sr


def audio_clipping(audio_path, text_df, patient_id):
    print(f"audio_clipping: audio_path: {audio_path}, patient_id: {patient_id}")
    count = 0
    for t in text_df.itertuples():
        if getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            else:
                start = getattr(t, 'start_time')
                stop = getattr(t, 'stop_time')
                if (stop-start) > 5:
                    output_path = f"{output_root}/audio_clip/{patient_id}_{count}.wav"
                    start_time_ms = start * 1000
                    end_time_ms = stop * 1000
                    save_wav_segment(audio_path, output_path, start_time_ms, end_time_ms)
                    count += 1


if __name__ == '__main__':

    # output root
    root = r'E:\myworkspace\DepressionEstimation\daic_woz_preprocessing\daic_woz_dataset'
    root_dir = os.path.join(root, 'DAIC_WOZ-generated_database_V2', 'test')

    # read training gt file
    gt_path = rf'{root}\test_split_Depression_AVEC2017.csv'
    gt_df = pd.read_csv(gt_path) 

    GT = {'original_data': {'ID_gt':[], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt':[], 'phq_subscores_gt':[]}, 
          'clipped_data': {'ID_gt':[], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt':[], 'phq_subscores_gt':[]}}
    
    for i in range(len(gt_df)):
        # extract training gt details
        patient_id = gt_df['participant_ID'][i]

        audio_path = f'{root}/{patient_id}_P/{patient_id}_AUDIO.wav'
        text_path = f'{root}/{patient_id}_P/{patient_id}_TRANSCRIPT.csv'

        # read transcipt file
        text_df = pd.read_csv(text_path, sep='\t').fillna('')


        print(f'Extracting feature of Participant {patient_id} for clipped_data...')

        output_root = os.path.join(root_dir, 'clipped_data')

        # audio
        clipped_audio = audio_clipping(audio_path, text_df, patient_id)

    print('All done!')


