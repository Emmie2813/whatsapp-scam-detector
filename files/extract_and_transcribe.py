import os
import zipfile
import pandas as pd
import re
from datetime import datetime
import mimetypes
import whisper

def extract_zips(zip_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for zip_file in os.listdir(zip_folder):
        if zip_file.endswith('.zip'):
            zip_path = os.path.join(zip_folder, zip_file)
            extract_path = os.path.join(output_folder, os.path.splitext(zip_file)[0])
            if not os.path.exists(extract_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"Extracted: {zip_file}")
                except Exception as e:
                    print(f"Failed to extract {zip_file}: {e}")

def parse_chats(chat_root):
    all_dfs = []
    for chat_folder in os.listdir(chat_root):
        folder_path = os.path.join(chat_root, chat_folder)
        if not os.path.isdir(folder_path):
            continue
        chat_txts = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not chat_txts:
            continue
        chat_file = os.path.join(folder_path, chat_txts[0])
        data = []
        with open(chat_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                match = re.match(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s-\s(.+?):\s(.+)', line)
                if match:
                    date, time, sender, message = match.groups()
                    try:
                        timestamp = datetime.strptime(date + ' ' + time, '%d/%m/%Y %I:%M %p')
                    except Exception:
                        try:
                            timestamp = datetime.strptime(date + ' ' + time, '%m/%d/%y %I:%M %p')
                        except Exception:
                            try:
                                timestamp = datetime.strptime(date + ' ' + time, '%d-%m-%Y %H:%M')
                            except Exception:
                                timestamp = None
                    data.append([timestamp, sender, message, chat_folder])
        if data:
            df = pd.DataFrame(data, columns=['Timestamp', 'Sender', 'Message', 'Chat Folder'])
            all_dfs.append(df)
    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
    else:
        master = pd.DataFrame(columns=['Timestamp', 'Sender', 'Message', 'Chat Folder'])
    return master

def transcribe_folder_audios(chat_root, output_csv='Output/Audios/audio_transcriptions.csv', model_name="base"):
    model = whisper.load_model(model_name)
    records = []
    for chat_folder in os.listdir(chat_root):
        folder_path = os.path.join(chat_root, chat_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.opus', '.ogg', '.mp3', '.wav', '.m4a')):
                audio_path = os.path.join(folder_path, file)
                try:
                    result = model.transcribe(audio_path)
                    trans = result['text'].strip()
                    records.append({'Chat Folder': chat_folder, 'Audio File': file, 'Transcription': trans})
                    print(f"Transcribed {audio_path}")
                except Exception as e:
                    print(f"Failed to transcribe {audio_path}: {e}")
    if records:
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Audio transcriptions saved to {output_csv}")
    else:
        print("No audio files found for transcription.")

def merge_text_and_audio(text_csv, audio_csv, output_csv):
    df_text = pd.read_csv(text_csv, encoding='utf-8-sig')
    df_audio = pd.read_csv(audio_csv, encoding='utf-8-sig') if os.path.exists(audio_csv) else pd.DataFrame(columns=['Chat Folder', 'Audio File', 'Transcription'])
    if not df_audio.empty:
        df_audio = df_audio.rename(columns={'Transcription': 'Message', 'Audio File': 'Media Path'})
        df_audio['Type'] = 'Audio'
    if not df_text.empty:
        df_text['Type'] = 'Text'
        df_text['Media Path'] = None
    merged = pd.concat([df_text, df_audio], ignore_index=True, sort=False)
    merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Merged dataset saved to {output_csv}")

def main():
    zip_folder = 'Data set'
    unzipped_folder = 'Unzipped data set'
    os.makedirs('Output/CSVs', exist_ok=True)
    os.makedirs('Output/Audios', exist_ok=True)

    extract_zips(zip_folder, unzipped_folder)
    df_text = parse_chats(unzipped_folder)
    text_csv = 'Output/CSVs/whatsapp_text_data_set.csv'
    df_text.to_csv(text_csv, index=False, encoding='utf-8-sig')
    print(f"Text chat dataset saved to {text_csv}")
    transcribe_folder_audios(unzipped_folder, 'Output/Audios/audio_transcriptions.csv')
    merge_text_and_audio(text_csv, 'Output/Audios/audio_transcriptions.csv', 'Output/CSVs/bi_modal_dataset.csv')

if __name__ == '__main__':
    main()