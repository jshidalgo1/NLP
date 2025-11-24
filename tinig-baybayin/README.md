# Tinig-Baybayin: Speech-to-Baybayin Converter

A web application that records Tagalog speech, transcribes it using OpenAI Whisper, and converts the text into the ancient Baybayin script.

## Features

- 🎙️ **Speech-to-Text**: Uses OpenAI's Whisper model to accurately transcribe Tagalog speech.
- 📝 **Baybayin Transliteration**: Custom engine that converts Tagalog text to Baybayin script using proper syllabic tokenization and Unicode characters.
- 🖥️ **Modern UI**: Built with Streamlit for a clean and responsive user interface.
- ⏺️ **Audio Recording**: Integrated audio recorder for real-time usage.

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install system dependencies**:
   - **macOS** (via Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```
   - **Windows**: Install via [ffmpeg.org](https://ffmpeg.org/download.html) or `choco install ffmpeg`.

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Using the App**:
   - Allow microphone access when prompted.
   - Click the microphone icon to start recording.
   - Speak in Tagalog.
   - Click stop to finish recording.
   - The app will automatically transcribe your speech and display the Baybayin translation.
   - Alternatively, you can upload an existing audio file (`.wav`, `.mp3`).

## Technical Details

### Baybayin Transliteration
The engine (`baybayin.py`) implements a syllabic tokenizer that respects the Abugida nature of the script:
- **CV Pattern**: Consonant + Vowel (e.g., "Ba" → ᜊ)
- **Standalone Vowels**: Maps 'a', 'e/i', 'o/u' (e.g., "Ako" → ᜀᜃᜓ)
- **Kudlit**: Diacritical marks for 'i/e' and 'o/u' sounds.
- **Virama/Pamudpod**: Cancels the inherent vowel for standalone consonants (e.g., "ok" → ᜂᜃ᜔).

### Models
- **ASR**: OpenAI Whisper (`base` model) is used for speech recognition. It downloads automatically on the first run (~140MB).

## Requirements
- Python 3.9+
- Internet connection (for first-time model download)
