import streamlit as st
import whisper
import numpy as np
import os
import tempfile
from baybayin import to_baybayin
import imageio_ffmpeg as iio_ffmpeg

# Page Configuration
st.set_page_config(
    page_title="Tinig-Baybayin",
    page_icon="ᜊ",
    layout="centered"
)

# Custom CSS for Baybayin font size
st.markdown("""
    <style>
    .baybayin-text {
        font-size: 60px;
        font-family: 'Noto Sans Tagalog', sans-serif;
        line-height: 1.5;
        text-align: center;
        margin: 20px 0;
        color: #E63946;
    }
    .transcribed-text {
        font-size: 24px;
        font-style: italic;
        text-align: center;
        color: #457B9D;
        margin-bottom: 10px;
    }
    .main-header {
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the Whisper model (cached to avoid reloading)"""
    # Debug: Check if ffmpeg is available
    import shutil
    import subprocess
    
    # Use bundled ffmpeg from imageio_ffmpeg if system ffmpeg is missing
    try:
        ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
        st.write(f"DEBUG: imageio_ffmpeg.get_ffmpeg_exe() returned: `{ffmpeg_path}`")
        
        if os.path.exists(ffmpeg_path):
            st.success(f"DEBUG: File exists at `{ffmpeg_path}`")
            st.write(f"DEBUG: File permissions: `{oct(os.stat(ffmpeg_path).st_mode)[-3:]}`")
            # Ensure it's executable
            os.chmod(ffmpeg_path, 0o755)
        else:
            st.error(f"DEBUG: File DOES NOT EXIST at `{ffmpeg_path}`")

        # Add the bundled binary to PATH temporarily
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = f"{ffmpeg_dir}:{os.environ.get('PATH','')}"
        st.write(f"DEBUG: Updated PATH with `{ffmpeg_dir}`")
        
    except Exception as e:
        st.error(f"DEBUG: Error setting up bundled ffmpeg: {e}")

    if not shutil.which("ffmpeg"):
        st.error("FFmpeg not found! Please ensure ffmpeg is installed.")
        st.write("DEBUG: shutil.which('ffmpeg') returned None")
    else:
        st.success(f"System/Bundled FFmpeg found at: `{shutil.which('ffmpeg')}`")
        # Try running it to see if it works
        try:
            version_output = subprocess.check_output(["ffmpeg", "-version"]).decode().splitlines()[0]
            st.write(f"DEBUG: ffmpeg version: `{version_output}`")
        except Exception as e:
            st.error(f"DEBUG: Failed to run ffmpeg: {e}")
    print("Loading Whisper model...")
    return whisper.load_model("base")

def save_audio_file(audio_bytes, file_extension="wav"):
    """Save audio bytes to a temporary file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name

def main():
    # Header
    st.markdown("<h1 class='main-header'>ᜆᜒᜈᜒᜄ᜔-ᜊᜌ᜔ᜊᜌᜒᜈ᜔</h1>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-header'>Tinig-Baybayin</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Speak in Tagalog and see it transformed into the ancient Baybayin script.</p>", unsafe_allow_html=True)
    
    st.divider()

    # Load Model
    with st.spinner("Loading speech recognition model..."):
        model = load_model()

    # Input Method Selection
    tab1, tab2 = st.tabs(["🎙️ Record Audio", "📂 Upload File"])

    audio_file_path = None

    with tab1:
        st.write("Click the microphone to start recording:")
        # Use native Streamlit audio input (requires Streamlit 1.40+)
        audio_value = st.audio_input("Record")
        
        if audio_value:
            st.audio(audio_value)
            # Read bytes from the UploadedFile-like object
            audio_bytes = audio_value.read()
            audio_file_path = save_audio_file(audio_bytes)

    with tab2:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            # Read bytes and save
            audio_bytes = uploaded_file.read()
            audio_file_path = save_audio_file(audio_bytes, file_extension=uploaded_file.name.split('.')[-1])

    # Processing
    if audio_file_path:
        st.divider()
        st.subheader("Processing...")
        
        try:
            with st.spinner("Transcribing audio..."):
                # Transcribe (fp16=False to avoid CPU warnings)
                result = model.transcribe(audio_file_path, language='tl', fp16=False)
                transcribed_text = result["text"]
            
            # Remove temp file
            os.unlink(audio_file_path)
            
            if not transcribed_text.strip():
                st.warning("No speech detected. Please try again.")
            else:
                # Transliterate
                baybayin_text = to_baybayin(transcribed_text)
                
                # Display Results
                st.subheader("Result:")
                
                st.markdown(f"<div class='transcribed-text'>\"{transcribed_text}\"</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='baybayin-text'>{baybayin_text}</div>", unsafe_allow_html=True)
                
                # Copy button (using code block as a workaround for simple copy)
                st.caption("Copy Baybayin text:")
                st.code(baybayin_text, language=None)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 12px;'>
        Powered by OpenAI Whisper & Custom Baybayin Engine
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
