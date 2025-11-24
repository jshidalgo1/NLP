Act as a Senior Python Engineer and Computational Linguist specializing in Philippine languages. 

I need you to build a "Speech-to-Baybayin" web application called "Tinig-Baybayin". 
The goal is to record a user's voice (Tagalog), transcribe it to text, and then transliterate that text into the ancient Baybayin script (Unicode).

**Tech Stack:**
1. Python 3.9+
2. Streamlit (for the Web UI)
3. OpenAI Whisper (Local installation) for ASR (Automatic Speech Recognition)
4. Regex (re) for syllabic tokenization

**Core Requirements:**

**Step 1: The Transliteration Engine (`baybayin.py`)**
Do not do simple character replacement. You must implement a Syllabic Tokenizer because Baybayin is an Abugida. 
Follow these logic rules strictly:
1. **Mapping:** create a dictionary mapping Tagalog syllables to Baybayin Unicode (Tagalog Block U+1700 to U+171F).
   - Example: "Ba" -> ᜊ, "Bi/Be" -> ᜊᜒ, "Bu/Bo" -> ᜊᜓ.
2. **Standalone Vowels:** Map 'a', 'e/i', 'o/u' to their vowel characters (ᜀ, ᜁ, ᜂ).
3. **Standalone Consonants (Coda):** If a consonant appears without a vowel (like the 'k' in "ok"), use the 'Virama' (Spanisher cross kudlit or pamudpod) to cancel the vowel sound. 
   - Example: 'k' -> ᜃ + ᜔ (Virama).
4. **Normalization:** Before processing, convert the text to lowercase and replace foreign letters with Tagalog equivalents (c->k, j->dy, f->p, v->b, x->ks, z->s).

**Step 2: The Application Logic (`app.py`)**
1. Use `streamlit` to create a clean UI.
2. Use `streamlit-audio-recorder` or standard file uploader to get audio input.
3. Load the `openai-whisper` model (use the "base" model for speed). 
4. Transcribe audio with `model.transcribe(audio, language='tl')`.
5. Pass the transcribed text to the `baybayin.py` engine.
6. Display:
   - The Transcribed Text (Latin Alphabet).
   - The Baybayin Output (Large font size).

**Deliverables:**
Please provide the full code for:
1. `requirements.txt` (include torch, openai-whisper, streamlit, streamlit-audio-recorder).
2. `baybayin.py` (The logic engine).
3. `app.py` (The frontend).

**Important Constraint:** Ensure the code handles errors gracefully (e.g., if no audio is recorded). Add comments explaining the Baybayin Unicode mapping used.