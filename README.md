# SRT Formatter Web App

This is a simple web frontend for the Recursive SRT Formatter.

## Features
- Upload an SRT file
- Automatically formats subtitles (max 45 chars/line, etc.)
- Download the reformatted SRT file

## Setup
1. Make sure `recursive_srt_formatter.py` is in the parent directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Notes
- Only `.srt` files are supported.
- Output files are saved in the `outputs/` directory. 