FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY recursive_srt_formatter.py .

# Create directories for input and output SRT files
RUN mkdir -p input_srt_files output_srt_files

# Set the entry point
ENTRYPOINT ["python", "recursive_srt_formatter.py"] 
