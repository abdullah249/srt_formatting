FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY recursive_srt_formatter.py .

# Create directories for input and output SRT files
RUN mkdir -p /app/input_srt_files /app/output_srt_files /tmp
RUN chmod -R 777 /tmp /app/input_srt_files /app/output_srt_files

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Set the entry point to run the Flask app
CMD ["python", "app.py"]
