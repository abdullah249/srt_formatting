FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY recursive_srt_formatter.py .
COPY templates/ templates/
RUN mkdir -p /app/uploads /app/outputs /tmp
RUN chmod -R 777 /tmp /app/uploads /app/outputs
EXPOSE 8080
CMD ["python", "app.py"]
