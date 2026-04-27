FROM python:3.12-slim
WORKDIR /app
COPY clean_requirements.txt .
RUN pip install --no-cache-dir -r clean_requirements.txt
#COPY run_transform.py .
#CMD ["python", "run_transform.py"]
COPY main_program.py .
CMD ["python", "main_program.py"]