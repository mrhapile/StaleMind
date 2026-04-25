FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD bash -c "uvicorn main:app --host 0.0.0.0 --port 8000 & sleep 5 && python app.py"