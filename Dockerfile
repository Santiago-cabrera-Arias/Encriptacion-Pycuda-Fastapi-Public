FROM nvidia/cuda:12.1.1-devel-ubuntu22.04


RUN apt-get update && \
    apt-get install -y build-essential python3-pip && \
    pip3 install --upgrade pip && apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0
    
RUN pip3 install pycuda fastapi numpy opencv-python jinja2 python-multipart uvicorn

COPY . /app
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
