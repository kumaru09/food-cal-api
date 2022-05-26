# FROM python:3.7.12-slim-buster as builder
# RUN apt-get update \
#         && apt-get install ffmpeg libsm6 libxext6  -y \
#         && apt-get clean

# RUN python3 -m venv /opt/venv
# # Make sure we use the virtualenv:
# ENV PATH="/opt/venv/bin:$PATH"

# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY . .

# FROM python:3.7.12-slim-buster as app
# COPY --from=builder /opt/venv /opt/venv

# ENV PATH="/opt/venv/bin:$PATH"
# EXPOSE 8000
# ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--timeout", "600", "app:app"]

FROM python:3.7.12-slim-buster

RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 -y \
    && apt-get clean

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt --no-cache-dir
COPY . /app/

EXPOSE 8000

#CMD ["python", "/code/manage.py", "runserver", "0.0.0.0:8000"]
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--timeout", "600", "--workers", "2", "app:app"]