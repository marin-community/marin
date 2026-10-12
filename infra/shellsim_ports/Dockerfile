FROM python:3.12-slim
ARG SOURCE_REVISION
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AWS_EC2_METADATA_DISABLED=true
RUN pip install --no-cache-dir marin-rigging==0.2.165.dev38153482028
ADD --chmod=0444 https://raw.githubusercontent.com/rjpower/shellsim/${SOURCE_REVISION}/ports/buildomatic/s3_proxy.py /app/s3_proxy.py
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python", "/app/s3_proxy.py", "--bind", "0.0.0.0", "--port", "8080"]
