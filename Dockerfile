FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir \
    onnxruntime \
    numpy \
    pillow

COPY lambda_function.py .
COPY ./artefacts/models/onnx/model.onnx .
COPY ./artefacts/models/onnx/model.onnx.data .

CMD ["lambda_function.lambda_handler"]