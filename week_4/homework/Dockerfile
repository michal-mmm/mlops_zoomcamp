FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv
RUN mkdir ./results

COPY [ "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY [  "starter.py", "./" ]

ENTRYPOINT [ "python", "starter.py" ]
