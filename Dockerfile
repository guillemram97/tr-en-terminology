
FROM python:3.8
# set up working directory
COPY . .
#RUN apk add --no-cache --virtual .build-deps gcc musl-dev \
#    && pip install --no-cache-dir -r /code/requirements.txt \
#    && apk del .build-deps
RUN python3 -m pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 download_nltk.py
CMD [ "python3", "app.py", "./", "term_model/", "../sentence.term.bpe.model", "tr_TR", "en_XX" ]
