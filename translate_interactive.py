import os
import time
import subprocess
import sentencepiece as spm
import nltk
import spacy
import zeyrek

import warnings #treure!
warnings.filterwarnings("ignore") #treure aixo

from fairseq.models.transformer import TransformerModel

#Wilker's splitter from NTMI course
def naive_splitter(text, delimiters="."):
    sentences = []
    start = 0
    for i, ch in enumerate(text): # scan the string
        if ch in delimiters or i + 1 == len(text): # looking for a delimiter or the end of the string
            sentence = text[start:i + 1].strip() # we've found a "sentence" (as far as our delimiters suggest)
            if sentence:
                sentences.append(sentence)
            start = i + 1
    return sentences

#taken from AEVNMT data/datasets.py
def split_list(x, size):
    output = []
    if size < 0:
        return [x]
    elif size == 0:
        raise ValueError("Use size -1 for no splitting or size more than 0.")
    while True:
        if len(x) > size:
            output.append(x[:size])
            x = x[size:]
        else:
            output.append(x)
            break
    return output

class TranslationInter:
    def __init__(self, bin_path, model_path, spm_path, src, tgt, multi=False):
        self.bin_path = bin_path
        self.model_path = model_path
        self.spm_path = spm_path
        self.src = src
        self.tgt = tgt
        self.report = {}
        self.max_len = 50
        # note that bin_path is relative to model_path see source of .from_pretrained (https://github.com/pytorch/fairseq/blob/master/fairseq/models/fairseq_model.py)
        self.model = TransformerModel.from_pretrained(self.model_path,
                                                      checkpoint_file='checkpoint_best.pt',
                                                      data_name_or_path=self.bin_path,
                                                      bpe='sentencepiece',
                                                      sentencepiece_model=self.spm_path,
                                                      lang_dict='langs.file',
                                                      source_lang=self.src,
                                                      target_lang=self.tgt,
                                                      )
        self.tgt_idx = self.model.tgt_dict.index("[{}]".format(self.tgt))
        if (self.src == 'en_XX') or (self.src == 'en'):
            self.en_sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        if (self.src =='tr_TR') or (self.src =='tr_TR'):
            self.tr_sent_detector = nltk.data.load('tokenizers/punkt/turkish.pickle')

    def custom_translate(self, sentence):
        src_bpe = self.model.apply_bpe(sentence)
        src_bin = self.model.binarize(src_bpe)
        tgt_bin = self.model.generate(src_bin, beam=5)
        tgt_sample = tgt_bin[0]['tokens']
        #mbart outputs lang id removes this.
        if self.tgt_idx in tgt_sample:
            tgt_sample = tgt_sample[tgt_sample != self.tgt_idx]
        tgt_bpe = self.model.string(tgt_sample)
        tgt_toks = self.model.remove_bpe(tgt_bpe)
        tgt = self.model.detokenize(tgt_toks)
        return tgt
     #we need to install spacy and zeyrek
     #python -m spacy download en   
    def tag_term(self, sen, terminologyList, nlp_lemma):
        if len(terminologyList)==0: return sen
        new_sen=''
        for word in sen.split(' '):
            if word in terminologyList.keys():
                if self.src=='en_XX':
                    new_sen=new_sen+"<0> "+ word +" <1> "+nlp_lemma.analyze(terminologyList[word])[0][0].lemma+" <2> "   
                elif self.src=='tr_TR':
                    new_sen=new_sen+"<0> "+ word +" <1> "+nlp_lemma(terminologyList[word])[0].lemma_+" <2> "
            else:
               new_sen=new_sen+word+' '
        return new_sen[:-1]

    def translate(self, sentence, terminologyList={}):
        self.model.eval()
        translations = []
        t0 = time.time()
        processed_sents = []
        if self.src == 'en_XX':
            sentences = self.en_sent_detector.tokenize(sentence.strip())
        if self.src == 'tr_TR':
            #sentences = naive_splitter(sentence.strip())
            sentences = self.tr_sent_detector.tokenize(sentence.strip())
        if self.tgt=='en_XX':
            nlp_lemma=spacy.load("en_core_web_sm")
        elif self.tgt=='tr_TR':
            nlp_lemma=zeyrek.MorphAnalyzer()    
        for sent in sentences:
            sent=self.tag_term(sent, terminologyList, nlp_lemma)
            split_sent = sent.split(" ")
            token_count = len(split_sent)
            if token_count <= self.max_len:
                processed_sents.append(sent)
            else:
                for snt in split_list(split_sent, self.max_len):
                    processed_sents.append(" ".join(snt))
        dt0 = time.time() - t0
        self.report['preprocessing'] = dt0

        t1 = time.time()
        for idx, sent in enumerate(processed_sents):
            #translation = self.model.translate(sent, verbose=False)
            translation = self.custom_translate(sent)
            translations.append(translation)
        dt1 = time.time() - t1
        self.report['translating'] = dt1

        t2 = time.time()
        final = " ".join(translations)
        dt2 = time.time() - t2
        self.report['postprocessing'] = dt2
        self.report['translation'] = final
        return self.report

print("Translating tr-en", '*' * 100)
translator = TranslationInter('../', 'models', '../sentence.term.bpe.model', 'tr_TR', 'en_XX', multi=False)
sent = "Hükümetin kendisi hakkındake suçlamalarını. Soros ile ilgili ortaya <0> atılan <1> BB <2> iddiaları bir kez daha reddeden Kavala, siyasi gerekçelerle tutuklu bulunduğunu belirterek, bunun Avrupa İnsan Hakları Mahkemesi (AİHM) kararında da ortaya koyulduğunu vurguladı."
report = translator.translate(sent)
print(report)
#print("Translating tr-en", '*' * 100)
#translator = TranslationInter('../../data/bin', '../ckpts/en-tr', '../data/mbart.cc25.v2/sentence.bpe.model', 'en_XX', 'tr_TR', multi=True)
#sent = "If they ask you the wrong questions, there are no right answers."
#report = translator.translate(sent)
#print(report)


