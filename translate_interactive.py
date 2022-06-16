import os
import time
import subprocess
import sentencepiece as spm
import nltk
import spacy
import warnings #treure
from TurkishStemmer import TurkishStemmer
warnings.filterwarnings("ignore") 
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
        self.terminologyList = {}
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

    def set_terminologyList(self, terminologyList):
        self.terminologyList=terminologyList
        return
    def custom_translate(self, sentences):
            src_bpe = [self.model.apply_bpe(sentence) for sentence in sentences]
            src_bin = [self.model.binarize(x) for x in src_bpe]
            tgt_bin = self.model.generate(src_bin, beam=5) # batching happens here
            tgt_sample = [x[0]['tokens'] for x in tgt_bin]
            #mbart outputs lang id removes this.
            tgt_sample = [x[x != self.tgt_idx] if self.tgt_idx in x else x for x in tgt_sample]
            tgt_bpe = [self.model.string(x) for x in tgt_sample]
            tgt_toks = [self.model.remove_bpe(x) for x in tgt_bpe]
            tgt = [self.model.detokenize(x) for x in tgt_toks]
            return tgt

#we need to install spacy and zeyrek
#python -m spacy download en   
    def limits_word(self, sen, i, up):
            lower=i
            upper=up+i-1
            while lower>0 and sen[lower]!=' ': lower-=1
            while upper<len(sen) and sen[upper] not in [' ', '.', ',', '?', '!', '-']: upper+=1 
            if sen[lower]==' ': lower+=1
            return [lower, upper]

    def tag_term(self, sen):
        if self.src=='tr_TR': stemmer = TurkishStemmer()
        if self.src=='en_XX': nlp_lemma=spacy.load("en_core_web_sm")
        aux_words=''
        dic_aux=[]
        sen_sep=sen.split(' ')
        tags=[]
        tags_to=[]
        lemmas=[]
        lemmas_orig=[]
        for idx, word in enumerate(sen_sep):
            tags.append(-1)
            for idx, word in enumerate(sen_sep):
                tags.append(-1)
                if self.src=='tr_TR':
                    aux_lemma=stemmer.stem(word)
                    lemmas.append(aux_lemma) 
                    lemmas_orig.append(idx)
                elif self.src=='en_XX':
                    aux_lemma=nlp_lemma(word)
                    for i in aux_lemma:
                        lemmas.append(i.lemma_) 
                    for el in aux_lemma: lemmas_orig.append(idx)                        
        lemmas_sep=' '.join([i for i in lemmas if len(i)>0])
        for term in sorted(self.terminologyList.keys(), reverse=True, key=len):
            if len(term)>5 and term[:6]=='PROPN_': 
                aux_term=term[6:]
                x=0
                w_acumm=0
                i=sen[x:].find(aux_term) 
                while i>-1:
                    i+=x
                    limits=self.limits_word(sen, i, len(aux_term))
                    src_lemma=sen[limits[0]:limits[1]]
                    w_acumm+=len(sen[x:x+sen[x:].find(src_lemma)].split(' '))-1
                    aux_orig=[]
                    possible=True
                    for idx in range(len(src_lemma.split(' '))):
                        aux_orig.append(w_acumm+idx)
                        if tags[w_acumm+idx]!=-1: possible=False
                    if possible and limits[0]==i and limits[1]==i+len(aux_term):
                        for ele in aux_orig:
                            tags[ele]=len(tags_to)
                        tags_to.append(self.terminologyList[term])
                    x=limits[1]
                    i=sen[x:].find(aux_term)
                    w_acumm+=len(src_lemma.split(' '))-1
            elif self.src=="tr_TR":
                aux_term=" ".join(term.split(" ")[:-1])
                last_term=term.split(" ")[-1]
                x=0
                w_acumm=0
                if aux_term=='':
                    term=last_term
                    term_2=term
                    i=lemmas_sep[x:].find(term)
                    if i==-1: i=lemmas_sep[x:].find(term)
                    j=-1
                    while i>-1 or j>-1:
                        if i==-1: i=j
                        i+=x
                        limits=self.limits_word(lemmas_sep, i, len(term))            
                        src_lemma=lemmas_sep[limits[0]:limits[1]]
                        w_acumm+=len(lemmas_sep[x:x+lemmas_sep[x:].find(src_lemma)].split(' '))-1
                        aux_orig=[]
                        possible=True
                        for idx in range(len(src_lemma.split(' '))):
                            if w_acumm+idx<len(lemmas_orig): 
                                aux_orig.append(lemmas_orig[w_acumm+idx])
                                if tags[aux_orig[-1]]!=-1: possible=False
                        if possible and limits[0]==i and limits[1]==i+len(term):
                            for ele in aux_orig:
                                tags[ele]=len(tags_to)
                            tags_to.append(self.terminologyList[term])
                        x=limits[1]
                        i=lemmas_sep[x:].find(term)
                        j=lemmas_sep[x:].find(term.capitalize())
                        w_acumm+=len(src_lemma.split(' '))-1
                else:
                    x=0
                    w_acumm=0
                    i=sen[x:].find(aux_term)
                    if i==-1: i=lemmas_sep[x:].find(term.capitalize())
                    j=-1
                    while i>-1 or j>-1:
                        if i==-1: i=j
                        i+=x
                        limits=self.limits_word(sen, i, len(aux_term))
                        src_lemma=sen[limits[0]:limits[1]]
                        w_acumm+=len(sen[x:x+sen[x:].find(src_lemma)].split(' '))-1
                        aux_orig=[]
                        possible=True
                        for idx in range(len(src_lemma.split(' '))):
                            aux_orig.append(w_acumm+idx)
                            if tags[w_acumm+idx]!=-1: possible=False
                        x=limits[1]
                        w_acumm+=len(src_lemma.split(' '))-1
                        if possible and limits[0]==i and limits[1]==i+len(aux_term):
                            x=limits[1]+1
                            y=sen[x:].find(" ")
                            if stemmer.stem(sen[x:x+y])==last_term:
                                for ele in aux_orig:
                                    tags[ele]=len(tags_to)
                                tags[ele+1]=len(tags_to)
                                tags_to.append(self.terminologyList[term])
                                x+=y
                                w_acumm+=1
                        i=sen[x:].find(aux_term)
                        j=sen[x:].find(aux_term.capitalize())
                        
            elif self.src=="en_XX":
                x=0
                w_acumm=0
                i=lemmas_sep[x:].find(term)
                j=-1
                while i>-1 or j>-1:
                    if i==-1: i=j
                    i+=x
                    limits=self.limits_word(lemmas_sep, i, len(term))            
                    src_lemma=lemmas_sep[limits[0]:limits[1]]
                    w_acumm+=len(lemmas_sep[x:x+lemmas_sep[x:].find(src_lemma)].split(' '))-1
                    aux_orig=[]
                    possible=True
                    for idx in range(len(src_lemma.split(' '))):
                        aux_orig.append(lemmas_orig[w_acumm+idx])
                        if tags[aux_orig[-1]]!=-1: possible=False
                    if possible and limits[0]==i and limits[1]==i+len(term):
                        for ele in aux_orig:
                            tags[ele]=len(tags_to)
                        tags_to.append(self.terminologyList[term])
                    x=limits[1]
                    i=lemmas_sep[x:].find(term)
                    j=lemmas_sep[x:].find(term.capitalize)
                    w_acumm+=len(src_lemma.split(' '))-1
        idx=0
        new_sen=''
        while idx<len(sen_sep):
            if tags[idx]!=-1:
                aux_word=sen_sep[idx]
                tag=tags_to[tags[idx]]
                idx+=1
                while idx<len(sen_sep) and tags[idx-1]==tags[idx]:
                    aux_word=aux_word+' '+sen_sep[idx]
                    idx+=1
                symbol=''
                while aux_word[-1] in [',', '.', '?', '!']:
                        symbol=symbol+aux_word[-1]
                        aux_word=aux_word[:-1]
                new_sen=new_sen+' <0> '+aux_word+' <1> '+tag+' <2>'+symbol
                dic_aux.append([aux_word, tag])
            else: 
                new_sen=new_sen+' '+sen_sep[idx]
                idx+=1
        if new_sen[0]==' ': new_sen=new_sen[1:]
        return new_sen, dic_aux

    def translate(self, sentence):
            self.model.eval()
            translations = []
            t0 = time.time()
            processed_sents = []
            if self.src == 'en_XX':
                sentences = self.en_sent_detector.tokenize(sentence.strip())
            if self.src == 'tr_TR':
                sentences = self.tr_sent_detector.tokenize(sentence.strip())
            dic_auxs=[]
            for sent in sentences:
                sent, dic_aux =self.tag_term(sent)
                dic_auxs.append(dic_aux)
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
            translations = self.custom_translate(processed_sents)

            if self.src == 'en_XX': stemmer = TurkishStemmer()
            else: nlp_lemma=spacy.load("en_core_web_sm")
            for idx, translation in enumerate(translations):
                #translation = self.model.translate(sent, verbose=False)
                dic_aux=dic_auxs[idx]
                for pair in dic_aux:
                    s=translation.find(pair[0]+" "+pair[1])
                    if s>-1: translation=translation[:s]+translation[s+len(pair[0])+1:]
                    r=translation.find(pair[1]+" "+pair[1])
                    if r>-1:  translation=translation[:r]+translation[r+len(pair[1])+1:]
                    if self.src == 'en_XX': aux_word=" ".join([stemmer.stem(word) for word in pair[1].split(" ")])
                    else: aux_word=" ".join([nlp_lemma(word)[0].lemma_ for word in pair[1].split(" ")])
                    r=translation.find(aux_word)
                    aux=translation[:r]+translation[r+len(aux_word)+1:]
                    s=aux.find(aux_word)
                    if (s-r)<5 and r>-1 and s>-1: translation=translation[:r]+translation[len(aux_word)+s+1:]

                translations[idx]=translation


            dt1 = time.time() - t1
            self.report['translating'] = dt1

            t2 = time.time()
            final = " ".join(translations)
            dt2 = time.time() - t2
            self.report['postprocessing'] = dt2
            self.report['translation'] = final
            return self.report

#print("Translating tr-en", '*' * 100)
#translator = TranslationInter('../', 'models', '../sentence.term.bpe.model', 'tr_TR', 'en_XX', terminologyList=tr_en_dict, multi=False)
#sent="Balgam ve öksürüğünü kesmek için balgam söktürücü ilaçlar yazıldı."
#report = translator.translate(sent)
#print(report)





