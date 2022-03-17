from translate_interactive import TranslationInter
from models.terminology.terminologyList import tr_en_dict

print("Translating tr-en", '*' * 100)
translator = TranslationInter('../', 'models', '../sentence.term.bpe.model', 'tr_TR', 'en_XX', terminologyList=tr_en_dict, multi=False)
test = open('/fs/alvis0/guillem/bbc/data/tr-en')
for sent in test.readlines():
    report = translator.translate(sent)
    file_tag = open('results/translations.txt', 'a+')
    file_tra = open('results/tags.txt', 'a+')
    report = translator.translate(sent)
    file_tag.write(report["tagged"]+'\n')
    file_tag.write('\n')
    file_tra.write(report["translation"]+'\n')
    file_tra.write('\n')
    file_tra.close()
    file_tag.close()
