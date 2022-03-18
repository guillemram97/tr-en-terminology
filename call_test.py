from translate_interactive import TranslationInter
from models.terminology.terminologyList import tr_en_dict

print("Translating tr-en", '*' * 100)
translator = TranslationInter('../', 'models', '../sentence.term.bpe.model', 'en_XX', 'tr_TR', terminologyList=tr_en_dict, multi=False)
test = open('/fs/alvis0/guillem/bbc/data/en-tr')
for sent in test.readlines():
    report = translator.translate(sent)
    file_tag = open('results/translations.txt', 'a+')
    file_tra = open('results/tags.txt', 'a+')
    report = translator.translate(sent)
    print(report["tagged"])
    print(report["translation"])
    #file_tag.write(report["tagged"]+'\n')
    #file_tag.write('\n')
    #file_tra.write(report["translation"]+'\n')
    #file_tra.write('\n')
    #file_tra.close()
    #file_tag.close()
