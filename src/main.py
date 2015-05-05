import json
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import *
import codecs


all_tags = {}
max_num = 1

def read_annotated_file():
    with open('annotation_train.json') as data_file:    
        data = json.load(data_file)
    return data

def full_capital(w):
    return w.isupper()
    
def starts_with_capital(w):
    return w[0].isupper()

def isinPositiveExample(w):
    annotations = read_annotated_file()
    
    for a in annotations:
        try:  
            for t in annotations[a]['pref']:
                if w==t:
                    return True
            
            for t in annotations[a]['req']: 
                if w==t:
                    return True
        except:
            pass
    
    return False

def contains_special_character(w):
    set = ['.', '/', '#', '+']
    return 1 in [c in w for c in set]

def sentence_contains(w, line, kw):
    
    if kw in line.lower() and w!=kw:
        return True
    else:
        return False
    
def get_pos_tag(w,line, pos_tags):
    
    
    for wd in pos_tags:
        if w == wd[0]:
            return convert_tag_to_num(wd[1])
        

def get_pos_tag_before(w,line, pos_tags):
    
    i = 0
    for i in range(len(pos_tags)):
        if pos_tags[i][0]==w:
            break
    try:
        return convert_tag_to_num(pos_tags[i-1][1])
    except:
        return convert_tag_to_num('')
            
def get_pos_tag_after(w,line, pos_tags):
    
    i = 0
    for i in range(len(pos_tags)):
        if pos_tags[i][0]==w:
            break
    try:
        return convert_tag_to_num(pos_tags[i+1][1])
    except:
        return convert_tag_to_num('')
    

def convert_tag_to_num(tag):
    
    global max_num
    global all_tags
    
    if tag in all_tags:
        return all_tags[tag]
    else:
        all_tags[tag]=max_num
        max_num = max_num+1
        return all_tags[tag]
        
        
def create_features(file_name):
     
    tagged_sentences = treebank.tagged_sents()[0:1000]
    training_data = tagged_sentences[0:500]
    tagger0 = nltk.DefaultTagger('NN')
    tagger1 = nltk.UnigramTagger(training_data, backoff=tagger0)
    tagger2 = nltk.BigramTagger(training_data, backoff=tagger1)
    
    with codecs.open(file_name, encoding='utf-8') as data_file:
        content = data_file.readlines()
        
    training_features = []  
    for line in content:
        words = word_tokenize(line)
        for w in words:
            features = []
            
            features.append(full_capital(w))
            features.append(starts_with_capital(w))
            features.append(contains_special_character(w))
            features.append(contains_special_character(w))
            features.append(sentence_contains(w, line, 'experience'))
            features.append(sentence_contains(w, line, 'technologies'))
            features.append(sentence_contains(w, line, 'technology'))
            features.append(sentence_contains(w, line, 'language'))
            features.append(sentence_contains(w, line, 'languages'))
            features.append(sentence_contains(w, line, 'platforms'))
            features.append(sentence_contains(w, line, 'knowledge'))
            features.append(sentence_contains(w, line, 'tools'))
            features.append(sentence_contains(w, line, 'protocols'))
            list_tokens = word_tokenize(line)
            features.append(get_pos_tag(w, line, tagger2.tag(list_tokens)))
            features.append(get_pos_tag_before(w, line,  tagger2.tag(list_tokens)))
            features.append(get_pos_tag_after(w, line, tagger2.tag(list_tokens)))
            
            training_features.append(features)
    
    return training_features

        
def learn_features(file_name):
     
    tagged_sentences = treebank.tagged_sents()[0:1000]
    
    training_data = tagged_sentences[0:500]
    tagger0 = nltk.DefaultTagger('NN')
    tagger1 = nltk.UnigramTagger(training_data, backoff=tagger0)
    tagger2 = nltk.BigramTagger(training_data, backoff=tagger1)

    
    with codecs.open(file_name, encoding='utf-8') as data_file:
        content = data_file.readlines()
    
    labels = []  
    training_features = [] 
    
    cont = True
    
    for line in content:
        words = word_tokenize(line)
        for w in words:
        
            features = []

            features.append(full_capital(w))
            features.append(starts_with_capital(w))
            features.append(contains_special_character(w))
            features.append(contains_special_character(w))
            features.append(sentence_contains(w, line, 'experience'))
            features.append(sentence_contains(w, line, 'technologies'))
            features.append(sentence_contains(w, line, 'technology'))
            features.append(sentence_contains(w, line, 'language'))
            features.append(sentence_contains(w, line, 'languages'))
            features.append(sentence_contains(w, line, 'platforms'))
            features.append(sentence_contains(w, line, 'knowledge'))
            features.append(sentence_contains(w, line, 'tools'))
            features.append(sentence_contains(w, line, 'protocols'))
            
            list_tokens = word_tokenize(line)
            
            #features.append(get_pos_tag(w, line, tagger2.tag(list_tokens)))
            features.append(get_pos_tag_before(w, line,  tagger2.tag(list_tokens)))
            features.append(get_pos_tag_after(w, line, tagger2.tag(list_tokens)))

         
            get_true_labels = labels.count(False)
             
            if get_true_labels>50:
                cont = False
             
            if isinPositiveExample(w):
                labels.append(True)
                training_features.append(features)
            elif cont:
                labels.append(False)
                training_features.append(features)
                
    
    return training_features, labels


def main():
    annotations = read_annotated_file()
    
    print annotations
    
    total_cnt = 0
    cnt_fully_capital = 0
    cnt_starts_with_capital = 0
    
    for a in annotations:
        try:
            total_cnt = total_cnt + len(annotations[a]['pref'])
            
            for t in annotations[a]['pref']:
                if full_capital(t):
                    cnt_fully_capital = cnt_fully_capital+1
                    
                if starts_with_capital(t):
                    cnt_starts_with_capital = cnt_starts_with_capital+1
            
        except:
            pass
            
        try:
            total_cnt = total_cnt + len(annotations[a]['req'])
            
            for t in annotations[a]['req']:
                if full_capital(t):
                    cnt_fully_capital = cnt_fully_capital+1
                    
                if starts_with_capital(t):
                    cnt_starts_with_capital = cnt_starts_with_capital+1
                
        except:
            pass
    
    print "total count:"+str(total_cnt)
    print "fully capital count:"+str(cnt_fully_capital)
    print "first letter capital count:"+str(cnt_starts_with_capital)
    

def get_words_list(file_name):
    
    corpus_dir = 'NLP_dataset/training_set/'+file_name  
    with codecs.open(corpus_dir, 'r', encoding='utf-8') as f:
        tokens = word_tokenize(f.read())
        
    return tokens


# main function
#main()

training_data, labels = learn_features('positive_samples.txt')

print training_data
print labels

clf = svm.SVC()
clf.fit(training_data, labels)
predicted_labels = clf.predict(training_data)
test_data = create_features('NLP_dataset/training_set/train_1_JuniperNetworks.txt')
predicted_labels = clf.predict(training_data)
ii = np.where(predicted_labels == True)[0]
 
words = get_words_list('train_1_JuniperNetworks.txt')
 
for i in ii:
    print words[i]
print words

print(classification_report(predicted_labels, labels))
