#include "gtest/gtest.h"

#include "main.h"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
TEST(Test, Test_foofail)
{
<<<<<<< HEAD
	auto i = stoi("123"); //ïðåîáðàçóåò ñòðîêó â ÷èñëî, åñëè 

	if (i)
	{
		auto s = to_string(i);
	}
	std::optional<std::string> opt_string{ "fds" };
	if (opt_string) {
		std::size_t s = opt_string->size();
	}


	optional<string> o{ "234" };

	std::optional<int> io = o.and_then(stoi);

	//monadic_optional<string> opt{"123"};
	//


	//EXPECT_EQ(r, 2);
}

//TEST(Test, Test_sort1)
//{
//	list<int> a{ 1, 3, 2 };
//	sort(a);
//	//EXPECT_EQ(sort(a), {1, 2, 3});
//}
=======
	std::optional<int> opt; 
	if (opt) //ÑÑ‚Ñƒ ÑˆÑ‚ÑƒÐºÑƒ ÑÐ¾ÐºÑ€Ð°Ñ‰Ð°ÑŽ Ð½Ð¸Ð¶Ðµ
	{
		std::optional<int> b = cat(*opt); //Ð•ÑÐ»Ð¸ opt Ð½Ðµ False, Ñ‚Ð¾ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ Ð² ÐºÐ°Ñ‡ÐµÑÑ‚Ð²Ðµ Ð°Ñ€Ð³ÑƒÐ¼ÐµÐ½Ñ‚Ð° Ñ„ÑƒÐ½ÐºÑ†Ð¸Ð¸ cat
	}

	monadic_optional<int> opt2; //ÐžÐ±ÑŠÐµÐºÑ‚, Ñƒ ÐºÐ¾Ñ‚Ð¾Ñ€Ð¾Ð³Ð¾ ÐµÑÑ‚ÑŒ Ð¼ÐµÑ‚Ð¾Ð´ and_then
	monadic_optional<int> t = opt2.and_then(cat); //Ð—Ð´ÐµÑÑŒ Ð¾ÑˆÐ¸Ð±ÐºÐ°, error_type Ð´Ð»Ñ cat, Ð¸ Ð´Ð»Ñ cat Ð½Ðµ ÑƒÐ´Ð°ÐµÑ‚ÑÑ Ð¾Ð¿Ñ€ÐµÐ´ÐµÐ»Ð¸Ñ‚ÑŒ ÑˆÐ°Ð±Ð»Ð¾Ð½
	
	
	
	
	//EXPECT_EQ(b.foo(10), 10);
}
>>>>>>> b07b037b92d0356dfa87ecbf60b0da3f652e1e8e



from stop_words import get_stop_words
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import doc2vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pylab
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing
from multiprocessing import Process, freeze_support
import time
from progress.bar import IncrementalBar

"""
full_sport = []
full_business = []


for k in range(1, 1876):
    try:
        f = open('articles/sport/sport ({id}).txt'.format(id=k), 'r')
        full_sport.append(f.read())
        f.close()
    except:
        error = 1
target_sport = [0 for i in range(len(full_sport))]


for k in range(1, 1666):
    try:
        f = open('articles/business/business ({id}).txt'.format(id=k), 'r')
        full_business.append(f.read())
        f.close()
    except:
        error = 1

target_business = [1 for i in range(len(full_business))]


full_articles = full_sport + full_business
full_target = target_sport + target_business

# почистить данные
stops = list(get_stop_words('en'))

def clean_doc(myw, stop):
    final = []
    myw = myw.split()
    for w in myw:
        if w.lower() not in stop:
            final.append(w)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([c for c in final if not c.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    while "\n" in final:
        final = final.replace("\n", " ")
    return final

for i in range(len(full_articles)):
    full_articles[i] = clean_doc(full_articles[i], stops)




model_doc2vec = doc2vec.Doc2Vec.load("my_doc2vec")
"""
# ОБУЧИТЬ НА ВИКИПЕДИИ ************************************************************************************************

if __name__ == '__main__':
    wiki = WikiCorpus("enwiki-20210501-pages-articles-multistream.xml.bz2")
    class TaggedWikiDocument(object):
        def __init__(self, wiki):
            self.wiki = wiki
            self.wiki.metadata = True
        def __iter__(self):
            bar = IncrementalBar('Countdown', max=len(self.wiki.get_texts()))
            for content, (page_id, title) in self.wiki.get_texts():
                bar.next()
                yield TaggedDocument([c.decode("utf-8") for c in content], [title])
            bar.finish()

    documents = TaggedWikiDocument(wiki)
    print('*****************************************************************************')
    cores = multiprocessing.cpu_count()
    model_wiki = Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=19, iter=10, workers=cores, epochs=30,)
    model_wiki.build_vocab(documents)
    model_wiki.train(documents)
    model_wiki.save("wiki_doc2vec")


#my_model_wiki = doc2vec.Doc2Vec.load("wiki_doc2vec")

"""
# SVM (опорные вектора) ### DOC2VEC
doc2vec_text = []
X = []
for i in range(len(full_articles)):
    doc2vec_text.append(full_articles[i].split(' '))
    X.append(model_doc2vec.infer_vector(doc2vec_text[i]))
y = full_target # метки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# кросс - валидация ***************************************************************************************************
scores = cross_val_score(clf, X, y, cv=10)
print(scores.mean()) # ошибка анализатора

pylab.subplot(1, 2, 1)
lr_probs = clf.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print('DOC2VEC: ROC AUC=%.3f' % lr_auc)
# рассчитываем roc-кривую
fpr, tpr, treshold = roc_curve(y_test, lr_probs)
roc_auc = auc(fpr, tpr)
# строим график
pylab.plot(fpr, tpr, color='darkorange', label='area = %0.2f' % roc_auc)
pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
pylab.title('DOC2VEC')
pylab.legend(loc="lower right")




# SVM (опорные вектора) ### TF IDF

# TF-IDF
vectorizer = TfidfVectorizer(use_idf=True)
tfidf = vectorizer.fit_transform(full_articles)

X = tfidf # значения (tf idf)
y = full_target # метки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

pylab.subplot(1, 2, 2)
lr_probs = clf.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print('TF IDF: ROC AUC=%.3f' % lr_auc)
# рассчитываем roc-кривую
fpr, tpr, treshold = roc_curve(y_test, lr_probs)
roc_auc = auc(fpr, tpr)
# строим график
pylab.plot(fpr, tpr, color='darkorange', label='area = %0.2f' % roc_auc)
pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
pylab.title('TF IDF')
pylab.legend(loc="lower right")

pylab.show()
"""

	    
	    
	    *****************************************************************************************************

import pandas as pd
import re
from stop_words import get_stop_words
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import csv
from itertools import zip_longest
from sklearn.model_selection import cross_val_score
from gensim.models import doc2vec
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import pylab

"""
text = []
data_folder = 'D:/IPMKN/archive/'
emails_raw = pd.read_csv(data_folder + 'Emails.csv', parse_dates=['Id', 'DocNumber', 'MetadataSubject', 'MetadataTo',
                                                                  'MetadataFrom', 'SenderPersonId', 'MetadataDateSent',
                                                                  'MetadataDateReleased', 'MetadataPdfLink',
                                                                  'MetadataCaseNumber', 'MetadataDocumentClass',
                                                                  'ExtractedSubject', 'ExtractedTo', 'ExtractedFrom',
                                                                  'ExtractedCc', 'ExtractedDateSent',
                                                                  'ExtractedCaseNumber', 'ExtractedDocNumber',
                                                                  'ExtractedDateReleased',
                                                                  'ExtractedReleaseInPartOrFull',
                                                                  'ExtractedBodyText', 'RawText'])
# полные тексты писем
emails = emails_raw[['Id', 'RawText']]
stops = list(get_stop_words('en'))
my_emails = []
#for i in range(len(emails.RawText)):



pattern_test = r"\bmay\b|\bMay\b"

def clean_doc(myw, stop):
    final = []
    for w in myw:
        if w.lower() not in stop:
            final.append(w)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([c for c in final if not c.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    while "\n" in final:
        final = final.replace("\n", " ")
    final = final.split(' ')
    return final
full = []
for i in range(len(emails.RawText)):
    my_emails = (emails.RawText[i].split(' '))
    my_emails = clean_doc(my_emails, stops)
    full.append(my_emails)
for myi in range(len(emails.RawText)):
    words = emails.RawText[myi].split(' ')
    words = clean_doc(words, stops)
    # Индекс позиции, итерация вручную
    k = 0
    # Сюда записывать позиции найденных патернов
    inx = []
    # Для всех элементов в списке
    for s in words:
        # Проверка на соответствие регулярному выражению
        match = re.search(pattern_test, s)
        # Если нам подходит
        if match:
            word = match.string
            # Записать номер позиции
            inx.append(k)
        # Следующая позиция
        k += 1

    fragment = []
    if len(inx) != 0:
        for i in range(len(inx)):
            # Количество слов до и после найденного патерна
            rang = 3
            # Ниже вычисление диапазона элементов для нужного фрагмента
            if inx[i]-rang <= 0:
                c1 = 0
            else:
                c1 = inx[i]-rang

            if inx[i]+rang >= len(words)-1:
                c2 = len(words)-1
            else:
                c2 = inx[i]+rang

            for j in range(c1, c2+1):
                fragment.append(words[j])
            text.append(' '.join(fragment))
            fragment.clear()


# первые 100 значений
temp = []
[temp.append(x) for x in text if x not in temp]
text = temp
text = text[1:101]
print(text[0])

# 0 - may - глагол
# 1 - May - месяц
list1 = text
list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

d = [list1, list2]
export_data = zip_longest(*d, fillvalue='')
with open('text.csv', 'w', encoding="utf-8", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("Data", "Class"))
    wr.writerows(export_data)
myfile.close()



csv_text = pd.read_csv('text.csv')
target = csv_text['Class']

def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
       yield doc2vec.TaggedDocument(list_of_words, [i])
data_training = list(tagged_document(full))
model = doc2vec.Doc2Vec(vector_size=50, epochs=30, workers=10)
model.build_vocab(data_training)
model.train(data_training, total_examples=model.corpus_count, epochs=model.epochs)
# print(model.infer_vector(['battlefield', 'the', 'entire', 'trojan', 'army', 'flees', 'behind']))
# print(model.docvecs[0])



# TF-IDF
vectorizer = TfidfVectorizer(use_idf=True)
matrix = vectorizer.fit_transform(text)

# SVM (опорные вектора) ### DOC2VEC
doc_text = []
X = []
for i in range(len(text)):
    doc_text.append(text[i].split(' '))
    X.append(model.infer_vector(doc_text[i]))
#X = model.docvecs # значения (tf idf)
y = target # метки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

pylab.subplot (1, 2, 1)
lr_probs = clf.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print('DOC2VEC: ROC AUC=%.3f' % (lr_auc))
# рассчитываем roc-кривую
fpr, tpr, treshold = roc_curve(y_test, lr_probs)
roc_auc = auc(fpr, tpr)
# строим график
pylab.plot(fpr, tpr, color='darkorange',
         label='area = %0.2f' % roc_auc)
pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
pylab.title('DOC2VEC')
pylab.legend(loc="lower right")
#plt.show()

# SVM (опорные вектора) ### TF IDF
X = matrix # значения (tf idf)
y = target # метки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

pylab.subplot (1, 2, 2)
lr_probs = clf.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print('TF IDF: ROC AUC=%.3f' % (lr_auc))
# рассчитываем roc-кривую
fpr, tpr, treshold = roc_curve(y_test, lr_probs)
roc_auc = auc(fpr, tpr)
# строим график
pylab.plot(fpr, tpr, color='darkorange',
         label='area = %0.2f' % roc_auc)
pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
pylab.title('TF IDF')
pylab.legend(loc="lower right")
#plt.show()

pylab.show()

"""



"""
dataset = api.load("text8")
data = [d for d in dataset]

def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
       yield doc2vec.TaggedDocument(list_of_words, [i])
data_training = list(tagged_document(data))
model_doc2vec = doc2vec.Doc2Vec(vector_size=50, epochs=30, workers=10)
model_doc2vec.build_vocab(data_training)
model_doc2vec.train(data_training, total_examples=model_doc2vec.corpus_count, epochs=model_doc2vec.epochs)
model_doc2vec.save("my_doc2vec")
"""



"""

dataset = api.load("text8")
data = [d for d in dataset]
print(data[0])

def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
       yield doc2vec.TaggedDocument(list_of_words, [i])
data_training = list(tagged_document(data))
model = doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=100, workers=10)
model.build_vocab(data_training)
model.train(data_training, total_examples=model.corpus_count, epochs=model.epochs)
# print(model.infer_vector(['battlefield', 'the', 'entire', 'trojan', 'army', 'flees', 'behind']))
# print(model.docvecs[0])

#model.save("my_doc2vec_model")

# doc2vec
model = doc2vec.Doc2Vec.load("my_doc2vec_model")

# TF-IDF
vectorizer = TfidfVectorizer(use_idf=True)
full_data = []
for i in range(1701):
    full_data.append(' '.join(data[i]))
matrix = vectorizer.fit_transform(full_data)


# SVM (опорные вектора) ###1
X = model.docvecs # значения doc2vec
# y = [random.randint(0, 1) for i in range(1701)] # метки РАНДОМНЫЕ
# np.save("Test y", np.array(y))
y = list(np.load("Test y.npy")) # метки РАНДОМНЫЕ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('*************************************************************************')

# SVM (опорные вектора) ###2
X = matrix # значения tf idf
# y = [random.randint(0, 1) for i in range(1701)] # метки РАНДОМНЫЕ
# np.save("Test y", np.array(y))
y = list(np.load("Test y.npy")) # метки РАНДОМНЫЕ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

"""















# кросс - валидация
#scores = cross_val_score(clf, X, y, cv=10)
#print(scores.mean()) # ошибка анализатора


# -*- coding: utf-8 -*-

from random import randint

# Доработать практическую часть урока lesson_007/python_snippets/08_practice.py

# Необходимо создать класс кота. У кота есть аттрибуты - сытость и дом (в котором он живет).
# Кот живет с человеком в доме.
# Для кота дом характеризируется - миской для еды и грязью.
# Изначально в доме нет еды для кота и нет грязи.

# Доработать класс человека, добавив методы
#   подобрать кота - у кота появляется дом.
#   купить коту еды - кошачья еда в доме увеличивается на 50, деньги уменьшаются на 50.
#   убраться в доме - степень грязи в доме уменьшается на 100, сытость у человека уменьшается на 20.
# Увеличить кол-во зарабатываемых человеком денег до 150 (он выучил пайтон и устроился на хорошую работу :)

# Кот может есть, спать и драть обои - необходимо реализовать соответствующие методы.
# Когда кот спит - сытость уменьшается на 10
# Когда кот ест - сытость увеличивается на 20, кошачья еда в доме уменьшается на 10.
# Когда кот дерет обои - сытость уменьшается на 10, степень грязи в доме увеличивается на 5
# Если степень сытости < 0, кот умирает.
# Так же надо реализовать метод "действуй" для кота, в котором он принимает решение
# что будет делать сегодня

# Человеку и коту надо вместе прожить 365 дней.
from termcolor import cprint


class Man:

    def __init__(self, name):
        self.name = name
        self.fullness = 50
        self.house = None

    def __str__(self):
        return '{}: сытость {}'.format(
            self.name, self.fullness)

    def eat(self):
        if self.house.food >= 10:
            cprint('{} поел'.format(self.name), color='yellow')
            self.fullness += 10
            self.house.food -= 10
        else:
            cprint('{} нет еды'.format(self.name), color='red')

    def work(self):
        cprint('{} сходил на работу'.format(self.name), color='blue')
        self.house.money += 150
        self.fullness -= 10

    def watch_mtv(self):
        cprint('{} смотрел MTV целый день'.format(self.name), color='green')
        self.fullness -= 10

    def shopping(self):
        if self.house.money >= 50:
            cprint('{} сходил в магазин за едой'.format(self.name), color='magenta')
            self.house.money -= 50
            self.house.food += 50
        else:
            cprint('{} деньги кончились!'.format(self.name), color='red')

    def go_to_the_house(self, house):
        self.house = house
        self.fullness -= 10
        cprint('{} Вьехал в дом'.format(self.name), color='cyan')

    def pick_up_a_cat(self, cat):
        cat.house = self.house
        cprint(f'{self.name} подобрал {cat.name}', color='white')

    def buy_cat_food(self):
        if self.house.money >= 50:
            cprint(f'{self.name} сходил за едой коту', color='magenta')
            self.house.cat_food += 50
            self.house.money -= 50
        else:
            cprint(f'{self.name} деньги кончились!', color='red')

    def cleaning(self):
        if self.house.mud >= 100:
            cprint(f'{self.name} убрался в доме', color='green')
            self.house.mud -= 100
            self.fullness -= 20
        else:
            cprint(f'{self.name}! В квартире очень грязно!', color='red')

    def act(self):
        if self.fullness <= 0:
            cprint('{} умер...'.format(self.name), color='red')
            return
        dice = randint(1, 6)
        if self.house.money < 50:
            self.work()
        elif self.fullness <= 20:
            self.eat()
        elif self.house.food < 20:
            self.shopping()
        elif self.house.cat_food <= 20:
            self.buy_cat_food()
        elif self.house.mud >= 100:
            self.cleaning()
        elif dice == 1:
            self.work()
        elif dice == 2:
            self.eat()
        elif dice == 3:
            self.buy_cat_food()
        else:
            self.watch_mtv()


class Cat:

    def __init__(self, name):
        self.name = name
        self.cat_fullness = 50
        self.house = None

    def __str__(self):
        return f'{self.name}: сытость {self.cat_fullness}'

    def cat_sleep(self):
        self.cat_fullness -= 10
        cprint(f'{self.name} спал весь день', color='green')

    def go_to_the_house(self, house):
        self.house = house
        self.cat_fullness -= 10
        cprint(f'{self.name} въехал в дом', color='cyan')

    def eat(self):
        if self.house.cat_food >= 0:
            self.cat_fullness += 20
            self.house.cat_food -= 10
            cprint(f'{self.name} поел', color='green')
        else:
            cprint(f'У котиков нет еды', color='red')

    def destroy_wallpaper(self):
        self.cat_fullness -= 10
        self.house.mud += 5
        cprint(f'{self.name} драл обои', color='green')

    def act(self):
        if self.cat_fullness <= 0:
            cprint(f'{self.name} умер...', color='red')
            return
        dice_cat = randint(1, 6)
        if self.cat_fullness < 20:
            self.eat()
        elif dice_cat == 1:
            self.destroy_wallpaper()
        elif dice_cat == 2:
            self.eat()
        else:
            self.cat_sleep()


class House:

    def __init__(self):
        self.food = 50
        self.money = 0
        self.cat_food = 0
        self.mud = 0

    def __str__(self):
        return f'В доме:\nеды - {self.food}\nденег - {self.money}\nкошачей еды - {self.cat_food}\nзагрязненность - {self.mud}'


citizens = [
    Man(name='Вася'),
]

cats = [
    Cat(name='Ешка'),
    Cat(name='Гриндерс'),
]

for citizens_cats in cats:
    citizens[0].pick_up_a_cat(cat=citizens_cats)
    citizens.append(citizens_cats)

my_sweet_home = House()

for citisen in citizens:
    citisen.go_to_the_house(house=my_sweet_home)

for day in range(1, 366):
    print('================ день {} ================'.format(day))
    for citisen in citizens:
        citisen.act()
    print('---------------- в конце дня ----------------')
    for citisen in citizens:
        print(citisen)
    print(my_sweet_home)

# Усложненное задание (делать по желанию)
# Создать несколько (2-3) котов и подселить их в дом к человеку.
# Им всем вместе так же надо прожить 365 дней.

# (Можно определить критическое количество котов, которое может прокормить человек...)









