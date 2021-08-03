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
