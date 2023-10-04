import spacy
nlp=spacy.load("en_core_web_md")
class Category:
    bio="bio"
    maths="maths"
train_x=["activity of cell","calculate a projectile","activity of a laser","understand metabolism","forces of a system","organ systems"]
train_y=[Category.bio ,Category.maths ,Category.maths ,Category.bio ,Category.maths ,Category.bio]

docs=[nlp(text)for text in train_x]
train_x_word_vectors= [x.vector for x in docs]

test_x=["activity of cell","understand laser fundamentals","molecular biology","power systems"]
test_docs=[nlp(text)for text in test_x]
test_x_word_vectors= [x.vector for x in test_docs]
from sklearn import svm
clf_svm=svm.SVC(kernel='linear')
clf_svm.fit(train_x_word_vectors, train_y)
r=clf_svm.predict(test_x_word_vectors)
print(r)

