class Category:
    bio="bio"
    maths="maths"
train_x=["activity of cell","calculate a projectile","activity of a laser","understand metabolism","forces of a system","organ systems"]
train_y=[Category.bio ,Category.maths ,Category.maths ,Category.bio ,Category.maths ,Category.bio]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer= CountVectorizer(binary=True, ngram_range=(1,2))# ngram good vs not good
test_vectors_x= vectorizer.fit_transform(train_x)
print(vectorizer.get_feature_names())
print(test_vectors_x.toarray())

from sklearn import svm

clf_svm=svm.SVC(kernel='linear')
clf_svm.fit(test_vectors_x, train_y)
test_x=vectorizer.transform(["understand laser", "cell metabloism system"])
r=clf_svm.predict(test_x)
print(r)


