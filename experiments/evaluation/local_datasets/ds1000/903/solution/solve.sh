#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
vectorizer = CountVectorizer(stop_words="english", binary=True, lowercase=False,
                             vocabulary=['Jscript', '.Net', 'TypeScript', 'NodeJS', 'Angular', 'Mongo',
                                         'CSS',
                                         'Python', 'PHP', 'Photoshop', 'Oracle', 'Linux', 'C++', "Java", 'TeamCity',
                                         'Frontend', 'Backend', 'Full stack', 'UI Design', 'Web', 'Integration',
                                         'Database design', 'UX'])
X = vectorizer.fit_transform(corpus).toarray()
feature_names = vectorizer.get_feature_names_out()

PY
