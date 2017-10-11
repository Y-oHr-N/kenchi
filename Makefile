PYTHON := python

all: clean code-analysis test release

clean:
	${PYTHON} setup.py clean
	rm -rf dist

code-analysis:
	flake8 kenchi

release:
	${PYTHON} setup.py sdist bdist_wheel

test:
	${PYTHON} setup.py test

upload:
	twine upload dist/*
