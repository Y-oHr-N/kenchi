all: clean code-analysis test release

clean:
	python setup.py clean
	rm -rf dist

code-analysis:
	flake8 kenchi

release:
	python setup.py sdist bdist_wheel

test:
	python setup.py test

upload:
	twine upload dist/*
