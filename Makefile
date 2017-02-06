test :
	python setup.py test

coverage-run :
	coverage run setup.py test

coverage : coverage-run
	coverage report

.PHONY: test
