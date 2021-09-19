
codecheck:
	@isort --check .
	@black --check -S .
	@bandit --recursive .

codeformat:
	@isort .
	@black -S .
