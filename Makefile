VENV_NAME=nn_forecast

.PHONY: install clean

install:
	python -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

clean:
	rm -rf $(VENV_NAME)

# To polecenie tylko pokazuje, jak aktywować środowisko
activate:
	@echo "Aby aktywowac srodowisko, wpisz:"
	@echo ".\$(VENV_NAME)\Scripts\Activate.ps1"