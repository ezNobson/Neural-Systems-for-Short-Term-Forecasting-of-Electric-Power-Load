VENV_NAME=nn_forecast

.PHONY: install clean

install:
	python -m venv $(VENV_NAME)
	$(VENV_NAME)\Scripts\pip.exe install --upgrade pip
	$(VENV_NAME)\Scripts\pip.exe install -r requirements.txt

clean:
	rm -rf $(VENV_NAME)

# To polecenie tylko pokazuje, jak aktywować środowisko
activate:
	@echo "Aby aktywowac srodowisko, wpisz:"
	@echo ".\$(VENV_NAME)\Scripts\Activate.ps1"