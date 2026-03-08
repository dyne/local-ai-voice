SHELL := /bin/sh
PY311 := $(shell command -v python3.11 2>/dev/null)
ifeq ($(OS),Windows_NT)
HOST_PYTHON ?= py -3.11
VENV_PYTHON := .venv/Scripts/python.exe
DATA_SEP := ;
else ifneq ($(PY311),)
HOST_PYTHON ?= python3.11
VENV_PYTHON := .venv/bin/python
DATA_SEP := :
else
HOST_PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
DATA_SEP := :
endif
PYTHON ?= $(VENV_PYTHON)
PIP := $(PYTHON) -m pip

SCRIPT := local-ai-voice.py
APP_NAME := local-ai-voice
SPEC := $(APP_NAME).spec
REQUIREMENTS := numpy noisereduce webrtcvad-wheels sounddevice fastapi uvicorn websockets pydantic av huggingface_hub[cli]
OPENVINO_PACKAGES := openvino openvino-genai openvino-tokenizers

.PHONY: all install install-build spec build build-webrtc run run-web clean

all: install

$(VENV_PYTHON):
	$(HOST_PYTHON) -m venv .venv

install: $(VENV_PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install $(REQUIREMENTS)
	$(PIP) install --upgrade $(OPENVINO_PACKAGES)

install-build: $(VENV_PYTHON)
	$(PIP) install pyinstaller

spec: install-build
	$(PYTHON) -m PyInstaller.utils.cliutils.makespec --onefile --name $(APP_NAME) \
		--hidden-import browser_webrtc \
		--hidden-import av \
		--hidden-import noisereduce \
		--hidden-import webrtcvad \
		--additional-hooks-dir hooks \
		--collect-binaries openvino \
		--collect-data openvino \
		--collect-binaries openvino_genai \
		--collect-data openvino_genai \
		--collect-binaries openvino_tokenizers \
		--collect-data openvino_tokenizers \
		--add-data "web/index.html$(DATA_SEP)web" \
		$(SCRIPT)

build: install spec
	$(PYTHON) -m PyInstaller --clean $(SPEC)

build-webrtc: build

run:
	$(PYTHON) $(SCRIPT)

run-web:
	$(PYTHON) $(SCRIPT) --web

clean:
	$(RM) -r build dist __pycache__ *.spec
