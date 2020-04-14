BUILD_DIR := $(shell pwd)/build
XYONIX_KERNEL=xyonix-flu
VENV_DIR := ${BUILD_DIR}/venv/${XYONIX_KERNEL}
NBEATS_DIR := ${BUILD_DIR}/src/n-beats

all: clean install jupyter-kernel

install-venv: ${VENV_DIR}/bin/activate

${VENV_DIR}/bin/activate: requirements.txt
	test -d "${VENV_DIR}" || python3 -m venv ${VENV_DIR}
	. ${VENV_DIR}/bin/activate; pip install -Ur requirements.txt
	@touch ${VENV_DIR}/bin/activate

clean:
	@rm -rf ${BUILD_DIR} __pycache__
	@rm -f nbeats-pytorch

install: install-venv nbeats-source ${BUILD_DIR}/.nbeats-pytorch

nbeats-source:
	@mkdir -p ${BUILD_DIR}/src
	@test -d "${NBEATS_DIR}" || git clone https://github.com/philipperemy/n-beats.git ${NBEATS_DIR} && \
	cd ${NBEATS_DIR} && git checkout -q ceeb6f4f88592c8226bf7a0151d303aad22714f7

${BUILD_DIR}/.nbeats-pytorch:
	. ${VENV_DIR}/bin/activate && cd ${NBEATS_DIR} && $(MAKE) install-pytorch
	@touch ${BUILD_DIR}/.nbeats-pytorch

jupyter-kernel: install-venv
	${VENV_DIR}/bin/ipython kernel install --user --name=${XYONIX_KERNEL}

run-jupyter: jupyter-kernel
	${VENV_DIR}/bin/jupyter notebook flu-forecasting.ipynb

lint:
	${VENV_DIR}/bin/pylint *.py


