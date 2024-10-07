FROM jupyter/tensorflow-notebook:python-3.10@sha256:e0dcbd81e61ed43523558a3444e9e98a44f46ddd18216e3061973783bd261253
# NOTE: This is the most recent version of jupyter/scipy-notebook:python-3.10
# As of 11 January 2023

# System packages
USER root
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim", "lmodern", "libsuitesparse-dev"]

# Python packages
ENV REQUIREMENTS_LOC=/rome-requirements.txt
COPY ./docker/requirements.txt ${REQUIREMENTS_LOC}
RUN pip install --upgrade pip
RUN pip install -r ${REQUIREMENTS_LOC}

# Configure some files
ENV RUN_MAKE_LOC=/run_make.sh
COPY ./run_make.sh ${RUN_MAKE_LOC}
RUN chmod 777 ${RUN_MAKE_LOC}
ENV PYTHONPATH=${PYTHONPATH}:/home/jovyan

# Entrypoint
# ENTRYPOINT [ "/run_make.sh" ]