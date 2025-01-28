FROM gitlab-master.nvidia.com/dler/evfm/evfm:nov24-ftup

RUN pip install --no-deps albumentations albucore simsimd stringzilla

