# IDENTIFYING VIOLENCE IN VIDEOS USING HIERARCHICAL ATTENTION NETWORKS

Tamir Arbel, Nikita Shavit Dydo and Shay Doner.
Afeka College of Engineering, Tel Aviv, Israel.


## This GIT

This branch contains the Keras implementation of our work, which is a novel approach to detecting violence in videos by using hierarchical attention networks.

## Abstract

Detecting violence in videos is a well known problem that has various solutions. This work presents a novel approach to detecting violence in videos by using hierarchical attention networks (HAN) to analyze the context of each frame in relation to previous and future frames. The study evaluates the method using three existing violent behavior recognition datasets. In addition, ViTPose was chosen as the pose estimator due to its high accuracy and efficiency. The videos were transformed into a series of pose vectors, which were saved as the input for the HAN model. The HAN model was implemented using the Keras interface for TensorFlow framework, with modifications to receive pose estimation vectors instead of word embedding. The results showed improved accuracy in violent behavior recognition compared to previous methods, demonstrating the potential for this approach to contribute to the field of violence detection in videos.
The Full article can be read [HERE](https://github.com/shay681/Violence-Detection/blob/main/Identifying%20violence%20in%20videos%20using%20hierarchical%20attention%20networks.pdf).

## Usage
###  Crete Data
```mermaid
graph LR
A[Create-pose estimations of your data set using VitPose] ----> B[Save the pose-estimations as H5 files]
```
### Use Code
```mermaid
graph LR
A[DATA] ----> C[Train a model with train.py]
A ----> B
B[Use grid-search and cross validation to tune your parameters] ----> C[Train a model with train.py]
```
### Evaluation
```mermaid
graph LR
A[Model] ----> C[Test your model with test.py] ----> D[Improve the logic, have fun, and don't forget to credit us]
```
## Notes

 1. The code may need some alteration.
 2. Use TensoFlow environment with GPU support to speed up the learning process.
 3. Use requirements.txt to create a conda environment (may change with time).

### requirements.txt

    platform: win-64
    absl-py=1.4.0=pypi_0
    asttokens=2.0.5=pyhd3eb1b0_0
    astunparse=1.6.3=pypi_0
    backcall=0.2.0=pyhd3eb1b0_0
    ca-certificates=2023.01.10=haa95532_0
    cachetools=5.3.0=pypi_0
    certifi=2022.12.7=py39haa95532_0
    charset-normalizer=3.0.1=pypi_0
    colorama=0.4.6=py39haa95532_0
    comm=0.1.2=py39haa95532_0
    contourpy=1.0.7=pypi_0
    cudatoolkit=11.2.2=h933977f_10
    cudnn=8.1.0.77=h3e0f4f4_0
    cycler=0.11.0=pypi_0
    debugpy=1.5.1=py39hd77b12b_0
    decorator=5.1.1=pyhd3eb1b0_0
    entrypoints=0.4=py39haa95532_0
    executing=0.8.3=pyhd3eb1b0_0
    flatbuffers=23.1.21=pypi_0
    fonttools=4.38.0=pypi_0
    gast=0.4.0=pypi_0
    google-auth=2.16.0=pypi_0
    google-auth-oauthlib=0.4.6=pypi_0
    google-pasta=0.2.0=pypi_0
    grpcio=1.51.1=pypi_0
    h5py=3.8.0=pypi_0
    idna=3.4=pypi_0
    importlib-metadata=6.0.0=pypi_0
    ipykernel=6.19.2=py39hd4e2768_0
    ipython=8.8.0=py39haa95532_0
    jedi=0.18.1=py39haa95532_1
    joblib=1.2.0=pypi_0
    jupyter_client=7.4.8=py39haa95532_0
    jupyter_core=5.1.1=py39haa95532_0
    keras=2.10.0=pypi_0
    keras-preprocessing=1.1.2=pypi_0
    kiwisolver=1.4.4=pypi_0
    libclang=15.0.6.1=pypi_0
    libsodium=1.0.18=h62dcd97_0
    markdown=3.4.1=pypi_0
    markupsafe=2.1.2=pypi_0
    matplotlib=3.6.3=pypi_0
    matplotlib-inline=0.1.6=py39haa95532_0
    nest-asyncio=1.5.6=py39haa95532_0
    numpy=1.24.1=pypi_0
    oauthlib=3.2.2=pypi_0
    openssl=1.1.1s=h2bbff1b_0
    opt-einsum=3.3.0=pypi_0
    packaging=22.0=py39haa95532_0
    pandas=1.5.3=pypi_0
    parso=0.8.3=pyhd3eb1b0_0
    pickleshare=0.7.5=pyhd3eb1b0_1003
    pillow=9.4.0=pypi_0
    pip=22.3.1=py39haa95532_0
    platformdirs=2.5.2=py39haa95532_0
    prompt-toolkit=3.0.36=py39haa95532_0
    protobuf=3.19.6=pypi_0
    psutil=5.9.0=py39h2bbff1b_0
    pure_eval=0.2.2=pyhd3eb1b0_0
    pyasn1=0.4.8=pypi_0
    pyasn1-modules=0.2.8=pypi_0
    pygments=2.11.2=pyhd3eb1b0_0
    pyparsing=3.0.9=pypi_0
    python=3.9.16=h6244533_0
    python-dateutil=2.8.2=pyhd3eb1b0_0
    pytz=2022.7.1=pypi_0
    pywin32=305=py39h2bbff1b_0
    pyzmq=23.2.0=py39hd77b12b_0
    requests=2.28.2=pypi_0
    requests-oauthlib=1.3.1=pypi_0
    rsa=4.9=pypi_0
    scikeras=0.10.0=pypi_0
    scikit-learn=1.2.1=pypi_0
    scipy=1.10.0=pypi_0
    seaborn=0.12.2=pypi_0
    setuptools=65.6.3=py39haa95532_0
    six=1.16.0=pyhd3eb1b0_1
    sklearn=0.0.post1=pypi_0
    sqlite=3.40.1=h2bbff1b_0
    stack_data=0.2.0=pyhd3eb1b0_0
    tensorboard=2.10.1=pypi_0
    tensorboard-data-server=0.6.1=pypi_0
    tensorboard-plugin-wit=1.8.1=pypi_0
    tensorflow=2.10.1=pypi_0
    tensorflow-estimator=2.10.0=pypi_0
    tensorflow-io-gcs-filesystem=0.30.0=pypi_0
    termcolor=2.2.0=pypi_0
    threadpoolctl=3.1.0=pypi_0
    tornado=6.2=py39h2bbff1b_0
    traitlets=5.7.1=py39haa95532_0
    typing-extensions=4.4.0=pypi_0
    tzdata=2022g=h04d1e81_0
    urllib3=1.26.14=pypi_0
    vc=14.2=h21ff451_1
    vs2015_runtime=14.27.29016=h5e58377_2
    wcwidth=0.2.5=pyhd3eb1b0_0
    werkzeug=2.2.2=pypi_0
    wheel=0.37.1=pyhd3eb1b0_0
    wincertstore=0.2=py39haa95532_2
    wrapt=1.14.1=pypi_0
    zeromq=4.3.4=hd77b12b_0
    zipp=3.12.0=pypi_0

