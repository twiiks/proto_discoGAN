proto_discoGAN
===========================

개요
---------------------------

- 한글 폰트 자동 생성 서비스 [fontto](http://fontto.creatorlink.net/) 의 한글 학습 모델링 연구를 위한 프로젝트입니다.
- [discoGAN](https://arxiv.org/abs/1703.05192)을 참고하여 한글 폰트 자동완성 모델링을 연구합니다.
- 미리 학습된 모델을 pth 파일에서 업로드해 새로운 한글 이미지를 생성합니다.

api
---------------------------

written2all 프로그램은 api로 사용될 수 있도록 함수로 제공됩니다.

- 입력 : written = {unicode_input : path_to_input_image}
- 출력 : output = {unicode_output : path_to_output_image}

실행파일
---------------------------
written2all 프로그램을 직접 구동할 수 있습니다.
아래 코드는 미리 학습시켜놓은 데이터를 이용해 데모를 진행합니다.

    $ python3 written2all.py

라이센스
----------------------------
이 코드의 모든 권리는 twiiks에서 소유합니다.
