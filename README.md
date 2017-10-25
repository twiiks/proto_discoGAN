## proto_discoGAN

### Motivation
- How to lower the dataset for learning
- How to quickly learn user's own style

### Approach #4
- If you can learn the relation between one character(for example '가') and another(for example '나'), you would generate your own styled '나' with your input '가'


## Structure
- **written2all.py** : get input as {unicode_input : path_to_input_image} and return output as {unicode_output : path_to_output_image} after editting the raw images
- **reference** : implemented code for discoGAN


## Limits
- Many different styled fonts interrupt learning general shape of characters generating noise
- lack of free Korean character font


## Example & Screenshot
![screenshot1](https://s3.ap-northeast-2.amazonaws.com/fontto/repository-images/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA+2017-10-24+%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE+2.57.17.png =200*200)


## Reference
> [discoGAN](https://github.com/SKTBrain/DiscoGAN)
> [paper](https://arxiv.org/abs/1703.05192)
written2all 프로그램을 직접 구동할 수 있습니다.
아래 코드는 미리 학습시켜놓은 데이터를 이용해 데모를 진행합니다.

bash-3.2$ exit

[No write since last change]
bash-3.2$ cat disgoGAN.md
## proco_discoGAN

### Motivation
- How to lower the dataset for learning
- How to quickly learn user's own style

### Approach #4
- If you can learn the relation between one character(for example '가') and another(for example '나'), you would generate your own styled '나' with your input '가'


## Structure
- **written2all.py** : get input as {unicode_input : path_to_input_image} and return output as {unicode_output : path_to_output_image} after editting the raw images
- **reference** : implemented code for discoGAN


## Limits
- Many different styled fonts interrupt learning general shape of characters generating noise
- lack of free Korean character font


## Example & Screenshot
<img src='https://s3.ap-northeast-2.amazonaws.com/fontto/repository-images/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA+2017-10-24+%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE+2.57.17.png' width='200px' height='200px' /> <img src='https://s3.ap-northeast-2.amazonaws.com/fontto/repository-images/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA+2017-10-24+%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE+2.57.27.png' width='200px' height='200px' />


## Reference
> [discoGAN](https://github.com/SKTBrain/DiscoGAN)
> [paper](https://arxiv.org/abs/1703.05192)
