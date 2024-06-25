import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.local import LocalSession
import os
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
import time


os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-1'
region = os.environ['AWS_DEFAULT_REGION']
session = boto3.Session()
s3_client = session.client(service_name="s3", region_name=region)

local = False
if local:
    sagemaker_session = LocalSession(boto_session=session)
    role = 'arn:aws:iam::590184069860:role/mlops-template' # arn:aws:iam::590184069860:role/mlops-template
else:
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = 'arn:aws:iam::590184069860:role/mlops-template'



endpoint_image_uri = '763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker'
model_url = 's3://mlops-flower-photos/output/pytorch-training-2024-06-18-12-13-49-815/output/model.tar.gz'

model_package_info = {
 "ModelPackageGroupName" : 'FlowerModelPackageGroup', 
 "ModelPackageGroupDescription" : 'Flower PyTorch model package'
}

# モデルパッケージの作成
sm_client = boto3.client('sagemaker', region_name=region)

# modelを model registry に登録
model_package = sm_client.create_model_package(
    ModelPackageGroupName='FlowerModelPackageGroup',
    ModelPackageDescription='Flower PyTorch model package',
    InferenceSpecification={
        'Containers': [
            {
                'Image': endpoint_image_uri,
                'ModelDataUrl': model_url,
            },
        ],
        'SupportedContentTypes': ['application/json'],
        'SupportedRealtimeInferenceInstanceTypes': ['ml.g4dn.xlarge'],
        'SupportedTransformInstanceTypes': ['ml.g4dn.xlarge'],
        'SupportedResponseMIMETypes': ['application/json'],
    },
    ModelApprovalStatus='Approved',
)