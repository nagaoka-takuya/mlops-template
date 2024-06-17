import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.local import LocalSession
import os
from sagemaker.estimator import Estimator

session = boto3.Session()
os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-1'
# 
#sagemaker_session = sagemaker.Session(boto_session=session)
sagemaker_session = LocalSession(boto_session=session, default_bucket="sample-mnist")
s3_client = session.client(service_name="s3", region_name="ap-northeast-1")

# get role
#role = sagemaker.get_execution_role()

role = 'arn:aws:iam::590184069860:role/mlops-template'

# PyTorch Estimatorの作成
estimator = Estimator(entry_point='train.py',  # 上記のPyTorch学習コードをtrain.pyとして保存
                    role=role,
                    image_uri='763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker',
                    #framework_version='1.8.0',
                    #py_version='py3',
                    instance_count=1,
                    #instance_type='ml.t3.medium',
                    instance_type = "local",
                    output_path=f's3://mlops-flower-photos/output/',
                    sagemaker_session=sagemaker_session,)

# トレーニングジョブの開始
s3_dir = 's3://mlops-flower-photos/input/ver3/'
estimator.fit({'train': os.path.join(s3_dir, 'train'),
               'val': os.path.join(s3_dir, 'val'),
               'test': os.path.join(s3_dir, 'test')})


# モデルの登録
model = Model(model_data=estimator.model_data,
              image_uri=estimator.image_uri,
              role=role,
              sagemaker_session=sagemaker_session)

model_package_group_name = 'MNISTModelPackageGroup'
model_package_description = 'MNIST PyTorch model package'

'''
# モデルパッケージの作成
model_package = sagemaker_session.create_model_package_group(model=model,
                                    model_package_group_name=model_package_group_name,
                                    model_package_description=model_package_description)

# モデルパッケージをモデルレジストリに登録
model_package.register()
'''

# endpointの作成
predictor = model.deploy(initial_instance_count=1, instance_type='local')

# random データを生成して推論
import numpy as np
import torch
inputs = torch.tensor(np.random.rand(1, 1, 28, 28), dtype=torch.float32)
result = predictor.predict(inputs)
print(result)

# 6. 評価結果の保存
'''
model_package.update_model_package(
    model_package_arn=model_package.model_package_arn,
    model_metrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "Statistics": {
                    "Accuracy": 0.80
                }
            }
        }
    }
)
'''