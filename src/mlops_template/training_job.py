import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.local import LocalSession
import os

session = boto3.Session()
default_bucket = 'sample-mnist'
os.environ['AWS_DEFAULT_REGION'] = 'ap-northeast-1'
sagemaker_session = LocalSession(boto_session=session, default_bucket="sample-mnist")
s3_client = session.client(service_name="s3", region_name="ap-northeast-1")

# ロールの設定
role = 'arn:aws:iam::590184069860:user/nagaoka'

# PyTorch Estimatorの作成
estimator = PyTorch(entry_point='train.py',  # 上記のPyTorch学習コードをtrain.pyとして保存
                    role=role,
                    framework_version='1.8.0',
                    py_version='py3',
                    instance_count=1,
                    #instance_type='ml.m5.large',
                    instance_type = "local",
                    output_path=f's3://{default_bucket}/output',
                    sagemaker_session=sagemaker_session,)

# トレーニングジョブの開始
estimator.fit({'training': f's3://sample-mnist/input'})


# モデルの登録
model = Model(model_data=estimator.model_data,
              image_uri=estimator.image_uri,
              role=role,
              sagemaker_session=sagemaker_session)

model_package_group_name = 'MNISTModelPackageGroup'
model_package_description = 'MNIST PyTorch model package'

# モデルパッケージの作成
model_package = sagemaker_session.create_model_package_group(model=model,
                                    model_package_group_name=model_package_group_name,
                                    model_package_description=model_package_description)

# モデルパッケージをモデルレジストリに登録
model_package.register()