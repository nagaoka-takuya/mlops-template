import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.local import LocalSession
import os
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
import time
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import MetricsSource, ModelMetrics

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


pipeline_session = PipelineSession()

print(role)
'''
sagemaker_policies = get_sagemaker_policies_for_iam_role(role)

# 取得したSageMakerポリシーを出力
for policy in sagemaker_policies:
    print(f"Policy Name: {policy['PolicyName']}")
    print(f"Policy Arn: {policy['PolicyArn']}")
    print(f"Statement: {policy['Statement']}")
    print("----")
'''

# PyTorch Estimatorの作成
estimator = Estimator(entry_point='train.py',  # 上記のPyTorch学習コードをtrain.pyとして保存
                    role=role,
                    image_uri='763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker',
                    instance_count=1,
                    instance_type='ml.g4dn.xlarge',
                    #instance_type = "local",
                    output_path=f's3://mlops-flower-photos/output/',
                    sagemaker_session=sagemaker_session,)

# トレーニングジョブの開始
s3_dir = 's3://mlops-flower-photos/input/ver3/'
estimator.fit({'train': os.path.join(s3_dir, 'train'),
               'val': os.path.join(s3_dir, 'val'),
               'test': os.path.join(s3_dir, 'test')})


endpoint_image_uri = '763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker'

# モデルの登録
model = Model(model_data=estimator.model_data,
              image_uri=endpoint_image_uri,
              role=role,
              sagemaker_session=sagemaker_session)

model_package_info = {
 "ModelPackageGroupName" : 'FlowerModelPackageGroup', 
 "ModelPackageGroupDescription" : 'Flower PyTorch model package'
}

# モデルパッケージの作成
sm_client = boto3.client('sagemaker', region_name=region)
#model_package = sm_client.create_model_package_group(**model_package_info)

# モデルパッケージをモデルレジストリに登録
#model_package_version = model_package['ModelPackageGroupArn']

# model deploy
predictor = model.deploy(initial_instance_count=1, instance_type='ml.g4dn.xlarge')


# トレーニングジョブの評価結果を取得
job_description = sagemaker_session.describe_training_job(estimator.latest_training_job.name)
print(job_description)
