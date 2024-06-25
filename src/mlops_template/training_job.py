import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.local import LocalSession
import os
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
import time
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep, CreateModelStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda

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

# PyTorch Estimatorの作成
estimator = Estimator(entry_point='train.py',  # 上記のPyTorch学習コードをtrain.pyとして保存
                    role=role,
                    image_uri='763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker',
                    instance_count=1,
                    instance_type='ml.g4dn.xlarge',
                    #instance_type = "local",
                    output_path=f's3://mlops-flower-photos/output/',
                    sagemaker_session=pipeline_session,)

# トレーニングジョブの開始
s3_dir = 's3://mlops-flower-photos/input/ver3/'


training_step = TrainingStep(
    name="ImageClassificationTraining",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data= os.path.join(s3_dir, 'train'),
            content_type='x-image'
        )
        ,
        "val": TrainingInput(
            s3_data= os.path.join(s3_dir, 'val'),
            content_type='x-image'
        ),
        "test": TrainingInput(
            s3_data= os.path.join(s3_dir, 'test'),
            content_type='x-image'
        )
    }
)

endpoint_image_uri = '763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker'

# モデルの登録
model = Model(model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
              image_uri=endpoint_image_uri,
              role=role,
              sagemaker_session=pipeline_session)

model_step = ModelStep(
    name="CreateModel",
    step_args=model.create(instance_type="ml.g4.xlarge",accelerator_type="ml.eia1.medium")
    )

lambda_function = Lambda(
    function_arn="arn:aws:lambda:ap-northeast-1:770693421928:layer:Klayers-python39-pytorch:1",
    session=sagemaker_session
)
update_endpoint_step = LambdaStep(
    name="UpdateEndpoint",
    lambda_func=lambda_function,
    inputs={"ModelName": model_step.properties.ModelName, "EndpointName": "sample_endpoint"}
)

pipeline = Pipeline(
    name="TrainingPipeline",
    steps=[training_step, model_step],
)

# パイプラインの定義を作成または更新
pipeline.upsert(role_arn=role)
pipeline.start()